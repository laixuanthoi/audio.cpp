#include "engine/models/higgs_tts/codec_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

namespace weight_binding = engine::modules::binding;
constexpr size_t kDefaultGraphNodes = 65536;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct HiggsCodecConfig {
    int64_t num_quantizers = 0;
    int64_t codebook_size = 0;
    int64_t codebook_dim = 0;
    int64_t quantized_hidden_size = 0;
    int64_t acoustic_hidden_size = 0;
    int64_t decoder_hidden_size = 0;
    std::vector<int> upsampling_ratios;
    int sample_rate = 24000;
};

struct HiggsCodecQuantizerWeights {
    engine::core::TensorValue codebook_embed;
    engine::modules::LinearWeights project_out;
};

struct HiggsCodecResidualUnitWeights {
    engine::modules::Snake1dWeights snake1;
    engine::modules::Conv1dWeights conv1;
    engine::modules::Snake1dWeights snake2;
    engine::modules::Conv1dWeights conv2;
    int64_t channels = 0;
    int dilation = 1;
};

struct HiggsCodecDecoderBlockWeights {
    engine::modules::Snake1dWeights snake1;
    engine::modules::ConvTranspose1dWeights conv_t1;
    HiggsCodecResidualUnitWeights res_unit1;
    HiggsCodecResidualUnitWeights res_unit2;
    HiggsCodecResidualUnitWeights res_unit3;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int stride = 1;
};

struct HiggsCodecWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    HiggsCodecConfig config;
    std::vector<HiggsCodecQuantizerWeights> quantizers;
    engine::modules::LinearWeights fc2;
    engine::modules::Conv1dWeights conv1;
    std::vector<HiggsCodecDecoderBlockWeights> blocks;
    engine::modules::Snake1dWeights final_snake;
    engine::modules::Conv1dWeights conv2;
};

std::shared_ptr<const HiggsAssets> require_assets(std::shared_ptr<const HiggsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs codec decoder runtime requires assets");
    }
    return assets;
}

engine::core::TensorValue ensure_contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.rfind(prefix, 0) == 0;
}

int parse_trailing_index(std::string_view value, std::string_view prefix, std::string_view suffix) {
    if (!starts_with(value, prefix) || value.size() <= prefix.size() + suffix.size()) {
        return -1;
    }
    const auto middle = value.substr(prefix.size(), value.size() - prefix.size() - suffix.size());
    for (const char ch : middle) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return -1;
        }
    }
    return middle.empty() ? -1 : std::stoi(std::string(middle));
}

HiggsCodecConfig infer_codec_config(const assets::TensorSource & source, int sample_rate) {
    HiggsCodecConfig config;
    config.sample_rate = sample_rate;

    const auto codebook_shape = source.require_metadata("quantizer.quantizers.0.codebook.embed").shape;
    if (codebook_shape.size() != 2) {
        throw std::runtime_error("Higgs codec codebook embed must be rank 2");
    }
    config.codebook_size = codebook_shape[0];
    config.codebook_dim = codebook_shape[1];

    const auto project_out_shape = source.require_metadata("quantizer.quantizers.0.project_out.weight").shape;
    const auto fc2_shape = source.require_metadata("fc2.weight").shape;
    const auto conv1_shape = source.require_metadata("acoustic_decoder.conv1.weight").shape;
    if (project_out_shape.size() != 2 || fc2_shape.size() != 2 || conv1_shape.size() != 3) {
        throw std::runtime_error("Higgs codec core weight anchors have unexpected rank");
    }
    config.quantized_hidden_size = project_out_shape[0];
    config.acoustic_hidden_size = fc2_shape[0];
    config.decoder_hidden_size = conv1_shape[0];

    int max_quantizer = -1;
    int max_block = -1;
    for (const auto & tensor : source.tensors()) {
        max_quantizer = std::max(max_quantizer, parse_trailing_index(
            tensor.name,
            "quantizer.quantizers.",
            ".codebook.embed"));
        max_block = std::max(max_block, parse_trailing_index(
            tensor.name,
            "acoustic_decoder.block.",
            ".conv_t1.weight"));
    }
    config.num_quantizers = max_quantizer + 1;
    if (config.num_quantizers <= 0) {
        throw std::runtime_error("Higgs codec did not expose any quantizer weights");
    }
    config.upsampling_ratios.resize(static_cast<size_t>(max_block + 1));
    for (int block = 0; block <= max_block; ++block) {
        const auto shape = source.require_metadata(
            "acoustic_decoder.block." + std::to_string(block) + ".conv_t1.weight").shape;
        if (shape.size() != 3) {
            throw std::runtime_error("Higgs codec conv_t1 weight must be rank 3");
        }
        config.upsampling_ratios[static_cast<size_t>(block)] = static_cast<int>(shape[2] / 2);
    }
    return config;
}

engine::modules::Snake1dWeights snake_from_named_source(
    engine::core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name) {
    return {weight_binding::f32_tensor_from_named_source(store, source, name)};
}

engine::modules::Snake1dWeights normalize_snake_weights(
    engine::core::ModuleBuildContext & ctx,
    const engine::modules::Snake1dWeights & weights,
    int64_t channels) {
    if (weights.alpha.shape.rank == 3
        && weights.alpha.shape.dims[0] == 1
        && weights.alpha.shape.dims[1] == channels
        && weights.alpha.shape.dims[2] == 1) {
        return {engine::core::reshape_tensor(
            ctx,
            weights.alpha,
            engine::core::TensorShape::from_dims({channels}))};
    }
    return weights;
}

HiggsCodecResidualUnitWeights load_residual_unit_weights(
    engine::core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int dilation,
    engine::assets::TensorStorageType storage_type) {
    return {
        snake_from_named_source(store, source, prefix + ".snake1.alpha"),
        weight_binding::conv1d_from_source(store, source, prefix + ".conv1", storage_type, channels, channels, 7, true),
        snake_from_named_source(store, source, prefix + ".snake2.alpha"),
        weight_binding::conv1d_from_source(store, source, prefix + ".conv2", storage_type, channels, channels, 1, true),
        channels,
        dilation,
    };
}

std::shared_ptr<const HiggsCodecWeights> load_codec_weights(
    const HiggsAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<HiggsCodecWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(), execution_context.backend_type(),
        "higgs_tts.codec.weights", weight_context_bytes);
    auto & store = *weights->store;
    const auto & source = *assets.codec_weights;
    weights->config = infer_codec_config(source, 24000);

    weights->quantizers.reserve(static_cast<size_t>(weights->config.num_quantizers));
    for (int64_t i = 0; i < weights->config.num_quantizers; ++i) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(i);
        weights->quantizers.push_back({
            store.load_tensor(source, prefix + ".codebook.embed", storage_type, {weights->config.codebook_size, weights->config.codebook_dim}),
            weight_binding::linear_from_source(store, source, prefix + ".project_out", storage_type, weights->config.quantized_hidden_size, weights->config.codebook_dim, true),
        });
    }

    weights->fc2 = weight_binding::linear_from_source(
        store, source, "fc2", storage_type,
        weights->config.acoustic_hidden_size,
        weights->config.quantized_hidden_size,
        true);
    weights->conv1 = weight_binding::conv1d_from_source(
        store, source, "acoustic_decoder.conv1", storage_type,
        weights->config.decoder_hidden_size,
        weights->config.acoustic_hidden_size,
        7,
        true);

    weights->blocks.reserve(weights->config.upsampling_ratios.size());
    int64_t in_channels = weights->config.decoder_hidden_size;
    for (size_t i = 0; i < weights->config.upsampling_ratios.size(); ++i) {
        const int stride = weights->config.upsampling_ratios[i];
        const int64_t out_channels = in_channels / 2;
        const std::string prefix = "acoustic_decoder.block." + std::to_string(i);
        HiggsCodecDecoderBlockWeights block;
        block.snake1 = snake_from_named_source(store, source, prefix + ".snake1.alpha");
        block.conv_t1 = weight_binding::conv_transpose1d_from_source(
            store, source, prefix + ".conv_t1", storage_type,
            in_channels, out_channels, stride * 2, true);
        block.res_unit1 = load_residual_unit_weights(store, source, prefix + ".res_unit1", out_channels, 1, storage_type);
        block.res_unit2 = load_residual_unit_weights(store, source, prefix + ".res_unit2", out_channels, 3, storage_type);
        block.res_unit3 = load_residual_unit_weights(store, source, prefix + ".res_unit3", out_channels, 9, storage_type);
        block.in_channels = in_channels;
        block.out_channels = out_channels;
        block.stride = stride;
        weights->blocks.push_back(std::move(block));
        in_channels = out_channels;
    }
    weights->final_snake = snake_from_named_source(store, source, "acoustic_decoder.snake1.alpha");
    weights->conv2 = weight_binding::conv1d_from_source(
        store, source, "acoustic_decoder.conv2", storage_type,
        1, in_channels, 7, true);
    weights->store->upload();
    return weights;
}

engine::core::TensorValue maybe_center_crop(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t target_frames) {
    if (input.shape.dims[2] == target_frames) {
        return input;
    }
    if (input.shape.dims[2] < target_frames) {
        throw std::runtime_error("Higgs codec residual unit cannot grow the residual branch");
    }
    const int64_t start = (input.shape.dims[2] - target_frames) / 2;
    return engine::modules::SliceModule({2, start, target_frames}).build(ctx, input);
}

engine::core::TensorValue build_residual_unit(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const HiggsCodecResidualUnitWeights & weights) {
    auto x = engine::modules::Snake1dModule({weights.channels}).build(ctx, input, normalize_snake_weights(ctx, weights.snake1, weights.channels));
    x = engine::modules::Conv1dModule({weights.channels, weights.channels, 7, 1, ((7 - 1) / 2) * weights.dilation, weights.dilation, true})
        .build(ctx, x, weights.conv1);
    x = engine::modules::Snake1dModule({weights.channels}).build(ctx, x, normalize_snake_weights(ctx, weights.snake2, weights.channels));
    x = engine::modules::Conv1dModule({weights.channels, weights.channels, 1, 1, 0, 1, true})
        .build(ctx, x, weights.conv2);
    auto residual = maybe_center_crop(ctx, input, x.shape.dims[2]);
    return engine::modules::AddModule{}.build(ctx, residual, x);
}

engine::core::TensorValue build_decoder_block(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const HiggsCodecDecoderBlockWeights & weights) {
    auto x = engine::modules::Snake1dModule({weights.in_channels}).build(ctx, input, normalize_snake_weights(ctx, weights.snake1, weights.in_channels));
    x = engine::modules::ConvTranspose1dModule({weights.in_channels, weights.out_channels, weights.stride * 2, weights.stride, 0, 1, true})
        .build(ctx, x, weights.conv_t1);
    const int64_t target_frames = input.shape.dims[2] * weights.stride;
    const int64_t crop_start = (weights.stride + 1) / 2;
    x = engine::modules::SliceModule({2, crop_start, target_frames}).build(ctx, x);
    x = build_residual_unit(ctx, x, weights.res_unit1);
    x = build_residual_unit(ctx, x, weights.res_unit2);
    x = build_residual_unit(ctx, x, weights.res_unit3);
    return x;
}

}  // namespace

class HiggsCodecDecoderRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const HiggsAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType weight_storage_type)
        : assets_(require_assets(std::move(assets)))
        , backend_(execution_context.backend())
        , threads_(std::max(1, execution_context.config().threads))
        , graph_context_bytes_(graph_context_bytes)
        , weights_(load_codec_weights(*assets_, execution_context, weight_context_bytes, weight_storage_type)) {
        if (graph_context_bytes_ == 0) {
            throw std::runtime_error("Higgs codec graph context bytes must be non-zero");
        }
    }

    ~Impl() {
        release_graph();
    }

    runtime::AudioBuffer decode(const HiggsCodeSequence & raw_codes) {
        if (raw_codes.num_codebooks != weights_->config.num_quantizers) {
            throw std::runtime_error("Higgs codec decoder codebook count mismatch");
        }
        if (raw_codes.frames <= 0 || raw_codes.values.empty()) {
            throw std::runtime_error("Higgs codec decoder requires non-empty raw codes");
        }
        if (sequence_steps_ != raw_codes.frames) {
            build(raw_codes.frames);
        }
        ggml_backend_tensor_set(input_codes_, raw_codes.values.data(), 0, raw_codes.values.size() * sizeof(int32_t));
        engine::core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs codec decoder graph compute failed");
        }
        std::vector<float> samples(static_cast<size_t>(output_frames_), 0.0F);
        ggml_backend_tensor_get(output_audio_, samples.data(), 0, samples.size() * sizeof(float));
        return runtime::AudioBuffer{weights_->config.sample_rate, 1, std::move(samples)};
    }

private:
    void release_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        graph_ = nullptr;
        input_codes_ = nullptr;
        output_audio_ = nullptr;
        ctx_.reset();
        sequence_steps_ = 0;
        output_frames_ = 0;
    }

    void build(int64_t steps) {
        release_graph();
        ggml_init_params params{graph_context_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs codec graph context");
        }
        const auto & config = weights_->config;
        engine::core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.codec_decode"};
        input_codes_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_I32, config.num_quantizers, steps, 1);
        auto codes = engine::core::wrap_tensor(
            input_codes_,
            engine::core::TensorShape::from_dims({1, steps, config.num_quantizers}),
            GGML_TYPE_I32);

        engine::core::TensorValue quantized;
        bool has_quantized = false;
        for (int64_t i = 0; i < config.num_quantizers; ++i) {
            auto indices = engine::modules::SliceModule({2, i, 1}).build(ctx, codes);
            indices = engine::core::reshape_tensor(
                ctx,
                engine::core::ensure_backend_addressable_layout(ctx, indices),
                engine::core::TensorShape::from_dims({1, steps}));
            auto embedded = engine::modules::EmbeddingModule({config.codebook_size, config.codebook_dim})
                .build(ctx, indices, weights_->quantizers[static_cast<size_t>(i)].codebook_embed);
            auto projected = engine::modules::LinearModule({config.codebook_dim, config.quantized_hidden_size, true, GGML_PREC_DEFAULT})
                .build(ctx, embedded, weights_->quantizers[static_cast<size_t>(i)].project_out);
            if (!has_quantized) {
                quantized = projected;
                has_quantized = true;
            } else {
                quantized = engine::modules::AddModule{}.build(ctx, quantized, projected);
            }
        }
        if (!has_quantized) {
            throw std::runtime_error("Higgs codec graph has no quantizers");
        }

        auto quantized_acoustic = engine::modules::LinearModule({config.quantized_hidden_size, config.acoustic_hidden_size, true, GGML_PREC_DEFAULT})
            .build(ctx, quantized, weights_->fc2);
        quantized_acoustic = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, quantized_acoustic);
        auto x = engine::modules::Conv1dModule({config.acoustic_hidden_size, config.decoder_hidden_size, 7, 1, 3, 1, true})
            .build(ctx, quantized_acoustic, weights_->conv1);
        for (const auto & block : weights_->blocks) {
            x = build_decoder_block(ctx, x, block);
        }
        x = engine::modules::Snake1dModule({weights_->blocks.back().out_channels}).build(ctx, x, normalize_snake_weights(ctx, weights_->final_snake, weights_->blocks.back().out_channels));
        x = engine::modules::Conv1dModule({weights_->blocks.back().out_channels, 1, 7, 1, 3, 1, true})
            .build(ctx, x, weights_->conv2);
        output_frames_ = x.shape.dims[2];
        x = engine::core::reshape_tensor(ctx, ensure_contiguous(ctx, x), engine::core::TensorShape::from_dims({output_frames_}));
        output_audio_ = x.tensor;
        ggml_set_output(output_audio_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
        ggml_build_forward_expand(graph_, output_audio_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs codec graph");
        }
        sequence_steps_ = steps;
    }

    std::shared_ptr<const HiggsAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const HiggsCodecWeights> weights_;
    int64_t sequence_steps_ = 0;
    int64_t output_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_codes_ = nullptr;
    ggml_tensor * output_audio_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

HiggsCodecDecoderRuntime::HiggsCodecDecoderRuntime(
    std::shared_ptr<const HiggsAssets> assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
        std::move(assets),
        execution_context,
        weight_context_bytes,
        graph_context_bytes,
        weight_storage_type)) {}

HiggsCodecDecoderRuntime::~HiggsCodecDecoderRuntime() = default;

runtime::AudioBuffer HiggsCodecDecoderRuntime::decode(const HiggsCodeSequence & raw_codes) {
    return impl_->decode(raw_codes);
}

}  // namespace engine::models::higgs_tts
