#include "engine/models/higgs_tts/generator.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/higgs_tts/components.h"
#include "engine/models/higgs_tts/utils.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

constexpr size_t kDefaultGraphNodes = 65536;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::shared_ptr<const HiggsBackboneWeightsRuntime> require_weights(
    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights) {
    if (weights == nullptr) {
        throw std::runtime_error("Higgs runtime requires backbone weights");
    }
    return weights;
}

engine::core::TensorValue ensure_contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

}  // namespace

class HiggsTextEmbeddingRuntime::Impl {
public:
    Impl(std::shared_ptr<const HiggsBackboneWeightsRuntime> weights, size_t graph_context_bytes)
        : weights_(require_weights(std::move(weights))) {
        build(graph_context_bytes);
    }

    ~Impl() {
        engine::core::release_backend_graph_resources(weights_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    std::vector<float> embed_token(int32_t token_id) {
        const auto & config = weights_->assets().config.text;
        if (token_id < 0 || token_id >= config.vocab_size) {
            throw std::runtime_error("Higgs text embedding token id is out of range");
        }
        ggml_backend_tensor_set(token_id_, &token_id, 0, sizeof(token_id));
        engine::core::set_backend_threads(weights_->backend(), weights_->threads());
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), graph_);
        ggml_backend_synchronize(weights_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs text embedding graph compute failed");
        }
        std::vector<float> output(static_cast<size_t>(config.hidden_size), 0.0F);
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
    void build(size_t graph_context_bytes) {
        const auto & config = weights_->assets().config.text;
        if (graph_context_bytes == 0) {
            throw std::runtime_error("Higgs text embedding graph context bytes must be non-zero");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs text embedding graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.text_embedding"};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token = engine::core::wrap_tensor(token_id_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto embedding = engine::modules::EmbeddingModule({config.vocab_size, config.hidden_size})
            .build(ctx, token, weights_->weights().text_embedding);
        embedding = engine::core::reshape_tensor(ctx, ensure_contiguous(ctx, embedding), engine::core::TensorShape::from_dims({config.hidden_size}));
        output_ = embedding.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs text embedding graph");
        }
    }

    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class HiggsCodebookEmbeddingRuntime::Impl {
public:
    Impl(std::shared_ptr<const HiggsBackboneWeightsRuntime> weights, size_t graph_context_bytes)
        : weights_(require_weights(std::move(weights)))
        , graph_context_bytes_(graph_context_bytes) {
        if (graph_context_bytes_ == 0) {
            throw std::runtime_error("Higgs codebook embedding graph context bytes must be non-zero");
        }
    }

    ~Impl() {
        release_graph();
    }

    HiggsEmbeddedSequence embed_codes(const HiggsCodeSequence & delayed_codes) {
        if (delayed_codes.num_codebooks != weights_->assets().config.audio_encoder.num_codebooks) {
            throw std::runtime_error("Higgs codebook embedding codebook count mismatch");
        }
        const auto fused = fuse_codebook_ids(delayed_codes, weights_->assets().config.audio_encoder.vocab_size);
        if (fused.frames == 0) {
            return {{}, 0, weights_->assets().config.text.hidden_size};
        }
        if (sequence_steps_ != fused.frames) {
            build(fused.frames);
        }
        ggml_backend_tensor_set(input_fused_ids_, fused.values.data(), 0, fused.values.size() * sizeof(int32_t));
        engine::core::set_backend_threads(weights_->backend(), weights_->threads());
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), graph_);
        ggml_backend_synchronize(weights_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs codebook embedding graph compute failed");
        }
        HiggsEmbeddedSequence output;
        output.steps = fused.frames;
        output.hidden_size = weights_->assets().config.text.hidden_size;
        output.values.resize(static_cast<size_t>(output.steps * output.hidden_size), 0.0F);
        ggml_backend_tensor_get(output_embeddings_, output.values.data(), 0, output.values.size() * sizeof(float));
        return output;
    }

private:
    void release_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(weights_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        graph_ = nullptr;
        input_fused_ids_ = nullptr;
        output_embeddings_ = nullptr;
        ctx_.reset();
        sequence_steps_ = 0;
    }

    void build(int64_t steps) {
        release_graph();
        const auto & text = weights_->assets().config.text;
        const auto & audio = weights_->assets().config.audio_encoder;
        ggml_init_params params{graph_context_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs codebook embedding graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.codebook_embedding"};
        input_fused_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, audio.num_codebooks, steps);
        auto input = engine::core::wrap_tensor(
            input_fused_ids_,
            engine::core::TensorShape::from_dims({steps, audio.num_codebooks}),
            GGML_TYPE_I32);
        auto embedded = HiggsFusedCodebookEmbeddingModule({audio.num_codebooks, audio.vocab_size, text.hidden_size})
            .build(ctx, input, weights_->weights().modality_embedding);
        embedded = ensure_contiguous(ctx, embedded);
        output_embeddings_ = embedded.tensor;
        ggml_set_output(output_embeddings_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
        ggml_build_forward_expand(graph_, output_embeddings_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs codebook embedding graph");
        }
        sequence_steps_ = steps;
    }

    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights_;
    size_t graph_context_bytes_ = 0;
    int64_t sequence_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_fused_ids_ = nullptr;
    ggml_tensor * output_embeddings_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class HiggsPromptPrefillRuntime::Impl {
public:
    Impl(std::shared_ptr<const HiggsBackboneWeightsRuntime> weights, size_t graph_context_bytes)
        : weights_(require_weights(std::move(weights)))
        , graph_context_bytes_(graph_context_bytes) {
        if (graph_context_bytes_ == 0) {
            throw std::runtime_error("Higgs prompt prefill graph context bytes must be non-zero");
        }
    }

    ~Impl() {
        release_graph();
    }

    HiggsPromptPrefillOutput run(const HiggsPromptPrefillInput & input) {
        const auto & text = weights_->assets().config.text;
        if (input.steps <= 0) {
            throw std::runtime_error("Higgs prompt prefill requires positive steps");
        }
        if (static_cast<int64_t>(input.input_embeddings.size()) != input.steps * text.hidden_size) {
            throw std::runtime_error("Higgs prompt prefill input embedding size mismatch");
        }
        if (sequence_steps_ != input.steps) {
            build(input.steps);
        }
        ggml_backend_tensor_set(input_embeddings_, input.input_embeddings.data(), 0, input.input_embeddings.size() * sizeof(float));
        engine::core::set_backend_threads(weights_->backend(), weights_->threads());
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), graph_);
        ggml_backend_synchronize(weights_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs prompt prefill graph compute failed");
        }

        HiggsPromptPrefillOutput output;
        output.num_codebooks = weights_->assets().config.audio_encoder.num_codebooks;
        output.vocab_size = weights_->assets().config.audio_encoder.vocab_size;
        output.last_hidden.resize(static_cast<size_t>(text.hidden_size), 0.0F);
        ggml_backend_tensor_get(last_hidden_output_, output.last_hidden.data(), 0, output.last_hidden.size() * sizeof(float));
        output.logits.resize(static_cast<size_t>(output.num_codebooks * output.vocab_size), 0.0F);
        ggml_backend_tensor_get(logits_output_, output.logits.data(), 0, output.logits.size() * sizeof(float));
        return output;
    }

private:
    void release_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(weights_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        graph_ = nullptr;
        input_embeddings_ = nullptr;
        positions_ = nullptr;
        last_hidden_output_ = nullptr;
        logits_output_ = nullptr;
        ctx_.reset();
        sequence_steps_ = 0;
    }

    void build(int64_t steps) {
        release_graph();
        const auto & text = weights_->assets().config.text;
        const auto & audio = weights_->assets().config.audio_encoder;
        ggml_init_params params{graph_context_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs prompt prefill graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.prompt_prefill"};
        auto input = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, steps, text.hidden_size}));
        input_embeddings_ = input.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps);
        auto positions = engine::core::wrap_tensor(
            positions_,
            engine::core::TensorShape::from_dims({steps}),
            GGML_TYPE_I32);

        QwenDecoderStackModule decoder({
            text.hidden_size,
            text.num_attention_heads,
            text.num_key_value_heads,
            text.head_dim,
            text.intermediate_size,
            text.num_hidden_layers,
            text.rms_norm_eps,
            text.rope_theta,
            GGML_PREC_F32,
            GGML_PREC_DEFAULT,
        });
        auto decoded = decoder.build(ctx, input, positions, weights_->weights().decoder);
        auto normalized = engine::modules::RMSNormModule({text.hidden_size, text.rms_norm_eps, true, false})
            .build(ctx, decoded.output, weights_->weights().norm);
        auto last_hidden = engine::modules::SliceModule({1, steps - 1, 1}).build(ctx, normalized);
        auto logits = HiggsFusedCodebookHeadModule({audio.num_codebooks, audio.vocab_size, text.hidden_size})
            .build(ctx, last_hidden, weights_->weights().modality_head);
        last_hidden = engine::core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, last_hidden),
            engine::core::TensorShape::from_dims({text.hidden_size}));
        logits = engine::core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, logits),
            engine::core::TensorShape::from_dims({audio.num_codebooks, audio.vocab_size}));
        last_hidden_output_ = last_hidden.tensor;
        logits_output_ = logits.tensor;
        ggml_set_output(last_hidden_output_);
        ggml_set_output(logits_output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
        ggml_build_forward_expand(graph_, last_hidden_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs prompt prefill graph");
        }
        std::vector<int32_t> positions_values(static_cast<size_t>(steps));
        for (int64_t i = 0; i < steps; ++i) {
            positions_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions_values.data(), 0, positions_values.size() * sizeof(int32_t));
        sequence_steps_ = steps;
    }

    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights_;
    size_t graph_context_bytes_ = 0;
    int64_t sequence_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_embeddings_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * last_hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

HiggsTextEmbeddingRuntime::HiggsTextEmbeddingRuntime(
    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights,
    size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes)) {}

HiggsTextEmbeddingRuntime::~HiggsTextEmbeddingRuntime() = default;

std::vector<float> HiggsTextEmbeddingRuntime::embed_token(int32_t token_id) {
    return impl_->embed_token(token_id);
}

HiggsCodebookEmbeddingRuntime::HiggsCodebookEmbeddingRuntime(
    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights,
    size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes)) {}

HiggsCodebookEmbeddingRuntime::~HiggsCodebookEmbeddingRuntime() = default;

HiggsEmbeddedSequence HiggsCodebookEmbeddingRuntime::embed_codes(const HiggsCodeSequence & delayed_codes) {
    return impl_->embed_codes(delayed_codes);
}

HiggsPromptPrefillRuntime::HiggsPromptPrefillRuntime(
    std::shared_ptr<const HiggsBackboneWeightsRuntime> weights,
    size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes)) {}

HiggsPromptPrefillRuntime::~HiggsPromptPrefillRuntime() = default;

HiggsPromptPrefillOutput HiggsPromptPrefillRuntime::run(const HiggsPromptPrefillInput & input) {
    return impl_->run(input);
}

}  // namespace engine::models::higgs_tts
