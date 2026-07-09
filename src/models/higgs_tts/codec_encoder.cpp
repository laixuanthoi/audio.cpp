#include "engine/models/higgs_tts/codec_encoder.h"

#include "engine/framework/io/json.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/audio_tokenizer.h"

#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::higgs_tts {
namespace json = engine::io::json;
namespace {

class NativeAsF32TensorSource final : public engine::assets::TensorSource {
public:
    explicit NativeAsF32TensorSource(std::shared_ptr<const engine::assets::TensorSource> base)
        : base_(std::move(base)) {
        if (base_ == nullptr) {
            throw std::runtime_error("Higgs codec encoder tensor source wrapper requires a base tensor source");
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return base_->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override { return base_->has_tensor(name); }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        auto meta = base_->require_metadata(name);
        meta.dtype = "F32";
        return meta;
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        auto metas = base_->tensors();
        for (auto & meta : metas) {
            meta.dtype = "F32";
        }
        return metas;
    }
    void release_storage() const override { base_->release_storage(); }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override { return base_->require_tensor_data(name); }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const override {
        return base_->require_f32(name, expected_shape);
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const override {
        return base_->optional_f32(name, expected_shape);
    }
    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        engine::assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        (void) storage_type;
        base_->set_backend_f32_tensor(tensor, name, expected_shape);
    }
    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        base_->set_backend_f32_tensor(tensor, name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override { return base_->require_i64_scalar(name); }

private:
    std::shared_ptr<const engine::assets::TensorSource> base_;
};

std::shared_ptr<const HiggsAssets> require_assets(std::shared_ptr<const HiggsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs codec encoder runtime requires assets");
    }
    return assets;
}

std::filesystem::path resolve_codec_spec_path(const HiggsAssets & assets) {
    const auto candidate = assets.paths.model_root.parent_path() / ".higgs-v2-tokenizer-spec" / "config.json";
    if (std::filesystem::is_regular_file(candidate)) {
        return std::filesystem::weakly_canonical(candidate);
    }
    throw std::runtime_error(
        "Higgs codec encoder requires the Boson Higgs v2 tokenizer spec at: " + candidate.string());
}

int64_t product_or_throw(const std::vector<int64_t> & values, const char * label) {
    if (values.empty()) {
        throw std::runtime_error(std::string("Higgs codec encoder expected non-empty ") + label);
    }
    int64_t out = 1;
    for (const int64_t value : values) {
        if (value <= 0) {
            throw std::runtime_error(std::string("Higgs codec encoder expected positive ") + label);
        }
        out *= value;
    }
    return out;
}

omnivoice::OmniVoiceAudioTokenizerConfig load_compat_audio_tokenizer_config(const HiggsAssets & assets) {
    const auto root = json::parse_file(resolve_codec_spec_path(assets));
    omnivoice::OmniVoiceAudioTokenizerConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.sample_rate = json::require_i32(root, "sample_rate");
    config.semantic_sample_rate = json::optional_i32(root, "semantic_sample_rate", 16000);
    config.downsample_factor = json::require_i64(root, "downsample_factor");
    config.codebook_size = json::require_i64(root, "codebook_size");
    config.num_codebooks = assets.config.audio_encoder.num_codebooks;
    config.acoustic_codebooks = config.num_codebooks;
    config.codebook_dim = json::require_i64(root, "codebook_dim");
    config.kernel_size = json::require_i64(root, "kernel_size");
    config.unit_kernel_size = json::require_i64(root, "unit_kernel_size");
    config.target_bandwidths = json::optional_f32_array(root, "target_bandwidths");
    config.channel_ratios = json::require_i64_array(root, "channel_ratios");
    config.strides = json::require_i64_array(root, "strides");
    config.block_dilations = json::require_i64_array(root, "block_dilations");

    const auto & acoustic = root.require("acoustic_model_config");
    config.acoustic_model.codebook_dim = json::require_i64(acoustic, "codebook_dim");
    config.acoustic_model.encoder_hidden_size = json::require_i64(acoustic, "encoder_hidden_size");
    config.acoustic_model.decoder_hidden_size = json::require_i64(acoustic, "decoder_hidden_size");
    config.acoustic_model.hidden_size = json::require_i64(acoustic, "hidden_size");
    config.acoustic_model.downsampling_ratios = json::require_i64_array(acoustic, "downsampling_ratios");
    config.acoustic_model.upsampling_ratios = json::require_i64_array(acoustic, "upsampling_ratios");
    config.hop_length = json::optional_i64(
        acoustic,
        "hop_length",
        product_or_throw(config.acoustic_model.downsampling_ratios, "acoustic downsampling ratios"));

    const auto & semantic = root.require("semantic_model_config");
    config.semantic_model.hidden_size = json::require_i64(semantic, "hidden_size");
    config.semantic_model.intermediate_size = json::require_i64(semantic, "intermediate_size");
    config.semantic_model.num_attention_heads = json::require_i64(semantic, "num_attention_heads");
    config.semantic_model.num_hidden_layers = json::require_i64(semantic, "num_hidden_layers");
    config.semantic_model.num_conv_pos_embeddings = json::require_i64(semantic, "num_conv_pos_embeddings");
    config.semantic_model.num_conv_pos_embedding_groups = json::require_i64(semantic, "num_conv_pos_embedding_groups");
    config.semantic_model.layer_norm_eps = json::optional_f32(semantic, "layer_norm_eps", 1.0e-5F);
    config.semantic_model.feat_proj_layer_norm = json::optional_bool(semantic, "feat_proj_layer_norm", true);
    config.semantic_model.do_stable_layer_norm = json::optional_bool(semantic, "do_stable_layer_norm", false);
    config.semantic_model.conv_dim = json::require_i64_array(semantic, "conv_dim");
    config.semantic_model.conv_kernel = json::require_i64_array(semantic, "conv_kernel");
    config.semantic_model.conv_stride = json::require_i64_array(semantic, "conv_stride");

    config.hidden_size = config.acoustic_model.hidden_size + config.semantic_model.hidden_size;
    if (config.hidden_size <= 0) {
        throw std::runtime_error("Higgs codec encoder combined hidden_size must be positive");
    }
    if (config.hidden_size != config.acoustic_model.hidden_size + config.semantic_model.hidden_size) {
        throw std::runtime_error("Higgs codec encoder hidden_size does not match acoustic + semantic hidden sizes");
    }
    if (config.num_codebooks != assets.config.audio_encoder.num_codebooks) {
        throw std::runtime_error("Higgs codec encoder num_codebooks does not match the Higgs audio encoder config");
    }
    return config;
}

std::shared_ptr<const omnivoice::OmniVoiceAssets> make_compat_assets(const HiggsAssets & assets) {
    auto compat = std::make_shared<omnivoice::OmniVoiceAssets>();
    compat->paths.model_root = assets.paths.model_root;
    compat->config.audio_tokenizer = load_compat_audio_tokenizer_config(assets);
    compat->audio_tokenizer_weights = std::make_shared<NativeAsF32TensorSource>(assets.codec_weights);
    return compat;
}

}  // namespace

class HiggsCodecEncoderRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const HiggsAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t graph_context_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type)
        : assets_(require_assets(std::move(assets)))
        , compat_assets_(make_compat_assets(*assets_))
        , runtime_(compat_assets_, execution_context, graph_context_bytes, weight_context_bytes, weight_storage_type) {}

    HiggsCodeSequence encode_reference(const runtime::AudioBuffer & audio, bool has_reference_text) const {
        omnivoice::OmniVoiceReferenceAudioOptions options;
        options.preprocess_prompt = false;
        options.has_reference_text = has_reference_text;
        const auto tokens = runtime_.encode_reference_audio(audio, options);
        if (tokens.frames <= 0 || tokens.codebooks <= 0 || tokens.token_ids.empty()) {
            throw std::runtime_error("Higgs codec encoder returned an empty reference-code sequence");
        }
        if (tokens.codebooks != assets_->config.audio_encoder.num_codebooks) {
            throw std::runtime_error("Higgs codec encoder codebook count mismatch");
        }
        HiggsCodeSequence out;
        out.frames = tokens.frames;
        out.num_codebooks = tokens.codebooks;
        out.values = tokens.token_ids;
        return out;
    }

private:
    std::shared_ptr<const HiggsAssets> assets_;
    std::shared_ptr<const omnivoice::OmniVoiceAssets> compat_assets_;
    mutable omnivoice::OmniVoiceAudioTokenizerRuntime runtime_;
};

HiggsCodecEncoderRuntime::HiggsCodecEncoderRuntime(
    std::shared_ptr<const HiggsAssets> assets,
    engine::core::ExecutionContext & execution_context,
    size_t graph_context_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
        std::move(assets),
        execution_context,
        graph_context_bytes,
        weight_context_bytes,
        weight_storage_type)) {}

HiggsCodecEncoderRuntime::~HiggsCodecEncoderRuntime() = default;

HiggsCodeSequence HiggsCodecEncoderRuntime::encode_reference(
    const runtime::AudioBuffer & audio,
    bool has_reference_text) const {
    return impl_->encode_reference(audio, has_reference_text);
}

}  // namespace engine::models::higgs_tts
