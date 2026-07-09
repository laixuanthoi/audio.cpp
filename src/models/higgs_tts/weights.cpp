#include "engine/models/higgs_tts/weights.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/higgs_tts/utils.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

namespace weight_binding = engine::modules::binding;

std::shared_ptr<const HiggsAssets> require_assets(std::shared_ptr<const HiggsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs backbone runtime requires assets");
    }
    return assets;
}

QwenDecoderLayerWeights load_layer_weights(
    engine::core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const HiggsTextConfig & config,
    engine::assets::TensorStorageType storage_type) {
    QwenDecoderLayerWeights layer_weights;
    const int64_t head_dim = higgs_head_dim(config);
    layer_weights.input_norm = weight_binding::norm_weight_from_source(
        store, source, prefix + ".input_layernorm", config.hidden_size);
    layer_weights.self_attention = {
        store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.num_attention_heads * head_dim, config.hidden_size}),
        std::nullopt,
        store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.num_key_value_heads * head_dim, config.hidden_size}),
        std::nullopt,
        store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.num_key_value_heads * head_dim, config.hidden_size}),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.num_attention_heads * head_dim}),
        std::nullopt,
    };
    layer_weights.q_norm = weight_binding::norm_weight_from_source(
        store, source, prefix + ".self_attn.q_norm", head_dim);
    layer_weights.k_norm = weight_binding::norm_weight_from_source(
        store, source, prefix + ".self_attn.k_norm", head_dim);
    layer_weights.post_norm = weight_binding::norm_weight_from_source(
        store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    layer_weights.mlp.gate_proj = weight_binding::linear_from_source(
        store, source, prefix + ".mlp.gate_proj", storage_type,
        config.intermediate_size, config.hidden_size, false);
    layer_weights.mlp.up_proj = weight_binding::linear_from_source(
        store, source, prefix + ".mlp.up_proj", storage_type,
        config.intermediate_size, config.hidden_size, false);
    layer_weights.mlp.down_proj = weight_binding::linear_from_source(
        store, source, prefix + ".mlp.down_proj", storage_type,
        config.hidden_size, config.intermediate_size, false);
    return layer_weights;
}

std::shared_ptr<const HiggsBackboneWeights> load_weights(
    const HiggsAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type) {
    auto weights = std::make_shared<HiggsBackboneWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(), execution_context.backend_type(),
        "higgs_tts.backbone.weights", weight_context_bytes);

    auto & store = *weights->store;
    const auto & source = *assets.model_weights;
    const auto & text = assets.config.text;
    const auto & audio = assets.config.audio_encoder;
    weights->text_embedding = store.load_tensor(
        source,
        std::string(kHiggsTextEmbeddingWeightName),
        weight_storage_type,
        {text.vocab_size, text.hidden_size});
    weights->modality_embedding = store.load_tensor(
        source,
        std::string(kHiggsModalityEmbeddingWeightName),
        weight_storage_type,
        {audio.num_codebooks * audio.vocab_size, text.hidden_size});
    weights->modality_head = {weights->modality_embedding, std::nullopt};
    weights->norm = weight_binding::norm_weight_from_source(
        store,
        source,
        "body.norm",
        text.hidden_size);
    weights->decoder.layers.reserve(static_cast<size_t>(text.num_hidden_layers));
    for (int64_t layer = 0; layer < text.num_hidden_layers; ++layer) {
        weights->decoder.layers.push_back(load_layer_weights(
            store,
            source,
            "body.layers." + std::to_string(layer),
            text,
            weight_storage_type));
    }
    weights->store->upload();
    return weights;
}

}  // namespace

class HiggsBackboneWeightsRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const HiggsAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type)
        : assets_(require_assets(std::move(assets)))
        , backend_(execution_context.backend())
        , threads_(std::max(1, execution_context.config().threads))
        , weights_(load_weights(*assets_, execution_context, weight_context_bytes, weight_storage_type)) {}

    std::shared_ptr<const HiggsAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const HiggsBackboneWeights> weights_;
};

int64_t higgs_head_dim(const HiggsTextConfig & config) {
    if (config.num_attention_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("Higgs text attention config is invalid");
    }
    return config.head_dim;
}

HiggsBackboneWeightsRuntime::HiggsBackboneWeightsRuntime(
    std::shared_ptr<const HiggsAssets> assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
        std::move(assets),
        execution_context,
        weight_context_bytes,
        weight_storage_type)) {}

HiggsBackboneWeightsRuntime::~HiggsBackboneWeightsRuntime() = default;

const HiggsAssets & HiggsBackboneWeightsRuntime::assets() const noexcept {
    return *impl_->assets_;
}

const HiggsBackboneWeights & HiggsBackboneWeightsRuntime::weights() const noexcept {
    return *impl_->weights_;
}

ggml_backend_t HiggsBackboneWeightsRuntime::backend() const noexcept {
    return impl_->backend_;
}

int HiggsBackboneWeightsRuntime::threads() const noexcept {
    return impl_->threads_;
}

bool HiggsBackboneWeightsRuntime::weights_uploaded() const noexcept {
    return impl_ != nullptr && impl_->weights_ != nullptr && impl_->weights_->store != nullptr;
}

}  // namespace engine::models::higgs_tts
