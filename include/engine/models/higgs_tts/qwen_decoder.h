#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <vector>

struct ggml_cgraph;

namespace engine::models::higgs_tts {

struct QwenDecoderLayerConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
};

struct QwenMLPWeights {
    engine::modules::LinearWeights gate_proj;
    engine::modules::LinearWeights up_proj;
    engine::modules::LinearWeights down_proj;
};

struct QwenDecoderLayerWeights {
    engine::modules::NormWeights input_norm;
    engine::modules::AttentionWeights self_attention;
    engine::modules::NormWeights q_norm;
    engine::modules::NormWeights k_norm;
    engine::modules::NormWeights post_norm;
    QwenMLPWeights mlp;
};

struct QwenDecoderLayerOutputs {
    engine::core::TensorValue output;
    engine::core::TensorValue key;
    engine::core::TensorValue value;
};

class QwenDecoderLayerModule {
public:
    explicit QwenDecoderLayerModule(QwenDecoderLayerConfig config);

    const QwenDecoderLayerConfig & config() const noexcept;
    const engine::core::ModuleSchema & schema() const noexcept;

    QwenDecoderLayerOutputs build(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const std::optional<engine::core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<engine::core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) const;

    QwenDecoderLayerOutputs build_with_static_cache_tail(
        engine::core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const engine::core::TensorValue & cache_key,
        const engine::core::TensorValue & cache_value,
        const engine::core::TensorValue & attention_mask) const;

    static const engine::core::ModuleSchema & static_schema() noexcept;

private:
    QwenDecoderLayerConfig config_;
};

struct QwenDecoderStackConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
};

struct QwenDecoderStackWeights {
    std::vector<QwenDecoderLayerWeights> layers;
};

struct QwenDecoderStackLayerState {
    std::optional<engine::core::TensorValue> key;
    std::optional<engine::core::TensorValue> value;
};

struct QwenDecoderStackState {
    std::vector<QwenDecoderStackLayerState> layers;
};

struct QwenDecoderStackOutputs {
    engine::core::TensorValue output;
    QwenDecoderStackState state;
};

class QwenDecoderStackModule {
public:
    explicit QwenDecoderStackModule(QwenDecoderStackConfig config);

    const QwenDecoderStackConfig & config() const noexcept;

    QwenDecoderStackOutputs build(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & positions,
        const QwenDecoderStackWeights & weights,
        const std::optional<QwenDecoderStackState> & prefix_state = std::nullopt,
        const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) const;

private:
    QwenDecoderStackConfig config_;
};

}  // namespace engine::models::higgs_tts
