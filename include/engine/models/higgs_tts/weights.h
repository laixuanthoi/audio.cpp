#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/qwen_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::higgs_tts {

struct HiggsBackboneWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue text_embedding;
    engine::core::TensorValue modality_embedding;
    engine::modules::LinearWeights modality_head;
    engine::modules::NormWeights norm;
    QwenDecoderStackWeights decoder;
};

int64_t higgs_head_dim(const HiggsTextConfig & config);

class HiggsBackboneWeightsRuntime final {
public:
    HiggsBackboneWeightsRuntime(
        std::shared_ptr<const HiggsAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~HiggsBackboneWeightsRuntime();

    const HiggsAssets & assets() const noexcept;
    const HiggsBackboneWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    int threads() const noexcept;
    bool weights_uploaded() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_tts
