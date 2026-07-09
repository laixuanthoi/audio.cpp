#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"

#include <cstdint>

namespace engine::models::higgs_tts {

struct HiggsFusedCodebookConfig {
    int64_t num_codebooks = 0;
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
};

class HiggsFusedCodebookEmbeddingModule {
public:
    explicit HiggsFusedCodebookEmbeddingModule(HiggsFusedCodebookConfig config);

    const HiggsFusedCodebookConfig & config() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & fused_ids,
        const core::TensorValue & weight) const;

private:
    HiggsFusedCodebookConfig config_;
};

class HiggsFusedCodebookHeadModule {
public:
    explicit HiggsFusedCodebookHeadModule(HiggsFusedCodebookConfig config);

    const HiggsFusedCodebookConfig & config() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & hidden,
        const modules::LinearWeights & weights) const;

private:
    HiggsFusedCodebookConfig config_;
};

}  // namespace engine::models::higgs_tts
