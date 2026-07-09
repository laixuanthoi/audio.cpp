#include "engine/models/higgs_tts/components.h"

#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/higgs_tts/utils.h"

#include <stdexcept>

namespace engine::models::higgs_tts {
namespace {

void validate_config(const HiggsFusedCodebookConfig & config) {
    if (config.num_codebooks <= 0 || config.vocab_size <= 0 || config.hidden_size <= 0) {
        throw std::runtime_error("Higgs fused codebook config dimensions must be positive");
    }
}

core::TensorShape expected_weight_shape(const HiggsFusedCodebookConfig & config) {
    return core::TensorShape::from_dims({fused_embedding_rows(config.num_codebooks, config.vocab_size), config.hidden_size});
}

}  // namespace

HiggsFusedCodebookEmbeddingModule::HiggsFusedCodebookEmbeddingModule(HiggsFusedCodebookConfig config)
    : config_(config) {
    validate_config(config_);
}

const HiggsFusedCodebookConfig & HiggsFusedCodebookEmbeddingModule::config() const noexcept {
    return config_;
}

core::TensorValue HiggsFusedCodebookEmbeddingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & fused_ids,
    const core::TensorValue & weight) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (fused_ids.type != GGML_TYPE_I32) {
        throw std::runtime_error("Higgs fused codebook embedding expects GGML_TYPE_I32 fused ids");
    }
    core::validate_shape(weight, expected_weight_shape(config_), "weight");
    core::validate_rank_between(fused_ids, 2, 2, "fused_ids");
    if (fused_ids.shape.dims[1] != config_.num_codebooks) {
        throw std::runtime_error("Higgs fused codebook embedding input last dimension must equal num_codebooks");
    }

    const modules::EmbeddingModule embedding(
        {fused_embedding_rows(config_.num_codebooks, config_.vocab_size), config_.hidden_size});
    modules::AddModule add;

    core::TensorValue summed;
    bool has_sum = false;
    for (int64_t codebook = 0; codebook < config_.num_codebooks; ++codebook) {
        auto ids = modules::SliceModule({1, codebook, 1}).build(ctx, fused_ids);
        ids = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, ids),
            core::TensorShape::from_dims({fused_ids.shape.dims[0]}));
        auto embedded = embedding.build(ctx, ids, weight);
        if (!has_sum) {
            summed = embedded;
            has_sum = true;
        } else {
            summed = add.build(ctx, summed, embedded);
        }
    }
    if (!has_sum) {
        throw std::runtime_error("Higgs fused codebook embedding received zero codebooks");
    }
    return summed;
}

HiggsFusedCodebookHeadModule::HiggsFusedCodebookHeadModule(HiggsFusedCodebookConfig config)
    : config_(config) {
    validate_config(config_);
}

const HiggsFusedCodebookConfig & HiggsFusedCodebookHeadModule::config() const noexcept {
    return config_;
}

core::TensorValue HiggsFusedCodebookHeadModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const modules::LinearWeights & weights) const {
    const modules::LinearModule projection(
        {config_.hidden_size, fused_embedding_rows(config_.num_codebooks, config_.vocab_size), false, GGML_PREC_DEFAULT});
    auto logits = projection.build(ctx, hidden, weights);
    if (hidden.shape.rank == 2) {
        return core::reshape_tensor(
            ctx,
            logits,
            core::TensorShape::from_dims({hidden.shape.dims[0], config_.num_codebooks, config_.vocab_size}));
    }
    if (hidden.shape.rank == 3) {
        return core::reshape_tensor(
            ctx,
            logits,
            core::TensorShape::from_dims({hidden.shape.dims[0], hidden.shape.dims[1], config_.num_codebooks, config_.vocab_size}));
    }
    throw std::runtime_error("Higgs fused codebook head expects hidden rank 2 or 3");
}

}  // namespace engine::models::higgs_tts
