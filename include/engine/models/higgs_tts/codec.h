#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <memory>
#include <string_view>

namespace engine::models::higgs_tts {

constexpr std::string_view kHiggsCodecWeightPrefix = "tied.embedding.modality_embeddings.0.model.";

std::shared_ptr<const assets::TensorSource> make_higgs_codec_tensor_source(
    std::shared_ptr<const assets::TensorSource> source);

}  // namespace engine::models::higgs_tts
