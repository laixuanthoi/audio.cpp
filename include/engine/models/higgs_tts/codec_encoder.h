#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::higgs_tts {

class HiggsCodecEncoderRuntime final {
public:
    HiggsCodecEncoderRuntime(
        std::shared_ptr<const HiggsAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t graph_context_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~HiggsCodecEncoderRuntime();

    HiggsCodeSequence encode_reference(
        const runtime::AudioBuffer & audio,
        bool has_reference_text = false) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_tts
