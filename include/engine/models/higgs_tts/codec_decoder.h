#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::higgs_tts {

class HiggsCodecDecoderRuntime final {
public:
    HiggsCodecDecoderRuntime(
        std::shared_ptr<const HiggsAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~HiggsCodecDecoderRuntime();

    runtime::AudioBuffer decode(const HiggsCodeSequence & raw_codes);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_tts
