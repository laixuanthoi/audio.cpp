#pragma once

#include "engine/models/higgs_tts/types.h"
#include "engine/models/higgs_tts/utils.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::models::higgs_tts {

constexpr int32_t kHiggsStopCode = -1;

struct HiggsSamplerState {
    int64_t num_codebooks = 0;
    int64_t delay_count = 0;
    std::optional<int64_t> eoc_countdown = std::nullopt;
    bool generation_done = false;
    uint64_t sample_index = 0;
    std::vector<int32_t> last_codes;
};

std::vector<int32_t> higgs_sampler_step(
    const std::vector<float> & logits,
    int64_t vocab_size,
    HiggsSamplerState & state,
    const HiggsGenerationOptions & options,
    int32_t boc_id = kHiggsCodecBocId,
    int32_t eoc_id = kHiggsCodecEocId);

}  // namespace engine::models::higgs_tts
