#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::higgs_tts {

struct HiggsCodeSequence {
    std::vector<int32_t> values;
    int64_t frames = 0;
    int64_t num_codebooks = 0;
};

struct HiggsGenerationOptions {
    int64_t max_new_tokens = 2048;
    float temperature = 1.0F;
    std::optional<float> top_p = std::nullopt;
    std::optional<int> top_k = std::nullopt;
    std::optional<uint32_t> seed = std::nullopt;
};

struct HiggsVoiceCloneInput {
    runtime::AudioBuffer reference_audio;
    std::string reference_text;
    std::optional<HiggsCodeSequence> reference_codes = std::nullopt;
};

struct HiggsTTSRequest {
    std::string text;
    std::optional<HiggsVoiceCloneInput> voice_clone = std::nullopt;
    HiggsGenerationOptions generation;
};

enum class HiggsPromptSegmentKind {
    SystemText,
    Reference,
    History,
    Target,
};

struct HiggsPromptSegment {
    HiggsPromptSegmentKind kind = HiggsPromptSegmentKind::Reference;
    std::optional<std::string> text = std::nullopt;
    int64_t audio_placeholders = 0;
    bool parse_special_text = false;
};

struct HiggsPromptContext {
    std::vector<HiggsPromptSegment> segments;
};

struct HiggsTextPrompt {
    std::string text;
    std::vector<int32_t> input_ids;
    int64_t num_reference_placeholders = 0;
    std::vector<int64_t> audio_placeholder_lengths;
    std::optional<std::string> system_text = std::nullopt;
};

}  // namespace engine::models::higgs_tts
