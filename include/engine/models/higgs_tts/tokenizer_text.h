#pragma once

#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::higgs_tts {

class HiggsTextTokenizer {
public:
    struct Impl;

    explicit HiggsTextTokenizer(std::shared_ptr<const HiggsAssets> assets);

    std::vector<int32_t> encode(const std::string & text, bool parse_special = true) const;
    HiggsTextPrompt build_prompt(const HiggsPromptContext & context) const;
    HiggsTextPrompt build_prompt(
        const std::string & text,
        int64_t num_ref_tokens = 0,
        const std::optional<std::string> & reference_text = std::nullopt) const;
    int32_t token_id(const std::string & token) const;
    int32_t audio_placeholder_id() const noexcept;
    int32_t tts_token_id() const noexcept;
    int32_t ref_audio_token_id() const noexcept;
    std::optional<int32_t> ref_text_token_id() const noexcept;
    int32_t text_token_id() const noexcept;
    int32_t audio_token_id() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::higgs_tts
