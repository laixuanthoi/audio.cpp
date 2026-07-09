#include "engine/models/higgs_tts/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {

struct HiggsTextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t audio_placeholder_id = -100;
    int32_t tts_id = 0;
    int32_t ref_audio_id = 0;
    std::optional<int32_t> ref_text_id = std::nullopt;
    int32_t text_id = 0;
    int32_t audio_id = 0;
};

namespace {

int32_t require_token_id(
    const std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> & tokenizer,
    const std::string & token) {
    const auto token_id = tokenizer->find_token_id(token);
    if (!token_id.has_value()) {
        throw std::runtime_error("Higgs TTS tokenizer is missing token: " + token);
    }
    return *token_id;
}

std::shared_ptr<const HiggsTextTokenizer::Impl> load_impl(const HiggsAssets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_config_path = assets.paths.tokenizer_config_path;
    spec.tokenizer_json_path = assets.paths.tokenizer_json_path;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;

    auto impl = std::make_shared<HiggsTextTokenizer::Impl>();
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    impl->audio_placeholder_id = static_cast<int32_t>(assets.config.audio_token_id);
    impl->tts_id = require_token_id(impl->tokenizer, "<|tts|>");
    impl->ref_audio_id = require_token_id(impl->tokenizer, "<|ref_audio|>");
    if (const auto token_id = impl->tokenizer->find_token_id("<|ref_text|>"); token_id.has_value()) {
        impl->ref_text_id = *token_id;
    }
    impl->text_id = require_token_id(impl->tokenizer, "<|text|>");
    impl->audio_id = require_token_id(impl->tokenizer, "<|audio|>");
    return impl;
}

void append_encoded_text(
    std::vector<int32_t> & input_ids,
    const std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> & tokenizer,
    const std::string & text,
    bool parse_special) {
    if (text.empty()) {
        return;
    }
    const auto text_ids = tokenizer->encode(text, parse_special);
    input_ids.insert(input_ids.end(), text_ids.begin(), text_ids.end());
}

}  // namespace

HiggsTextTokenizer::HiggsTextTokenizer(std::shared_ptr<const HiggsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs TTS text tokenizer requires assets");
    }
    impl_ = load_impl(*assets);
}

std::vector<int32_t> HiggsTextTokenizer::encode(const std::string & text, bool parse_special) const {
    return impl_->tokenizer->encode(text, parse_special);
}

HiggsTextPrompt HiggsTextTokenizer::build_prompt(const HiggsPromptContext & context) const {
    HiggsTextPrompt prompt;
    prompt.input_ids.push_back(impl_->tts_id);

    std::vector<HiggsPromptSegment> target_segments;
    for (const auto & segment : context.segments) {
        switch (segment.kind) {
        case HiggsPromptSegmentKind::SystemText:
            if (segment.text.has_value() && !segment.text->empty()) {
                if (!prompt.system_text.has_value() || prompt.system_text->empty()) {
                    prompt.system_text = *segment.text;
                } else {
                    *prompt.system_text += "\n\n" + *segment.text;
                }
            }
            break;
        case HiggsPromptSegmentKind::Reference:
        case HiggsPromptSegmentKind::History:
            if (segment.text.has_value() && !segment.text->empty() && impl_->ref_text_id.has_value()) {
                prompt.input_ids.push_back(*impl_->ref_text_id);
                append_encoded_text(
                    prompt.input_ids,
                    impl_->tokenizer,
                    *segment.text,
                    segment.parse_special_text);
            }
            if (segment.audio_placeholders > 0) {
                prompt.input_ids.push_back(impl_->ref_audio_id);
                prompt.input_ids.insert(
                    prompt.input_ids.end(),
                    static_cast<size_t>(segment.audio_placeholders),
                    impl_->audio_placeholder_id);
                prompt.num_reference_placeholders += segment.audio_placeholders;
                prompt.audio_placeholder_lengths.push_back(segment.audio_placeholders);
            }
            break;
        case HiggsPromptSegmentKind::Target:
            target_segments.push_back(segment);
            if (segment.text.has_value() && !segment.text->empty()) {
                if (!prompt.text.empty()) {
                    prompt.text += "\n\n";
                }
                prompt.text += *segment.text;
            }
            break;
        }
    }

    if (prompt.text.empty()) {
        throw std::runtime_error("Higgs TTS requires non-empty target text in prompt context");
    }

    prompt.input_ids.push_back(impl_->text_id);
    bool wrote_text = false;
    if (prompt.system_text.has_value() && !prompt.system_text->empty()) {
        append_encoded_text(prompt.input_ids, impl_->tokenizer, *prompt.system_text, true);
        wrote_text = true;
    }
    for (const auto & segment : target_segments) {
        if (!segment.text.has_value() || segment.text->empty()) {
            continue;
        }
        if (wrote_text) {
            append_encoded_text(prompt.input_ids, impl_->tokenizer, "\n\n", false);
        }
        append_encoded_text(
            prompt.input_ids,
            impl_->tokenizer,
            *segment.text,
            segment.parse_special_text);
        wrote_text = true;
    }
    prompt.input_ids.push_back(impl_->audio_id);
    return prompt;
}

HiggsTextPrompt HiggsTextTokenizer::build_prompt(
    const std::string & text,
    int64_t num_ref_tokens,
    const std::optional<std::string> & reference_text) const {
    if (text.empty()) {
        throw std::runtime_error("Higgs TTS requires non-empty text input");
    }
    if (num_ref_tokens < 0) {
        throw std::runtime_error("Higgs TTS num_ref_tokens must be non-negative");
    }

    HiggsPromptContext context;
    if (num_ref_tokens > 0 || (reference_text.has_value() && !reference_text->empty())) {
        HiggsPromptSegment reference_segment;
        reference_segment.kind = HiggsPromptSegmentKind::Reference;
        reference_segment.text = reference_text;
        reference_segment.audio_placeholders = num_ref_tokens;
        reference_segment.parse_special_text = false;
        context.segments.push_back(std::move(reference_segment));
    }

    HiggsPromptSegment target_segment;
    target_segment.kind = HiggsPromptSegmentKind::Target;
    target_segment.text = text;
    target_segment.parse_special_text = true;
    context.segments.push_back(std::move(target_segment));
    return build_prompt(context);
}

int32_t HiggsTextTokenizer::token_id(const std::string & token) const {
    const auto token_id = impl_->tokenizer->find_token_id(token);
    if (!token_id.has_value()) {
        throw std::runtime_error("Higgs TTS tokenizer does not contain token: " + token);
    }
    return *token_id;
}

int32_t HiggsTextTokenizer::audio_placeholder_id() const noexcept {
    return impl_->audio_placeholder_id;
}

int32_t HiggsTextTokenizer::tts_token_id() const noexcept {
    return impl_->tts_id;
}

int32_t HiggsTextTokenizer::ref_audio_token_id() const noexcept {
    return impl_->ref_audio_id;
}

std::optional<int32_t> HiggsTextTokenizer::ref_text_token_id() const noexcept {
    return impl_->ref_text_id;
}

int32_t HiggsTextTokenizer::text_token_id() const noexcept {
    return impl_->text_id;
}

int32_t HiggsTextTokenizer::audio_token_id() const noexcept {
    return impl_->audio_id;
}

}  // namespace engine::models::higgs_tts
