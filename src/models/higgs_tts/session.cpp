#include "engine/models/higgs_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/higgs_tts/sampler.h"

#include <algorithm>
#include <cctype>

#include <chrono>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 512;

std::string normalize_higgs_control_value(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        if (std::isspace(ch) != 0 || ch == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::optional<std::string> find_request_emotion(const runtime::TaskRequest & request) {
    if (request.voice.has_value() && request.voice->style.has_value() && request.voice->style->emotion.has_value()) {
        const auto value = normalize_higgs_control_value(*request.voice->style->emotion);
        if (!value.empty()) {
            return value;
        }
    }
    if (const auto value = runtime::find_option(request.options, {"emotion"}); value.has_value()) {
        const auto normalized = normalize_higgs_control_value(*value);
        if (!normalized.empty()) {
            return normalized;
        }
    }
    return std::nullopt;
}

std::optional<std::string> find_scene_prompt(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"scene_prompt", "higgs_tts.scene_prompt"}); value.has_value()) {
        if (!value->empty()) {
            return *value;
        }
    }
    return std::nullopt;
}

std::string trim_ascii_whitespace(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::optional<std::string> find_request_option_value(
    const runtime::TaskRequest & request,
    const std::string & key) {
    if (const auto it = request.options.find(key); it != request.options.end() && !it->second.empty()) {
        return it->second;
    }
    const auto prefixed_key = std::string("higgs_tts.") + key;
    if (const auto it = request.options.find(prefixed_key); it != request.options.end() && !it->second.empty()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<std::string> resolve_builtin_voice_profile_description(std::string value) {
    value = trim_ascii_whitespace(std::move(value));
    constexpr char kProfilePrefix[] = "profile:";
    if (value.rfind(kProfilePrefix, 0) == 0) {
        value = trim_ascii_whitespace(value.substr(sizeof(kProfilePrefix) - 1));
    }
    static const std::unordered_map<std::string, std::string> kBuiltinProfiles = {
        {"male_en", "Male, American accent, modern speaking rate, moderate-pitch, friendly tone, and very clear audio."},
        {"female_en_story", "She speaks with a calm, gentle, and informative tone at a measured pace, with excellent articulation and very clear audio. She naturally brings storytelling to life with an articulate, genuine, and personable vocal style."},
        {"male_en_british", "He speaks with a clear British accent and a conversational, inquisitive tone. His delivery is articulate and at a moderate pace, and very clear audio."},
        {"female_en_british", "A female voice with a clear British accent speaking at a modern rate with a moderate-pitch in an expressive and friendly tone and very clear audio."},
    };
    if (const auto it = kBuiltinProfiles.find(value); it != kBuiltinProfiles.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<std::string> find_voice_profile_description(const runtime::TaskRequest & request) {
    for (const auto & key : {std::string("voice_profile"), std::string("speaker_profile"), std::string("profile")}) {
        if (const auto value = find_request_option_value(request, key); value.has_value()) {
            const auto trimmed = trim_ascii_whitespace(*value);
            if (trimmed.empty()) {
                continue;
            }
            if (const auto builtin = resolve_builtin_voice_profile_description(trimmed); builtin.has_value()) {
                return builtin;
            }
            return trimmed;
        }
    }
    return std::nullopt;
}

std::optional<std::string> build_system_scene_text(
    const std::optional<std::string> & scene_prompt,
    const std::optional<std::string> & voice_profile_description) {
    std::vector<std::string> sections;
    if (scene_prompt.has_value()) {
        const auto trimmed = trim_ascii_whitespace(*scene_prompt);
        if (!trimmed.empty()) {
            sections.push_back(trimmed);
        }
    }
    if (voice_profile_description.has_value()) {
        const auto trimmed = trim_ascii_whitespace(*voice_profile_description);
        if (!trimmed.empty()) {
            sections.push_back(std::string("SPEAKER0: ") + trimmed);
        }
    }
    if (sections.empty()) {
        return std::nullopt;
    }
    std::string body;
    for (const auto & section : sections) {
        if (!body.empty()) {
            body += "\n\n";
        }
        body += section;
    }
    return std::string("Generate audio following instruction.\n\n<|scene_desc_start|>\n")
        + body
        + "\n<|scene_desc_end|>";
}

void append_normalized_control_values(
    std::vector<std::string> & values,
    const std::string & raw_value) {
    size_t start = 0;
    while (start <= raw_value.size()) {
        const size_t comma = raw_value.find(',', start);
        const std::string piece = trim_ascii_whitespace(
            raw_value.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!piece.empty()) {
            values.push_back(normalize_higgs_control_value(piece));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
}

std::vector<std::string> find_request_style_controls(
    const runtime::TaskRequest & request,
    const std::string & category) {
    std::vector<std::string> values;
    if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto tag_it = request.voice->style->tags.find(category);
        if (tag_it != request.voice->style->tags.end()) {
            append_normalized_control_values(values, tag_it->second);
        }
    }
    if (const auto value = find_request_option_value(request, category); value.has_value()) {
        append_normalized_control_values(values, *value);
    }
    return values;
}

std::string prepend_control_tokens(const std::string & text, const std::vector<std::string> & tokens) {
    if (tokens.empty()) {
        return text;
    }
    std::string prefix;
    for (const auto & token : tokens) {
        if (token.empty()) {
            continue;
        }
        if (!prefix.empty()) {
            prefix += ' ';
        }
        prefix += token;
    }
    if (prefix.empty()) {
        return text;
    }
    if (text.empty()) {
        return prefix;
    }
    return prefix + " " + text;
}

std::shared_ptr<const HiggsAssets> require_assets(std::shared_ptr<const HiggsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs TTS session requires assets");
    }
    return assets;
}

const runtime::AudioBuffer * find_reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value()
        && request.voice->speaker.has_value()
        && request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) {
        return &*request.audio_input;
    }
    return nullptr;
}

void validate_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native
        || storage_type == engine::assets::TensorStorageType::F32
        || storage_type == engine::assets::TensorStorageType::F16
        || storage_type == engine::assets::TensorStorageType::BF16
        || storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

void parse_weight_type(
    const std::unordered_map<std::string, std::string> & options,
    const char * key,
    engine::assets::TensorStorageType & storage_type) {
    const auto it = options.find(key);
    if (it == options.end()) {
        return;
    }
    storage_type = engine::assets::parse_tensor_storage_type(it->second);
    validate_weight_storage(storage_type, key);
}

int64_t resolve_generation_chunk_buffer_size(const runtime::TaskRequest & request, int64_t chunk_count) {
    if (const auto value = runtime::parse_i64_option(
            request.options,
            {"generation_chunk_buffer_size", "higgs_tts.generation_chunk_buffer_size"})) {
        return std::max<int64_t>(0, *value);
    }
    return chunk_count > 1 ? 1 : 0;
}

int64_t resolve_longform_context_max_frames(const runtime::TaskRequest & request) {
    if (const auto value = runtime::parse_i64_option(
            request.options,
            {"longform_context_max_frames", "higgs_tts.longform_context_max_frames"})) {
        return std::max<int64_t>(0, *value);
    }
    return 64;
}

HiggsCodeSequence take_last_frames(const HiggsCodeSequence & codes, int64_t max_frames) {
    if (max_frames <= 0 || codes.frames <= max_frames) {
        return codes;
    }
    HiggsCodeSequence out;
    out.frames = max_frames;
    out.num_codebooks = codes.num_codebooks;
    out.values.resize(static_cast<size_t>(out.frames * out.num_codebooks));
    const int64_t start_frame = codes.frames - max_frames;
    for (int64_t frame = 0; frame < max_frames; ++frame) {
        const int64_t src_frame = start_frame + frame;
        const size_t src_offset = static_cast<size_t>(src_frame * codes.num_codebooks);
        const size_t dst_offset = static_cast<size_t>(frame * out.num_codebooks);
        std::copy_n(
            codes.values.begin() + static_cast<std::ptrdiff_t>(src_offset),
            static_cast<size_t>(out.num_codebooks),
            out.values.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    }
    return out;
}

void append_code_sequence(HiggsCodeSequence & dst, const HiggsCodeSequence & src) {
    if (src.frames <= 0 || src.values.empty()) {
        return;
    }
    if (dst.num_codebooks == 0) {
        dst.num_codebooks = src.num_codebooks;
    }
    if (dst.num_codebooks != src.num_codebooks) {
        throw std::runtime_error("Higgs long-form continuity codebook count mismatch");
    }
    dst.values.insert(dst.values.end(), src.values.begin(), src.values.end());
    dst.frames += src.frames;
}

std::vector<HiggsCodeSequence> build_continuity_reference_segments(
    const std::optional<HiggsCodeSequence> & base_reference,
    const std::vector<HiggsCodeSequence> & history_codes) {
    std::vector<HiggsCodeSequence> segments;
    if (base_reference.has_value() && base_reference->frames > 0 && !base_reference->values.empty()) {
        segments.push_back(*base_reference);
    }
    for (const auto & history : history_codes) {
        if (history.frames > 0 && !history.values.empty()) {
            segments.push_back(history);
        }
    }
    return segments;
}

std::vector<HiggsCodeSequence> apply_delay_pattern_to_segments(
    const std::vector<HiggsCodeSequence> & raw_segments) {
    std::vector<HiggsCodeSequence> delayed_segments;
    delayed_segments.reserve(raw_segments.size());
    for (const auto & segment : raw_segments) {
        if (segment.frames > 0 && !segment.values.empty()) {
            delayed_segments.push_back(apply_delay_pattern(segment));
        }
    }
    return delayed_segments;
}

int64_t total_code_frames(const std::vector<HiggsCodeSequence> & segments) {
    int64_t total = 0;
    for (const auto & segment : segments) {
        total += std::max<int64_t>(0, segment.frames);
    }
    return total;
}

HiggsPromptContext build_prompt_context(
    const HiggsTTSRequest & request,
    const std::optional<std::string> & system_scene_text,
    const std::optional<std::string> & base_reference_text,
    const std::optional<HiggsCodeSequence> & base_delayed_reference,
    const std::vector<std::string> & history_texts,
    const std::vector<HiggsCodeSequence> & history_delayed_segments) {
    HiggsPromptContext context;
    if (system_scene_text.has_value() && !system_scene_text->empty()) {
        HiggsPromptSegment system_segment;
        system_segment.kind = HiggsPromptSegmentKind::SystemText;
        system_segment.text = *system_scene_text;
        system_segment.parse_special_text = true;
        context.segments.push_back(std::move(system_segment));
    }

    if ((base_reference_text.has_value() && !base_reference_text->empty())
        || (base_delayed_reference.has_value() && base_delayed_reference->frames > 0)) {
        HiggsPromptSegment reference_segment;
        reference_segment.kind = HiggsPromptSegmentKind::Reference;
        if (base_reference_text.has_value() && !base_reference_text->empty()) {
            reference_segment.text = *base_reference_text;
        }
        if (base_delayed_reference.has_value()) {
            reference_segment.audio_placeholders = base_delayed_reference->frames;
        }
        context.segments.push_back(std::move(reference_segment));
    }

    const size_t history_count = std::max(history_texts.size(), history_delayed_segments.size());
    for (size_t index = 0; index < history_count; ++index) {
        HiggsPromptSegment history_segment;
        history_segment.kind = HiggsPromptSegmentKind::History;
        if (index < history_texts.size() && !history_texts[index].empty()) {
            history_segment.text = history_texts[index];
        }
        if (index < history_delayed_segments.size()) {
            history_segment.audio_placeholders = history_delayed_segments[index].frames;
        }
        if (history_segment.text.has_value() || history_segment.audio_placeholders > 0) {
            context.segments.push_back(std::move(history_segment));
        }
    }

    HiggsPromptSegment target_segment;
    target_segment.kind = HiggsPromptSegmentKind::Target;
    target_segment.text = request.text;
    target_segment.parse_special_text = true;
    context.segments.push_back(std::move(target_segment));
    return context;
}

}  // namespace

HiggsTTSSession::HiggsTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const HiggsAssets> assets)
    : RuntimeSessionBase(options)
    , task_(task)
    , assets_(require_assets(std::move(assets)))
    , text_tokenizer_(assets_) {
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS native skeleton currently supports only offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS native skeleton currently supports only the Tts task");
    }
    weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"higgs_tts.weight_context_mb"},
        weight_context_bytes_);
    text_embedding_graph_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"higgs_tts.text_embedding_graph_context_mb"},
        text_embedding_graph_context_bytes_);
    codebook_embedding_graph_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"higgs_tts.codebook_embedding_graph_context_mb"},
        codebook_embedding_graph_context_bytes_);
    prefill_graph_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"higgs_tts.prefill_graph_context_mb"},
        prefill_graph_context_bytes_);
    codec_weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"higgs_tts.codec_weight_context_mb"},
        codec_weight_context_bytes_);
    codec_graph_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"higgs_tts.codec_graph_context_mb"},
        codec_graph_context_bytes_);
    parse_weight_type(options.options, "higgs_tts.weight_type", weight_storage_type_);
}

std::string HiggsTTSSession::family() const {
    return "higgs_tts";
}

runtime::VoiceTaskKind HiggsTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HiggsTTSSession::run_mode() const {
    return task_.mode;
}

void HiggsTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

HiggsGenerationOptions HiggsTTSSession::generation_options_from_request(const runtime::TaskRequest & request) const {
    HiggsGenerationOptions options;
    if (const auto max_new_tokens = runtime::parse_i64_option(request.options, {"max_new_tokens", "max_tokens"})) {
        options.max_new_tokens = *max_new_tokens;
    }
    if (const auto temperature = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        options.temperature = *temperature;
    }
    options.top_p = runtime::parse_finite_float_option(request.options, {"top_p", "top-p"});
    if (const auto top_k = runtime::parse_int_option(request.options, {"top_k", "top-k"})) {
        options.top_k = *top_k;
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"});
    return options;
}

HiggsTTSRequest HiggsTTSSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Higgs TTS requires text input");
    }

    HiggsTTSRequest out;
    out.text = request.text_input->text;
    std::vector<std::string> control_tokens;
    if (const auto emotion = find_request_emotion(request); emotion.has_value()) {
        const auto token = std::string("<|emotion:") + *emotion + "|>";
        (void) text_tokenizer_.token_id(token);
        control_tokens.push_back(token);
    }
    for (const auto & value : find_request_style_controls(request, "style")) {
        const auto token = std::string("<|style:") + value + "|>";
        (void) text_tokenizer_.token_id(token);
        control_tokens.push_back(token);
    }
    for (const auto & value : find_request_style_controls(request, "prosody")) {
        const auto token = std::string("<|prosody:") + value + "|>";
        (void) text_tokenizer_.token_id(token);
        control_tokens.push_back(token);
    }
    for (const auto & value : find_request_style_controls(request, "sfx")) {
        const auto token = std::string("<|sfx:") + value + "|>";
        (void) text_tokenizer_.token_id(token);
        control_tokens.push_back(token);
    }
    out.text = prepend_control_tokens(out.text, control_tokens);
    out.generation = generation_options_from_request(request);
    if (const auto * reference_audio = find_reference_audio(request); reference_audio != nullptr) {
        HiggsVoiceCloneInput voice_clone;
        voice_clone.reference_audio = *reference_audio;
        if (const auto reference_text = runtime::find_option(request.options, {"reference_text"})) {
            voice_clone.reference_text = *reference_text;
        }
        out.voice_clone = std::move(voice_clone);
    }
    return out;
}

void HiggsTTSSession::validate_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Higgs TTS requires non-empty text input");
    }
    if (request.voice.has_value()
        && request.voice->speaker.has_value()
        && request.voice->speaker->cached_voice_id.has_value()
        && !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("Higgs TTS native skeleton does not support cached voice ids yet");
    }
}

bool HiggsTTSSession::debug_prefill_probe_requested(const runtime::TaskRequest & request) const {
    if (const auto match = runtime::find_option_match(request.options, {"higgs_tts.debug_prefill_probe", "debug_prefill_probe"}); match.has_value()) {
        return runtime::parse_bool_option(match->value, match->key);
    }
    return false;
}

void HiggsTTSSession::ensure_probe_runtimes() {
    if (backbone_weights_ != nullptr) {
        return;
    }
    backbone_weights_ = std::make_shared<HiggsBackboneWeightsRuntime>(
        assets_,
        execution_context(),
        weight_context_bytes_,
        weight_storage_type_);
    text_embedding_runtime_ = std::make_unique<HiggsTextEmbeddingRuntime>(
        backbone_weights_,
        text_embedding_graph_context_bytes_);
    codebook_embedding_runtime_ = std::make_unique<HiggsCodebookEmbeddingRuntime>(
        backbone_weights_,
        codebook_embedding_graph_context_bytes_);
    prompt_prefill_runtime_ = std::make_unique<HiggsPromptPrefillRuntime>(
        backbone_weights_,
        prefill_graph_context_bytes_);
    codec_encoder_runtime_ = std::make_unique<HiggsCodecEncoderRuntime>(
        assets_,
        execution_context(),
        codec_graph_context_bytes_,
        codec_weight_context_bytes_,
        engine::assets::TensorStorageType::F32);
    codec_decoder_runtime_ = std::make_unique<HiggsCodecDecoderRuntime>(
        assets_,
        execution_context(),
        codec_weight_context_bytes_,
        codec_graph_context_bytes_,
        engine::assets::TensorStorageType::F32);
}

std::vector<float> HiggsTTSSession::build_prompt_embeddings(
    const HiggsTextPrompt & prompt,
    const std::vector<HiggsCodeSequence> & delayed_reference_segments) {
    if (text_embedding_runtime_ == nullptr) {
        throw std::runtime_error("Higgs text embedding runtime is not initialized");
    }
    const int64_t hidden_size = assets_->config.text.hidden_size;
    const int32_t audio_placeholder_id = text_tokenizer_.audio_placeholder_id();

    std::vector<float> reference_embeddings;
    int64_t reference_steps = 0;
    if (!prompt.audio_placeholder_lengths.empty() || !delayed_reference_segments.empty()) {
        if (codebook_embedding_runtime_ == nullptr) {
            throw std::runtime_error("Higgs codebook embedding runtime is not initialized");
        }
        if (prompt.audio_placeholder_lengths.size() != delayed_reference_segments.size()) {
            throw std::runtime_error("Higgs prompt audio-segment count does not match delayed reference segment count");
        }
        for (size_t index = 0; index < delayed_reference_segments.size(); ++index) {
            const auto & segment = delayed_reference_segments[index];
            const int64_t expected_frames = prompt.audio_placeholder_lengths[index];
            if (segment.frames != expected_frames) {
                throw std::runtime_error("Higgs prompt placeholder count does not match delayed reference-code segment length");
            }
            const auto embedded = codebook_embedding_runtime_->embed_codes(segment);
            if (embedded.hidden_size != hidden_size || embedded.steps != segment.frames) {
                throw std::runtime_error("Higgs reference codebook embedding runtime returned an unexpected shape");
            }
            reference_embeddings.insert(
                reference_embeddings.end(),
                embedded.values.begin(),
                embedded.values.end());
            reference_steps += embedded.steps;
        }
    }

    std::vector<float> embeddings;
    embeddings.reserve(static_cast<size_t>(prompt.input_ids.size()) * static_cast<size_t>(hidden_size));
    int64_t consumed_reference_steps = 0;
    for (const int32_t token_id : prompt.input_ids) {
        if (token_id == audio_placeholder_id) {
            if (consumed_reference_steps >= reference_steps) {
                throw std::runtime_error("Higgs prompt consumed more audio placeholders than reference-code embeddings");
            }
            const size_t start = static_cast<size_t>(consumed_reference_steps * hidden_size);
            embeddings.insert(
                embeddings.end(),
                reference_embeddings.begin() + static_cast<std::ptrdiff_t>(start),
                reference_embeddings.begin() + static_cast<std::ptrdiff_t>(start + static_cast<size_t>(hidden_size)));
            consumed_reference_steps += 1;
            continue;
        }
        const auto token_embedding = text_embedding_runtime_->embed_token(token_id);
        embeddings.insert(embeddings.end(), token_embedding.begin(), token_embedding.end());
    }
    if (consumed_reference_steps != reference_steps) {
        throw std::runtime_error("Higgs prompt did not consume the full delayed reference-code sequence");
    }
    return embeddings;
}

HiggsCodeSequence HiggsTTSSession::generate_delayed_codes(
    const std::vector<float> & prompt_embeddings,
    int64_t prompt_steps,
    const HiggsGenerationOptions & options) {
    if (prompt_prefill_runtime_ == nullptr || codebook_embedding_runtime_ == nullptr) {
        throw std::runtime_error("Higgs probe runtimes are not initialized");
    }
    const int64_t hidden_size = assets_->config.text.hidden_size;
    const int64_t num_codebooks = assets_->config.audio_encoder.num_codebooks;
    if (prompt_steps <= 0) {
        throw std::runtime_error("Higgs generation prompt must contain at least one step");
    }
    if (static_cast<int64_t>(prompt_embeddings.size()) != prompt_steps * hidden_size) {
        throw std::runtime_error("Higgs generation prompt embedding size mismatch");
    }
    std::vector<float> input_embeddings = prompt_embeddings;
    auto prefill = prompt_prefill_runtime_->run({input_embeddings, prompt_steps});
    HiggsSamplerState sampler_state;
    sampler_state.num_codebooks = prefill.num_codebooks;
    HiggsCodeSequence delayed_codes;
    delayed_codes.num_codebooks = num_codebooks;

    const int64_t max_steps = std::max<int64_t>(1, options.max_new_tokens);
    while (delayed_codes.frames < max_steps) {
        const auto next_row = higgs_sampler_step(
            prefill.logits,
            prefill.vocab_size,
            sampler_state,
            options);
        if (next_row.empty() || next_row.front() == kHiggsStopCode) {
            break;
        }
        delayed_codes.values.insert(delayed_codes.values.end(), next_row.begin(), next_row.end());
        delayed_codes.frames += 1;
        if (sampler_state.generation_done || delayed_codes.frames >= max_steps) {
            break;
        }
        const HiggsCodeSequence one_step{next_row, 1, num_codebooks};
        const auto step_embedding = codebook_embedding_runtime_->embed_codes(one_step);
        if (step_embedding.hidden_size != hidden_size || step_embedding.steps != 1) {
            throw std::runtime_error("Higgs codebook embedding runtime returned an unexpected shape");
        }
        input_embeddings.insert(input_embeddings.end(), step_embedding.values.begin(), step_embedding.values.end());
        const int64_t total_steps = static_cast<int64_t>(input_embeddings.size()) / hidden_size;
        prefill = prompt_prefill_runtime_->run({input_embeddings, total_steps});
    }
    return delayed_codes;
}

runtime::TaskResult HiggsTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Higgs TTS run");
    validate_request(request);

    const auto wall_start = Clock::now();
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, engine::text::TextChunkMode::TagAware);
    if (chunk_requests.empty()) {
        throw std::runtime_error("Higgs TTS chunking produced no request chunks");
    }
    const int64_t continuity_chunk_buffer_size =
        resolve_generation_chunk_buffer_size(request, static_cast<int64_t>(chunk_requests.size()));
    const int64_t continuity_max_frames = resolve_longform_context_max_frames(request);

    const auto scene_prompt = find_scene_prompt(request);
    const auto voice_profile_description = find_voice_profile_description(request);
    const auto system_scene_text = build_system_scene_text(scene_prompt, voice_profile_description);

    ensure_probe_runtimes();
    const auto first_higgs_request = make_request(chunk_requests.front());

    std::optional<HiggsCodeSequence> base_reference_raw_codes = std::nullopt;
    std::optional<std::string> base_reference_text = std::nullopt;
    if (first_higgs_request.voice_clone.has_value()) {
        const auto & voice_clone = *first_higgs_request.voice_clone;
        const auto raw_reference_codes = voice_clone.reference_codes.has_value()
            ? *voice_clone.reference_codes
            : codec_encoder_runtime_->encode_reference(
                voice_clone.reference_audio,
                !voice_clone.reference_text.empty());
        codec_encoder_runtime_.reset();
        base_reference_raw_codes = raw_reference_codes;
        if (!voice_clone.reference_text.empty()) {
            base_reference_text = voice_clone.reference_text;
        }
        engine::debug::trace_log_scalar("higgs_tts.reference_raw_frames", raw_reference_codes.frames);
    }

    const auto base_delayed_reference = base_reference_raw_codes.has_value()
        ? std::optional<HiggsCodeSequence>(apply_delay_pattern(*base_reference_raw_codes))
        : std::nullopt;
    if (base_delayed_reference.has_value()) {
        engine::debug::trace_log_scalar(
            "higgs_tts.reference_delayed_frames",
            base_delayed_reference->frames);
    }

    int64_t total_prompt_tokens = 0;
    for (const auto & chunk_request : chunk_requests) {
        const auto higgs_request = make_request(chunk_request);
        const auto prompt_context = build_prompt_context(
            higgs_request,
            system_scene_text,
            base_reference_text,
            base_delayed_reference,
            {},
            {});
        const auto prompt = text_tokenizer_.build_prompt(prompt_context);
        total_prompt_tokens += static_cast<int64_t>(prompt.input_ids.size());
    }

    engine::debug::trace_log_scalar("higgs_tts.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("higgs_tts.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));
    engine::debug::trace_log_scalar("higgs_tts.prompt_token_count", total_prompt_tokens);
    engine::debug::trace_log_scalar("higgs_tts.generation_chunk_buffer_size", continuity_chunk_buffer_size);
    engine::debug::trace_log_scalar("higgs_tts.longform_context_max_frames", continuity_max_frames);

    runtime::TaskResult result;
    runtime::AudioBuffer combined_audio;
    bool has_audio = false;
    std::vector<HiggsCodeSequence> continuity_history_codes;
    std::vector<std::string> continuity_history_texts;
    for (size_t chunk_index = 0; chunk_index < chunk_requests.size(); ++chunk_index) {
        const auto higgs_request = make_request(chunk_requests[chunk_index]);
        const auto history_delayed_segments = apply_delay_pattern_to_segments(continuity_history_codes);
        const auto current_delayed_reference_segments = build_continuity_reference_segments(
            base_delayed_reference,
            history_delayed_segments);
        const auto current_reference_raw_segments = build_continuity_reference_segments(
            base_reference_raw_codes,
            continuity_history_codes);
        const auto prompt_context = build_prompt_context(
            higgs_request,
            system_scene_text,
            base_reference_text,
            base_delayed_reference,
            continuity_history_texts,
            history_delayed_segments);
        const auto prompt = text_tokenizer_.build_prompt(prompt_context);
        const auto prompt_embeddings = build_prompt_embeddings(prompt, current_delayed_reference_segments);
        const auto delayed_codes = generate_delayed_codes(
            prompt_embeddings,
            static_cast<int64_t>(prompt.input_ids.size()),
            higgs_request.generation);
        const auto raw_codes = reverse_delay_pattern(delayed_codes);

        engine::debug::trace_log_scalar(
            "higgs_tts.chunk_prompt_steps",
            static_cast<int64_t>(prompt.input_ids.size()));
        engine::debug::trace_log_scalar(
            "higgs_tts.chunk_generated_raw_steps",
            raw_codes.frames);
        engine::debug::trace_log_scalar(
            "higgs_tts.chunk_reference_raw_frames",
            total_code_frames(current_reference_raw_segments));
        engine::debug::trace_log_scalar(
            "higgs_tts.chunk_reference_segment_count",
            static_cast<int64_t>(current_delayed_reference_segments.size()));

        if (debug_prefill_probe_requested(request)) {
            engine::debug::trace_log_scalar("higgs_tts.prefill_probe.prompt_steps", static_cast<int64_t>(prompt.input_ids.size()));
            engine::debug::trace_log_scalar("higgs_tts.prefill_probe.generated_delayed_steps", delayed_codes.frames);
            engine::debug::trace_log_scalar("higgs_tts.prefill_probe.generated_raw_steps", raw_codes.frames);
            if (!delayed_codes.values.empty()) {
                const auto preview = code_row(delayed_codes, 0);
                engine::debug::trace_log_i32(
                    "higgs_tts.prefill_probe.first_delayed_row",
                    {delayed_codes.num_codebooks},
                    preview);
            }
        }

        if (raw_codes.frames <= 0 || raw_codes.values.empty()) {
            throw std::runtime_error(
                "Higgs TTS generated no audio codes for chunk " + std::to_string(chunk_index));
        }
        auto audio = codec_decoder_runtime_->decode(raw_codes);
        if (!has_audio) {
            combined_audio = std::move(audio);
            has_audio = true;
        } else {
            runtime::append_audio_buffer(combined_audio, audio);
        }

        if (continuity_chunk_buffer_size > 0) {
            continuity_history_codes.push_back(take_last_frames(raw_codes, continuity_max_frames));
            continuity_history_texts.push_back(higgs_request.text);
            while (static_cast<int64_t>(continuity_history_codes.size()) > continuity_chunk_buffer_size) {
                continuity_history_codes.erase(continuity_history_codes.begin());
            }
            while (static_cast<int64_t>(continuity_history_texts.size()) > continuity_chunk_buffer_size) {
                continuity_history_texts.erase(continuity_history_texts.begin());
            }
        }

        if (debug_prefill_probe_requested(request)) {
            break;
        }
    }

    if (!has_audio) {
        throw std::runtime_error("Higgs TTS produced no audio output across all text chunks");
    }

    result.audio_output = std::move(combined_audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

}  // namespace engine::models::higgs_tts
