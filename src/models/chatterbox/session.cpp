#include "engine/models/chatterbox/session.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/chatterbox/components.h"
#include "engine/models/chatterbox/s3gen_flow.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

constexpr int64_t kDefaultTextChunkSize = 128;

ChatterboxVoiceCloneConfig make_voice_clone_config(
    const std::unordered_map<std::string, std::string> & options) {
    ChatterboxVoiceCloneConfig config;
    config.exaggeration = runtime::parse_float_option(options, {"exaggeration"})
        .value_or(config.exaggeration);
    config.guidance_scale = runtime::parse_float_option(options, {"guidance_scale"})
        .value_or(config.guidance_scale);
    config.temperature = runtime::parse_float_option(options, {"temperature"})
        .value_or(config.temperature);
    config.repetition_penalty = runtime::parse_float_option(
        options,
        {"repetition_penalty"})
        .value_or(config.repetition_penalty);
    config.min_p = runtime::parse_float_option(options, {"min_p"}).value_or(config.min_p);
    config.top_p = runtime::parse_float_option(options, {"top_p"}).value_or(config.top_p);
    config.s3gen_cfg_rate = runtime::parse_float_option(
        options,
        {"s3gen_cfg_rate"})
        .value_or(config.s3gen_cfg_rate);
    config.max_new_tokens = runtime::parse_positive_i64_option(
        options,
        {"max_tokens"},
        config.max_new_tokens);
    config.seed = runtime::parse_u32_option(options, {"seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto value = runtime::find_option(options, {"do_sample"})) {
        config.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::find_option(options, {"stop_on_eos"})) {
        config.stop_on_eos = runtime::parse_bool_option(*value, "stop_on_eos");
    }
    if (const auto value = runtime::find_option(options, {"greedy"})) {
        if (runtime::parse_bool_option(*value, "greedy")) {
            config.do_sample = false;
        }
    }
    return config;
}

ChatterboxVoiceCloneConfig make_voice_clone_config(
    const std::unordered_map<std::string, std::string> & options,
    const std::string & language) {
    auto config = make_voice_clone_config(options);
    config.language = normalize_chatterbox_language_code(language);
    return config;
}

ChatterboxVoiceCloneConfig make_voice_clone_config(const runtime::TaskRequest & request) {
    return make_voice_clone_config(
        request.options,
        request.text_input.has_value() ? request.text_input->language : "en");
}

ChatterboxVoiceConversionConfig make_voice_conversion_config(const runtime::TaskRequest & request) {
    ChatterboxVoiceConversionConfig config;
    config.s3gen_cfg_rate = runtime::parse_float_option(request.options, {"s3gen_cfg_rate"})
        .value_or(config.s3gen_cfg_rate);
    config.num_steps = runtime::parse_positive_i64_option(request.options, {"num_inference_steps"}, config.num_steps);
    config.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    return config;
}

void validate_positive_audio(const runtime::AudioBuffer & audio, const char * role) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error(std::string("Chatterbox ") + role + " audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error(std::string("Chatterbox ") + role + " audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error(std::string("Chatterbox ") + role + " audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error(std::string("Chatterbox ") + role + " samples must be divisible by channels");
    }
}

runtime::AudioBuffer load_audio_buffer(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    return runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

runtime::AudioBuffer resolve_source_audio(const runtime::TaskRequest & request) {
    if (const auto path = runtime::find_option(request.options, {"source_audio"}); path.has_value()) {
        auto audio = load_audio_buffer(*path);
        validate_positive_audio(audio, "source");
        return audio;
    }
    if (request.audio_input.has_value()) {
        validate_positive_audio(*request.audio_input, "source");
        return *request.audio_input;
    }
    throw std::runtime_error("Chatterbox voice conversion requires audio_input or source_audio");
}

runtime::AudioBuffer resolve_target_audio(const runtime::TaskRequest & request) {
    if (const auto path = runtime::find_option(request.options, {"target_voice"}); path.has_value()) {
        auto audio = load_audio_buffer(*path);
        validate_positive_audio(audio, "target");
        return audio;
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->audio.has_value()) {
        validate_positive_audio(*request.voice->speaker->audio, "target");
        return *request.voice->speaker->audio;
    }
    throw std::runtime_error("Chatterbox voice conversion requires voice_ref or target_voice");
}

runtime::AudioBuffer slice_audio_frames(
    const runtime::AudioBuffer & audio,
    int64_t start_frame,
    int64_t frame_count) {
    if (audio.channels <= 0) {
        throw std::runtime_error("Chatterbox slice_audio_frames requires positive channel count");
    }
    const int64_t total_frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    if (start_frame < 0 || frame_count < 0 || start_frame > total_frames) {
        throw std::runtime_error("Chatterbox slice_audio_frames received invalid frame range");
    }
    const int64_t clamped_frames = std::min(frame_count, total_frames - start_frame);
    runtime::AudioBuffer out;
    out.sample_rate = audio.sample_rate;
    out.channels = audio.channels;
    const size_t begin = static_cast<size_t>(start_frame * audio.channels);
    const size_t end = static_cast<size_t>((start_frame + clamped_frames) * audio.channels);
    out.samples.assign(audio.samples.begin() + static_cast<std::ptrdiff_t>(begin), audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
    return out;
}

float resolve_vc_chunk_seconds(const runtime::TaskRequest & request) {
    return runtime::parse_float_option(
        request.options,
        {"chatterbox.vc_chunk_seconds", "vc_chunk_seconds", "audio_chunk_duration"})
        .value_or(15.0F);
}

float resolve_vc_chunk_threshold_seconds(const runtime::TaskRequest & request, float chunk_seconds) {
    return runtime::parse_float_option(
        request.options,
        {"chatterbox.vc_chunk_threshold_seconds", "vc_chunk_threshold_seconds", "audio_chunk_threshold"})
        .value_or(std::max(20.0F, chunk_seconds));
}

bool float_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

bool same_audio_buffer(const runtime::AudioBuffer & lhs, const runtime::AudioBuffer & rhs) {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.samples == rhs.samples;
}

bool same_voice_clone_config(const ChatterboxVoiceCloneConfig & lhs, const ChatterboxVoiceCloneConfig & rhs) {
    return float_equal(lhs.exaggeration, rhs.exaggeration) &&
        float_equal(lhs.guidance_scale, rhs.guidance_scale) &&
        float_equal(lhs.temperature, rhs.temperature) &&
        float_equal(lhs.repetition_penalty, rhs.repetition_penalty) &&
        float_equal(lhs.min_p, rhs.min_p) &&
        float_equal(lhs.top_p, rhs.top_p) &&
        float_equal(lhs.s3gen_cfg_rate, rhs.s3gen_cfg_rate) &&
        lhs.language == rhs.language &&
        lhs.max_new_tokens == rhs.max_new_tokens &&
        lhs.seed == rhs.seed &&
        lhs.do_sample == rhs.do_sample &&
        lhs.stop_on_eos == rhs.stop_on_eos;
}

ChatterboxPromptPrepConfig make_prompt_prep_config(const runtime::SessionOptions & options) {
    ChatterboxPromptPrepConfig config;
    config.encoder_condition_samples = runtime::parse_i64_option(
        options.options,
        {"chatterbox.encoder_condition_samples", "encoder_condition_samples"})
        .value_or(config.encoder_condition_samples);
    config.decoder_condition_samples = runtime::parse_i64_option(
        options.options,
        {"chatterbox.decoder_condition_samples", "decoder_condition_samples"})
        .value_or(config.decoder_condition_samples);
    config.t3_speech_cond_prompt_len = runtime::parse_i64_option(
        options.options,
        {"chatterbox.t3_speech_cond_prompt_len", "t3_speech_cond_prompt_len"})
        .value_or(config.t3_speech_cond_prompt_len);
    return config;
}

std::unique_ptr<ChatterboxTtsComponent> make_chatterbox_component_for_language(
    const ChatterboxAssetPaths & assets,
    const runtime::SessionOptions & options,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType t3_weight_storage_type,
    engine::assets::TensorStorageType component_weight_storage_type,
    bool mem_saver,
    const std::string & language) {
    const bool use_multilingual = chatterbox_language_uses_multilingual_t3(language);
    return std::make_unique<ChatterboxTtsComponent>(
        load_t3_inference_weights(
            use_multilingual ? assets.t3_multilingual_v2_weights : assets.t3_english_weights,
            execution_context,
            t3_weight_storage_type,
            false),
        load_chatterbox_english_tokenizer(
            use_multilingual ? assets.multilingual_tokenizer : assets.english_tokenizer),
        VoiceEncoderComponent::load_from_model_root(assets.model_root, options.backend),
        S3TokenizerComponent::load_from_checkpoint(
            assets.s3tokenizer_weights,
            execution_context,
            component_weight_storage_type),
        CAMPPlusEncoderComponent::load_from_checkpoint(
            assets.s3gen_weights,
            execution_context,
            component_weight_storage_type),
        load_s3_flow_encoder_weights(
            assets.s3gen_weights,
            execution_context,
            component_weight_storage_type),
        load_s3_flow_decoder_weights(
            assets.s3gen_weights,
            execution_context,
            component_weight_storage_type),
        HiFTVocoderComponent::load_from_checkpoint(
            assets.s3gen_weights,
            execution_context,
            component_weight_storage_type),
        make_prompt_prep_config(options),
        execution_context,
        mem_saver);
}

std::shared_ptr<const ChatterboxAssetPaths> require_assets(std::shared_ptr<const ChatterboxAssetPaths> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Chatterbox session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return fallback;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_chatterbox_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_chatterbox_options(const runtime::SessionOptions & options) {
    for (const auto & [key, _] : options.options) {
        if (key.rfind("chatterbox.", 0) == 0 &&
            key != "chatterbox.weight_type" &&
            key != "chatterbox.t3_weight_type" &&
            key != "chatterbox.conditionals_cache_slots" &&
            key != "chatterbox.mem_saver" &&
            key != "chatterbox.encoder_condition_samples" &&
            key != "chatterbox.decoder_condition_samples" &&
            key != "chatterbox.t3_speech_cond_prompt_len") {
            throw std::runtime_error("unknown Chatterbox session option: " + key);
        }
    }
}

engine::assets::TensorStorageType resolve_t3_weight_storage_type(const runtime::SessionOptions & options) {
    validate_chatterbox_options(options);
    const auto storage_type = option_weight_type(
        options,
        "chatterbox.t3_weight_type",
        option_weight_type(options, "chatterbox.weight_type", engine::assets::TensorStorageType::Native));
    validate_chatterbox_weight_storage(storage_type, "chatterbox.t3_weight_type");
    return storage_type;
}

engine::assets::TensorStorageType resolve_component_weight_storage_type(const runtime::SessionOptions & options) {
    const auto storage_type = option_weight_type(
        options,
        "chatterbox.weight_type",
        engine::assets::TensorStorageType::Native);
    validate_chatterbox_weight_storage(storage_type, "chatterbox.weight_type");
    return storage_type;
}

std::size_t resolve_conditionals_cache_slots(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultConditionalsCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"chatterbox.conditionals_cache_slots", "conditionals_cache_slots"})
        .value_or(kDefaultConditionalsCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("chatterbox.conditionals_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("chatterbox.conditionals_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

bool resolve_mem_saver(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"chatterbox.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "chatterbox.mem_saver");
    }
    return false;
}

}  // namespace

bool ChatterboxConditionalsCacheKeyEqual::operator()(
    const ChatterboxConditionalsCacheKey & lhs,
    const ChatterboxConditionalsCacheKey & rhs) const {
    return lhs.language == rhs.language &&
        float_equal(lhs.exaggeration, rhs.exaggeration) &&
        same_audio_buffer(lhs.reference_audio, rhs.reference_audio);
}

ChatterboxSession::ChatterboxSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ChatterboxAssetPaths> assets)
    : RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      t3_weight_storage_type_(resolve_t3_weight_storage_type(this->options())),
      component_weight_storage_type_(resolve_component_weight_storage_type(this->options())),
      mem_saver_(resolve_mem_saver(this->options())),
      conditionals_cache_(resolve_conditionals_cache_slots(this->options())) {
    if (task_.task != runtime::VoiceTaskKind::VoiceCloning &&
        task_.task != runtime::VoiceTaskKind::VoiceConversion) {
        throw std::runtime_error("Chatterbox session only supports --task clon or --task vc");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Chatterbox session only supports offline mode");
    }
}

ChatterboxSession::~ChatterboxSession() = default;

std::string ChatterboxSession::family() const {
    return "chatterbox";
}

runtime::VoiceTaskKind ChatterboxSession::task_kind() const {
    return task_.task;
}

runtime::RunMode ChatterboxSession::run_mode() const {
    return task_.mode;
}

void ChatterboxSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (task_.task == runtime::VoiceTaskKind::VoiceConversion) {
        if (!component_) {
            component_ = make_chatterbox_component_for_language(
                *assets_,
                this->options(),
                execution_context(),
                t3_weight_storage_type_,
                component_weight_storage_type_,
                mem_saver_,
                "en");
            component_language_ = "en";
        }
        mark_prepared();
        return;
    }

    if (!request.text.has_value() || request.text->text.empty()) {
        throw std::runtime_error("Chatterbox prepare requires text input");
    }
    if (!request.voice.has_value() || !request.voice->speaker.has_value() || !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("Chatterbox prepare requires speaker reference audio");
    }
    const auto session_config = make_voice_clone_config(request.options, request.text->language);
    voice_clone_config_ = session_config;
    if (!component_ || !component_language_.has_value() || *component_language_ != session_config.language) {
        component_.reset();
        component_language_.reset();
        component_ = make_chatterbox_component_for_language(
            *assets_,
            this->options(),
            execution_context(),
            t3_weight_storage_type_,
            component_weight_storage_type_,
            mem_saver_,
            session_config.language);
        component_language_ = session_config.language;
        cached_conditionals_.reset();
        conditionals_cache_.clear();
    }
    const auto & reference_audio = *request.voice->speaker->audio;
    ChatterboxConditionalsCacheKey conditionals_key{
        reference_audio,
        session_config.exaggeration,
        session_config.language,
    };
    const auto * conditionals_cache_entry = conditionals_cache_.find(conditionals_key);
    const bool conditionals_cache_hit =
        conditionals_cache_entry != nullptr;
    engine::debug::trace_log_scalar("chatterbox.conditionals.cache_hit", conditionals_cache_hit ? 1 : 0);
    engine::debug::trace_log_scalar(
        "chatterbox.conditionals.cache_slots",
        static_cast<int64_t>(conditionals_cache_.capacity()));
    if (conditionals_cache_hit) {
        cached_conditionals_ = *conditionals_cache_entry;
        cached_prompt_prep_ms_ = 0.0;
    } else {
        const auto prompt_prep_started = std::chrono::steady_clock::now();
        auto prepared_conditionals = component_->prepare_voice_clone_conditionals(
            reference_audio,
            session_config);
        cached_conditionals_ = prepared_conditionals;
        conditionals_cache_.put(std::move(conditionals_key), std::move(prepared_conditionals));
        cached_prompt_prep_ms_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prompt_prep_started).count();
    }
    mark_prepared();
}

runtime::TaskResult ChatterboxSession::run(const runtime::TaskRequest & request) {
    require_prepared("Chatterbox run()");
    if (task_.task == runtime::VoiceTaskKind::VoiceConversion) {
        return run_voice_conversion(request);
    }
    return run_voice_cloning(request);
}

runtime::TaskResult ChatterboxSession::run_voice_cloning(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Chatterbox voice cloning requires text_input");
    }
    if (!voice_clone_config_.has_value() || !cached_conditionals_.has_value()) {
        throw std::runtime_error("Chatterbox voice cloning requires cached prepared conditionals");
    }
    if (!component_) {
        throw std::runtime_error("Chatterbox voice cloning requires prepared model component");
    }
    auto request_config = make_voice_clone_config(request);
    if (!runtime::parse_u32_option(request.options, {"seed"}).has_value()) {
        request_config.seed = voice_clone_config_->seed;
    }
    if (!same_voice_clone_config(*voice_clone_config_, request_config)) {
        throw std::runtime_error("Chatterbox voice cloning session config is fixed; create a new session for different config");
    }

    runtime::TaskResult result;
    runtime::AudioBuffer merged_audio;
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    engine::debug::trace_log_scalar("chatterbox.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("chatterbox.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));
    for (const auto & chunk_request : chunk_requests) {
        auto outputs = component_->synthesize_voice_clone_with_conditionals(
            chunk_request.text_input->text,
            *cached_conditionals_,
            *voice_clone_config_);
        outputs.prompt_prep_ms = cached_prompt_prep_ms_;
        runtime::append_audio_buffer(merged_audio, runtime::AudioBuffer{
            24000,
            1,
            std::move(outputs.waveform),
        });
    }
    result.audio_output = std::move(merged_audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::TaskResult ChatterboxSession::run_voice_conversion(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    if (!component_) {
        throw std::runtime_error("Chatterbox voice conversion requires prepared model component");
    }
    const auto source_audio = resolve_source_audio(request);
    const auto target_audio = resolve_target_audio(request);
    const auto config = make_voice_conversion_config(request);
    const float chunk_seconds = resolve_vc_chunk_seconds(request);
    const float chunk_threshold_seconds = resolve_vc_chunk_threshold_seconds(request, chunk_seconds);
    if (!(chunk_seconds > 0.0F) || !(chunk_threshold_seconds > 0.0F)) {
        throw std::runtime_error("Chatterbox voice conversion chunk settings must be positive");
    }

    const int64_t total_frames = static_cast<int64_t>(source_audio.samples.size() / static_cast<size_t>(source_audio.channels));
    const int64_t chunk_frames = static_cast<int64_t>(chunk_seconds * static_cast<float>(source_audio.sample_rate));
    const int64_t threshold_frames = static_cast<int64_t>(chunk_threshold_seconds * static_cast<float>(source_audio.sample_rate));
    if (chunk_frames <= 0 || threshold_frames <= 0) {
        throw std::runtime_error("Chatterbox voice conversion computed invalid chunk frame counts");
    }

    runtime::TaskResult result;
    runtime::AudioBuffer merged_audio;
    if (total_frames > threshold_frames) {
        const int64_t chunk_count = (total_frames + chunk_frames - 1) / chunk_frames;
        engine::debug::trace_log_scalar("chatterbox.voice_conversion.chunk_seconds", static_cast<double>(chunk_seconds));
        engine::debug::trace_log_scalar("chatterbox.voice_conversion.chunk_threshold_seconds", static_cast<double>(chunk_threshold_seconds));
        engine::debug::trace_log_scalar("chatterbox.voice_conversion.chunk_count", chunk_count);
        for (int64_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
            const int64_t start_frame = chunk_index * chunk_frames;
            const auto chunk_audio = slice_audio_frames(source_audio, start_frame, chunk_frames);
            auto outputs = component_->synthesize_voice_conversion(chunk_audio, target_audio, config);
            runtime::append_audio_buffer(merged_audio, runtime::AudioBuffer{24000, 1, std::move(outputs.waveform)});
        }
    } else {
        auto outputs = component_->synthesize_voice_conversion(source_audio, target_audio, config);
        merged_audio = runtime::AudioBuffer{24000, 1, std::move(outputs.waveform)};
    }

    result.audio_output = std::move(merged_audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::chatterbox
