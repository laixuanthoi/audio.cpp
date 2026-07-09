#include "engine/models/higgs_tts/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/higgs_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Higgs TTS model path does not exist: " + model_path.string());
}

bool has_higgs_tts_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json")
        && engine::io::is_existing_file(root / "model.safetensors")
        && engine::io::is_existing_file(root / "tokenizer_config.json")
        && engine::io::is_existing_file(root / "tokenizer.json")
        && engine::io::is_existing_file(root / "chat_template.jinja");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors", "model.safetensors.index.json"});
}

runtime::CapabilitySet higgs_tts_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"Auto"};
    capabilities.supports_speaker_reference = true;
    return capabilities;
}

runtime::ModelMetadata higgs_tts_metadata(const HiggsAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "higgs_tts";
    metadata.variant = assets.config.architecture;
    metadata.description = "Boson Higgs Audio v3 TTS assets loaded from a local checkpoint.";
    metadata.config_candidates = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    };
    metadata.weight_candidates = {"model.safetensors", "model.safetensors.index.json"};
    return metadata;
}

class HiggsTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "higgs_tts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_higgs_tts_assets(root)
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_higgs_tts_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata = higgs_tts_metadata(*assets);
        inspection.capabilities = higgs_tts_capabilities();
        inspection.cli.request_options = {
            {"reference_text", "text", "Transcript of the reference voice when known."},
            {"scene_prompt", "text", "Boson-style scene description placed into the Higgs prompt context."},
            {"voice_profile", "name-or-text", "Boson-style speaker profile preset such as profile:male_en_british, or a raw speaker-description string."},
            {"emotion", "name", "Boson Higgs emotion control token, e.g. anger, sadness, or determination."},
            {"style", "name[,name...]", "Boson Higgs style token(s), e.g. singing, shouting, or whispering."},
            {"prosody", "name[,name...]", "Boson Higgs prosody token(s), e.g. speed_fast, pause, or expressive_high."},
            {"sfx", "name[,name...]", "Boson Higgs sound-effect token(s), e.g. laughter, cough, or humming."},
            {"text_chunk_size", "chars", "Sequential long-form chunk size for Higgs text generation."},
            {"generation_chunk_buffer_size", "n", "How many prior generated chunks to keep as prompt continuity context."},
            {"longform_context_max_frames", "n", "Maximum raw audio frames kept from each prior chunk for continuity."},
            {"max_tokens", "n", "Maximum generated audio-token steps per chunk."},
            {"temperature", "float", "Sampling temperature."},
            {"top_p", "float", "Top-p sampling."},
            {"top_k", "n", "Top-k sampling."},
            {"seed", "n", "Sampling seed."},
        };
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_higgs_tts_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

HiggsTTSLoadedModel::HiggsTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HiggsAssets> assets)
    : metadata_(std::move(metadata))
    , capabilities_(std::move(capabilities))
    , assets_(std::move(assets)) {}

const runtime::ModelMetadata & HiggsTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HiggsTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HiggsTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS native skeleton currently supports only offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS native skeleton currently supports only the Tts task");
    }
    return std::make_unique<HiggsTTSSession>(task, options, assets_);
}

std::unique_ptr<HiggsTTSLoadedModel> load_higgs_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_higgs_tts_assets(model_path);
    return std::make_unique<HiggsTTSLoadedModel>(
        higgs_tts_metadata(*assets),
        higgs_tts_capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_tts_loader() {
    return std::make_shared<HiggsTTSLoader>();
}

}  // namespace engine::models::higgs_tts
