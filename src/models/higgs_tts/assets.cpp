#include "engine/models/higgs_tts/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/models/higgs_tts/codec.h"
#include "engine/models/higgs_tts/utils.h"

#include <stdexcept>
#include <string>

namespace engine::models::higgs_tts {
namespace json = engine::io::json;
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

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"config", "config.json", true},
        {"weights", "model.safetensors", true},
        {"weights_index", "model.safetensors.index.json", false},
        {"tokenizer_config", "tokenizer_config.json", true},
        {"tokenizer_json", "tokenizer.json", true},
        {"chat_template", "chat_template.jinja", true},
    });
    return resources;
}

void require_positive(int64_t value, const char * label) {
    if (value <= 0) {
        throw std::runtime_error(std::string("Higgs TTS config contains non-positive ") + label);
    }
}

std::string require_architecture_name(const engine::io::json::Value & root) {
    const auto & architectures = root.require("architectures").as_array();
    if (architectures.empty() || !architectures.front().is_string()) {
        throw std::runtime_error("Higgs TTS config architectures must contain a model architecture name");
    }
    return architectures.front().as_string();
}

HiggsAudioEncoderConfig parse_audio_encoder_config(const engine::io::json::Value & value) {
    HiggsAudioEncoderConfig config;
    config.encoder_type = json::optional_string(value, "encoder_type", config.encoder_type);
    config.max_chunk_size = json::optional_i64(value, "max_chunk_size", config.max_chunk_size);
    config.mel_per_sample = json::optional_i64(value, "mel_per_sample", config.mel_per_sample);
    config.num_codebooks = json::require_i64(value, "num_codebooks");
    config.out_dim = json::require_i64(value, "out_dim");
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.use_delay_pattern = json::optional_bool(value, "use_delay_pattern", config.use_delay_pattern);
    config.vocab_size = json::require_i64(value, "vocab_size");
    require_positive(config.num_codebooks, "audio_encoder num_codebooks");
    require_positive(config.out_dim, "audio_encoder out_dim");
    require_positive(config.vocab_size, "audio_encoder vocab_size");
    if (config.encoder_type != "discrete") {
        throw std::runtime_error("Higgs TTS native integration currently supports only discrete audio encoders");
    }
    return config;
}

HiggsTextConfig parse_text_config(const engine::io::json::Value & value) {
    HiggsTextConfig config;
    config.bos_token_id = json::require_i64(value, "bos_token_id");
    config.eos_token_id = json::require_i64(value, "eos_token_id");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.use_cache = json::optional_bool(value, "use_cache", config.use_cache);
    config.use_sliding_window = json::optional_bool(value, "use_sliding_window", config.use_sliding_window);
    if (const auto * rope_parameters = value.find("rope_parameters"); rope_parameters != nullptr) {
        config.rope_theta = json::optional_f32(*rope_parameters, "rope_theta", config.rope_theta);
    } else {
        config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    }
    require_positive(config.hidden_size, "text hidden_size");
    require_positive(config.intermediate_size, "text intermediate_size");
    require_positive(config.max_position_embeddings, "text max_position_embeddings");
    require_positive(config.num_attention_heads, "text num_attention_heads");
    require_positive(config.num_hidden_layers, "text num_hidden_layers");
    require_positive(config.num_key_value_heads, "text num_key_value_heads");
    require_positive(config.head_dim, "text head_dim");
    require_positive(config.vocab_size, "text vocab_size");
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("Higgs TTS text num_attention_heads must be divisible by num_key_value_heads");
    }
    return config;
}

HiggsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    HiggsConfig config;
    config.architecture = require_architecture_name(root);
    config.model_type = json::require_string(root, "model_type");
    config.audio_token_id = json::optional_i64(root, "audio_token_id", config.audio_token_id);
    config.ignore_index = json::optional_i64(root, "ignore_index", config.ignore_index);
    config.audio_encoder = parse_audio_encoder_config(root.require("audio_encoder_config"));
    config.text = parse_text_config(root.require("text_config"));
    if (config.architecture != "HiggsMultimodalQwen3ForConditionalGeneration") {
        throw std::runtime_error("Higgs TTS config architecture mismatch: " + config.architecture);
    }
    if (config.model_type != "higgs_multimodal_qwen3") {
        throw std::runtime_error("Higgs TTS config model_type mismatch: " + config.model_type);
    }
    if (config.audio_token_id != config.ignore_index) {
        throw std::runtime_error("Higgs TTS currently expects audio_token_id to match ignore_index");
    }
    if (config.audio_encoder.out_dim != config.text.hidden_size) {
        throw std::runtime_error("Higgs TTS audio encoder out_dim must match text hidden_size");
    }
    return config;
}

void fill_paths(HiggsAssetPaths & paths, assets::ResourceBundle & resources) {
    paths.model_root = resources.model_root();
    paths.config_path = resources.require_file("config");
    paths.model_weights_path = resources.require_file("weights");
    if (const auto * path = resources.find_file("weights_index"); path != nullptr) {
        paths.model_weights_index_path = *path;
    }
    paths.tokenizer_config_path = resources.require_file("tokenizer_config");
    paths.tokenizer_json_path = resources.require_file("tokenizer_json");
    paths.chat_template_path = resources.require_file("chat_template");
}

void require_tensor_shape(
    const assets::TensorSource & source,
    std::string_view name,
    std::initializer_list<int64_t> expected_shape) {
    const auto metadata = source.require_metadata(name);
    const std::vector<int64_t> expected(expected_shape);
    if (metadata.shape != expected) {
        throw std::runtime_error(
            std::string("Higgs TTS weight shape mismatch for ") + std::string(name)
            + ": expected rank-shape anchor not found");
    }
}

void require_tensor_exists(const assets::TensorSource & source, std::string_view name) {
    (void) source.require_metadata(name);
}

void validate_weight_anchors(const HiggsAssets & assets) {
    const auto & weights = *assets.model_weights;
    const auto & codec = *assets.codec_weights;
    const auto & text = assets.config.text;
    const auto & audio = assets.config.audio_encoder;
    require_tensor_shape(weights, kHiggsTextEmbeddingWeightName, {text.vocab_size, text.hidden_size});
    require_tensor_shape(weights, kHiggsModalityEmbeddingWeightName, {audio.num_codebooks * audio.vocab_size, text.hidden_size});
    require_tensor_shape(weights, kHiggsBodyNormWeightName, {text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.input_layernorm.weight", {text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.post_attention_layernorm.weight", {text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.self_attn.q_norm.weight", {text.head_dim});
    require_tensor_shape(weights, "body.layers.0.self_attn.k_norm.weight", {text.head_dim});
    require_tensor_shape(weights, "body.layers.0.self_attn.q_proj.weight", {text.num_attention_heads * text.head_dim, text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.self_attn.k_proj.weight", {text.num_key_value_heads * text.head_dim, text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.self_attn.v_proj.weight", {text.num_key_value_heads * text.head_dim, text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.self_attn.o_proj.weight", {text.hidden_size, text.num_attention_heads * text.head_dim});
    require_tensor_shape(weights, "body.layers.0.mlp.gate_proj.weight", {text.intermediate_size, text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.mlp.up_proj.weight", {text.intermediate_size, text.hidden_size});
    require_tensor_shape(weights, "body.layers.0.mlp.down_proj.weight", {text.hidden_size, text.intermediate_size});
    require_tensor_exists(codec, "acoustic_encoder.conv1.weight");
    require_tensor_exists(codec, "acoustic_decoder.conv1.weight");
    require_tensor_exists(codec, "quantizer.quantizers.0.codebook.embed");
    require_tensor_exists(codec, "semantic_model.encoder.layers.0.attention.q_proj.weight");
}

}  // namespace

HiggsAssetPaths resolve_higgs_tts_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    HiggsAssetPaths paths;
    fill_paths(paths, resources);
    return paths;
}

std::shared_ptr<const HiggsAssets> load_higgs_tts_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    HiggsAssets assets;
    fill_paths(assets.paths, resources);
    assets.config = parse_config(resources);
    assets.model_weights = resources.open_tensor_source("weights");
    assets.codec_weights = make_higgs_codec_tensor_source(assets.model_weights);
    validate_weight_anchors(assets);
    return std::make_shared<HiggsAssets>(std::move(assets));
}

}  // namespace engine::models::higgs_tts
