#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::higgs_tts {

struct HiggsAudioEncoderConfig {
    std::string encoder_type = "discrete";
    int64_t max_chunk_size = 0;
    int64_t mel_per_sample = 0;
    int64_t num_codebooks = 0;
    int64_t out_dim = 0;
    bool tie_word_embeddings = true;
    bool use_delay_pattern = true;
    int64_t vocab_size = 0;
};

struct HiggsTextConfig {
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 0;
    int64_t head_dim = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
    bool use_cache = true;
    bool use_sliding_window = false;
};

struct HiggsConfig {
    std::string architecture;
    std::string model_type;
    int64_t audio_token_id = -100;
    int64_t ignore_index = -100;
    HiggsAudioEncoderConfig audio_encoder;
    HiggsTextConfig text;
};

struct HiggsAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path config_path;
    std::filesystem::path model_weights_path;
    std::filesystem::path model_weights_index_path;
    std::filesystem::path tokenizer_config_path;
    std::filesystem::path tokenizer_json_path;
    std::filesystem::path chat_template_path;
};

struct HiggsAssets {
    HiggsAssetPaths paths;
    HiggsConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

HiggsAssetPaths resolve_higgs_tts_assets(const std::filesystem::path & model_path);
std::shared_ptr<const HiggsAssets> load_higgs_tts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::higgs_tts
