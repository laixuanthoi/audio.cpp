#pragma once

#include "engine/models/higgs_tts/types.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace engine::models::higgs_tts {

constexpr int32_t kHiggsCodecBocId = 1024;
constexpr int32_t kHiggsCodecEocId = 1025;

constexpr std::string_view kHiggsTextEmbeddingWeightName = "tied.embedding.text_embedding.weight";
constexpr std::string_view kHiggsModalityEmbeddingWeightName = "tied.embedding.modality_embeddings.0.embedding.weight";
constexpr std::string_view kHiggsCodecPrefix = "tied.embedding.modality_embeddings.0.model.";
constexpr std::string_view kHiggsBodyNormWeightName = "body.norm.weight";

HiggsCodeSequence apply_delay_pattern(
    const HiggsCodeSequence & codes,
    int32_t boc_id = kHiggsCodecBocId,
    int32_t eoc_id = kHiggsCodecEocId);

HiggsCodeSequence reverse_delay_pattern(
    const HiggsCodeSequence & delayed_codes,
    int32_t boc_id = kHiggsCodecBocId,
    int32_t eoc_id = kHiggsCodecEocId);

HiggsCodeSequence fuse_codebook_ids(
    const HiggsCodeSequence & codes,
    int64_t vocab_size);

int64_t delayed_frame_count(int64_t raw_frames, int64_t num_codebooks);
int64_t raw_frame_count_from_delayed(int64_t delayed_frames, int64_t num_codebooks);
int64_t fused_embedding_rows(int64_t num_codebooks, int64_t vocab_size);

std::vector<int32_t> code_row(const HiggsCodeSequence & codes, int64_t frame_index);

}  // namespace engine::models::higgs_tts
