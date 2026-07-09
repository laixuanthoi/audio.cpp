#include "engine/models/higgs_tts/utils.h"

#include <stdexcept>

namespace engine::models::higgs_tts {
namespace {

void validate_code_sequence(const HiggsCodeSequence & codes, const char * label) {
    if (codes.frames < 0 || codes.num_codebooks <= 0) {
        throw std::runtime_error(std::string(label) + " dimensions must be non-negative with positive codebook count");
    }
    const int64_t expected = codes.frames * codes.num_codebooks;
    if (expected != static_cast<int64_t>(codes.values.size())) {
        throw std::runtime_error(std::string(label) + " values size does not match frames * num_codebooks");
    }
}

size_t flat_index(const HiggsCodeSequence & codes, int64_t frame_index, int64_t codebook_index) {
    return static_cast<size_t>(frame_index * codes.num_codebooks + codebook_index);
}

}  // namespace

int64_t delayed_frame_count(int64_t raw_frames, int64_t num_codebooks) {
    if (raw_frames < 0 || num_codebooks <= 0) {
        throw std::runtime_error("Higgs delayed_frame_count expects non-negative raw_frames and positive num_codebooks");
    }
    return raw_frames == 0 ? 0 : raw_frames + num_codebooks - 1;
}

int64_t raw_frame_count_from_delayed(int64_t delayed_frames, int64_t num_codebooks) {
    if (delayed_frames < 0 || num_codebooks <= 0) {
        throw std::runtime_error("Higgs raw_frame_count_from_delayed expects non-negative delayed_frames and positive num_codebooks");
    }
    if (delayed_frames == 0) {
        return 0;
    }
    if (delayed_frames < num_codebooks) {
        throw std::runtime_error("Higgs delayed code sequence is shorter than the number of codebooks");
    }
    return delayed_frames - num_codebooks + 1;
}

int64_t fused_embedding_rows(int64_t num_codebooks, int64_t vocab_size) {
    if (num_codebooks <= 0 || vocab_size <= 0) {
        throw std::runtime_error("Higgs fused_embedding_rows expects positive dimensions");
    }
    return num_codebooks * vocab_size;
}

HiggsCodeSequence apply_delay_pattern(
    const HiggsCodeSequence & codes,
    int32_t boc_id,
    int32_t eoc_id) {
    validate_code_sequence(codes, "Higgs raw code sequence");
    HiggsCodeSequence out;
    out.frames = delayed_frame_count(codes.frames, codes.num_codebooks);
    out.num_codebooks = codes.num_codebooks;
    out.values.assign(static_cast<size_t>(out.frames * out.num_codebooks), eoc_id);
    if (codes.frames == 0) {
        return out;
    }
    for (int64_t codebook = 0; codebook < codes.num_codebooks; ++codebook) {
        for (int64_t frame = 0; frame < out.frames; ++frame) {
            const size_t dst = flat_index(out, frame, codebook);
            if (frame < codebook) {
                out.values[dst] = boc_id;
            } else if (frame < codebook + codes.frames) {
                const int64_t source_frame = frame - codebook;
                out.values[dst] = codes.values[flat_index(codes, source_frame, codebook)];
            }
        }
    }
    return out;
}

HiggsCodeSequence reverse_delay_pattern(
    const HiggsCodeSequence & delayed_codes,
    int32_t,
    int32_t) {
    validate_code_sequence(delayed_codes, "Higgs delayed code sequence");
    HiggsCodeSequence out;
    out.frames = raw_frame_count_from_delayed(delayed_codes.frames, delayed_codes.num_codebooks);
    out.num_codebooks = delayed_codes.num_codebooks;
    out.values.resize(static_cast<size_t>(out.frames * out.num_codebooks));
    if (out.frames == 0) {
        return out;
    }
    for (int64_t codebook = 0; codebook < out.num_codebooks; ++codebook) {
        for (int64_t frame = 0; frame < out.frames; ++frame) {
            out.values[flat_index(out, frame, codebook)] =
                delayed_codes.values[flat_index(delayed_codes, frame + codebook, codebook)];
        }
    }
    return out;
}

HiggsCodeSequence fuse_codebook_ids(const HiggsCodeSequence & codes, int64_t vocab_size) {
    validate_code_sequence(codes, "Higgs code sequence");
    if (vocab_size <= 0) {
        throw std::runtime_error("Higgs fuse_codebook_ids expects positive vocab_size");
    }
    HiggsCodeSequence out = codes;
    for (int64_t codebook = 0; codebook < codes.num_codebooks; ++codebook) {
        const int64_t offset = codebook * vocab_size;
        for (int64_t frame = 0; frame < codes.frames; ++frame) {
            const size_t index = flat_index(codes, frame, codebook);
            const int32_t value = codes.values[index];
            if (value < 0 || value >= vocab_size) {
                throw std::runtime_error("Higgs code token is out of range for the configured codebook vocab size");
            }
            out.values[index] = static_cast<int32_t>(offset + value);
        }
    }
    return out;
}

std::vector<int32_t> code_row(const HiggsCodeSequence & codes, int64_t frame_index) {
    validate_code_sequence(codes, "Higgs code sequence");
    if (frame_index < 0 || frame_index >= codes.frames) {
        throw std::runtime_error("Higgs code_row frame index is out of range");
    }
    const size_t start = static_cast<size_t>(frame_index * codes.num_codebooks);
    return std::vector<int32_t>(
        codes.values.begin() + static_cast<std::ptrdiff_t>(start),
        codes.values.begin() + static_cast<std::ptrdiff_t>(start + static_cast<size_t>(codes.num_codebooks)));
}

}  // namespace engine::models::higgs_tts
