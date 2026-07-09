#include "engine/models/higgs_tts/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::higgs_tts {
namespace {

constexpr float kGreedyTemperatureThreshold = 1.0e-5F;

std::vector<float> filtered_logits(
    const std::vector<float> & logits,
    const HiggsGenerationOptions & options) {
    if (options.temperature <= kGreedyTemperatureThreshold) {
        return logits;
    }

    std::vector<float> scaled(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled[i] = logits[i] / options.temperature;
    }

    if (options.top_k.has_value() && *options.top_k > 0 && *options.top_k < static_cast<int>(scaled.size())) {
        std::vector<float> sorted = scaled;
        std::nth_element(sorted.begin(), sorted.begin() + (*options.top_k - 1), sorted.end(), std::greater<float>());
        const float threshold = sorted[static_cast<size_t>(*options.top_k - 1)];
        for (float & value : scaled) {
            if (value < threshold) {
                value = -INFINITY;
            }
        }
    }

    if (options.top_p.has_value() && *options.top_p > 0.0F && *options.top_p < 1.0F) {
        std::vector<int> order(static_cast<int>(scaled.size()));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs) { return scaled[lhs] > scaled[rhs]; });

        float max_logit = -INFINITY;
        for (const float value : scaled) {
            max_logit = std::max(max_logit, value);
        }
        std::vector<float> probs(order.size(), 0.0F);
        float sum = 0.0F;
        for (size_t i = 0; i < order.size(); ++i) {
            if (std::isfinite(scaled[static_cast<size_t>(order[i])])) {
                probs[i] = std::exp(scaled[static_cast<size_t>(order[i])] - max_logit);
                sum += probs[i];
            }
        }
        if (sum > 0.0F) {
            float cumulative = 0.0F;
            for (size_t i = 0; i < order.size(); ++i) {
                cumulative += probs[i] / sum;
                if (i > 0 && cumulative > *options.top_p) {
                    scaled[static_cast<size_t>(order[i])] = -INFINITY;
                }
            }
        }
    }

    return scaled;
}

int32_t sample_one(
    const std::vector<float> & logits,
    const HiggsGenerationOptions & options,
    uint64_t draw_index) {
    if (logits.empty()) {
        throw std::runtime_error("Higgs sampler received an empty logits vector");
    }
    if (options.temperature <= kGreedyTemperatureThreshold) {
        return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    const auto filtered = filtered_logits(logits, options);
    float max_logit = -INFINITY;
    for (const float value : filtered) {
        max_logit = std::max(max_logit, value);
    }
    if (!std::isfinite(max_logit)) {
        return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    std::vector<double> probs(filtered.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < filtered.size(); ++i) {
        if (std::isfinite(filtered[i])) {
            probs[i] = std::exp(static_cast<double>(filtered[i] - max_logit));
            sum += probs[i];
        }
    }
    if (!(sum > 0.0)) {
        return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    for (double & value : probs) {
        value /= sum;
    }

    std::mt19937 rng;
    if (options.seed.has_value()) {
        rng.seed(*options.seed + static_cast<uint32_t>(draw_index & 0xFFFFFFFFu));
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

void validate_state_shape(const HiggsSamplerState & state) {
    if (state.num_codebooks <= 0) {
        throw std::runtime_error("Higgs sampler state must declare a positive number of codebooks");
    }
    if (!state.last_codes.empty() && static_cast<int64_t>(state.last_codes.size()) != state.num_codebooks) {
        throw std::runtime_error("Higgs sampler last_codes size does not match num_codebooks");
    }
}

}  // namespace

std::vector<int32_t> higgs_sampler_step(
    const std::vector<float> & logits,
    int64_t vocab_size,
    HiggsSamplerState & state,
    const HiggsGenerationOptions & options,
    int32_t boc_id,
    int32_t eoc_id) {
    validate_state_shape(state);
    if (vocab_size <= 0) {
        throw std::runtime_error("Higgs sampler expects a positive vocab size");
    }
    const int64_t num_codebooks = state.num_codebooks;
    if (static_cast<int64_t>(logits.size()) != num_codebooks * vocab_size) {
        throw std::runtime_error("Higgs sampler logits size does not match num_codebooks * vocab_size");
    }
    if (state.generation_done) {
        return std::vector<int32_t>(static_cast<size_t>(num_codebooks), kHiggsStopCode);
    }

    std::vector<int32_t> codes(static_cast<size_t>(num_codebooks), 0);
    for (int64_t codebook = 0; codebook < num_codebooks; ++codebook) {
        const size_t start = static_cast<size_t>(codebook * vocab_size);
        codes[static_cast<size_t>(codebook)] = sample_one(
            std::vector<float>(logits.begin() + static_cast<std::ptrdiff_t>(start),
                               logits.begin() + static_cast<std::ptrdiff_t>(start + static_cast<size_t>(vocab_size))),
            options,
            state.sample_index * static_cast<uint64_t>(num_codebooks) + static_cast<uint64_t>(codebook));
    }
    ++state.sample_index;

    if (state.delay_count < num_codebooks) {
        const int64_t next_codebook = state.delay_count + 1;
        if (next_codebook < num_codebooks) {
            for (int64_t codebook = next_codebook; codebook < num_codebooks; ++codebook) {
                codes[static_cast<size_t>(codebook)] = boc_id;
            }
        }
        ++state.delay_count;
    } else if (state.eoc_countdown.has_value()) {
        --*state.eoc_countdown;
        if (*state.eoc_countdown <= 0) {
            state.generation_done = true;
        }
    } else if (codes.front() == eoc_id) {
        if (num_codebooks <= 2) {
            state.generation_done = true;
        } else {
            state.eoc_countdown = num_codebooks - 2;
        }
    }

    if (!state.generation_done) {
        state.last_codes = codes;
    }
    return codes;
}

}  // namespace engine::models::higgs_tts
