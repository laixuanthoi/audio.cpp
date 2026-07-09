#pragma once

#include "engine/models/higgs_tts/types.h"
#include "engine/models/higgs_tts/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::higgs_tts {

struct HiggsEmbeddedSequence {
    std::vector<float> values;
    int64_t steps = 0;
    int64_t hidden_size = 0;
};

struct HiggsPromptPrefillInput {
    std::vector<float> input_embeddings;
    int64_t steps = 0;
};

struct HiggsPromptPrefillOutput {
    std::vector<float> last_hidden;
    std::vector<float> logits;
    int64_t num_codebooks = 0;
    int64_t vocab_size = 0;
};

class HiggsTextEmbeddingRuntime final {
public:
    HiggsTextEmbeddingRuntime(
        std::shared_ptr<const HiggsBackboneWeightsRuntime> weights,
        size_t graph_context_bytes);
    ~HiggsTextEmbeddingRuntime();

    std::vector<float> embed_token(int32_t token_id);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsCodebookEmbeddingRuntime final {
public:
    HiggsCodebookEmbeddingRuntime(
        std::shared_ptr<const HiggsBackboneWeightsRuntime> weights,
        size_t graph_context_bytes);
    ~HiggsCodebookEmbeddingRuntime();

    HiggsEmbeddedSequence embed_codes(const HiggsCodeSequence & delayed_codes);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsPromptPrefillRuntime final {
public:
    HiggsPromptPrefillRuntime(
        std::shared_ptr<const HiggsBackboneWeightsRuntime> weights,
        size_t graph_context_bytes);
    ~HiggsPromptPrefillRuntime();

    HiggsPromptPrefillOutput run(const HiggsPromptPrefillInput & input);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_tts
