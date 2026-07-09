#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/codec_encoder.h"
#include "engine/models/higgs_tts/codec_decoder.h"
#include "engine/models/higgs_tts/generator.h"
#include "engine/models/higgs_tts/tokenizer_text.h"
#include "engine/models/higgs_tts/types.h"
#include "engine/models/higgs_tts/weights.h"

#include <memory>
#include <string>

namespace engine::models::higgs_tts {

class HiggsTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    HiggsTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const HiggsAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    HiggsTTSRequest make_request(const runtime::TaskRequest & request) const;
    HiggsGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) const;
    void validate_request(const runtime::TaskRequest & request) const;
    bool debug_prefill_probe_requested(const runtime::TaskRequest & request) const;
    void ensure_probe_runtimes();
    std::vector<float> build_prompt_embeddings(
        const HiggsTextPrompt & prompt,
        const std::vector<HiggsCodeSequence> & delayed_reference_segments = {});
    HiggsCodeSequence generate_delayed_codes(
        const std::vector<float> & prompt_embeddings,
        int64_t prompt_steps,
        const HiggsGenerationOptions & options);

    runtime::TaskSpec task_;
    std::shared_ptr<const HiggsAssets> assets_;
    size_t weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t text_embedding_graph_context_bytes_ = 8ull * 1024ull * 1024ull;
    size_t codebook_embedding_graph_context_bytes_ = 32ull * 1024ull * 1024ull;
    size_t prefill_graph_context_bytes_ = 256ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_graph_context_bytes_ = 512ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType weight_storage_type_ = engine::assets::TensorStorageType::Native;
    HiggsTextTokenizer text_tokenizer_;
    std::shared_ptr<const HiggsBackboneWeightsRuntime> backbone_weights_;
    std::unique_ptr<HiggsTextEmbeddingRuntime> text_embedding_runtime_;
    std::unique_ptr<HiggsCodebookEmbeddingRuntime> codebook_embedding_runtime_;
    std::unique_ptr<HiggsPromptPrefillRuntime> prompt_prefill_runtime_;
    std::unique_ptr<HiggsCodecEncoderRuntime> codec_encoder_runtime_;
    std::unique_ptr<HiggsCodecDecoderRuntime> codec_decoder_runtime_;
};

}  // namespace engine::models::higgs_tts
