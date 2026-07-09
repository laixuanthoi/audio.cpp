#include "engine/models/higgs_tts/codec.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

class PrefixedTensorSource final : public assets::TensorSource {
public:
    PrefixedTensorSource(std::shared_ptr<const assets::TensorSource> base, std::string prefix)
        : base_(std::move(base)), prefix_(std::move(prefix)) {
        if (base_ == nullptr) {
            throw std::runtime_error("Higgs codec tensor source requires a base tensor source");
        }
        if (prefix_.empty()) {
            throw std::runtime_error("Higgs codec tensor source prefix must not be empty");
        }
    }

    const std::filesystem::path & source_path() const noexcept override {
        return base_->source_path();
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return base_->has_tensor(prefixed_name(name));
    }

    assets::TensorMetadata require_metadata(std::string_view name) const override {
        auto metadata = base_->require_metadata(prefixed_name(name));
        metadata.name = std::string(name);
        return metadata;
    }

    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> out;
        for (auto metadata : base_->tensors()) {
            if (metadata.name.rfind(prefix_, 0) != 0) {
                continue;
            }
            metadata.name.erase(0, prefix_.size());
            out.push_back(std::move(metadata));
        }
        return out;
    }

    void release_storage() const override {
        base_->release_storage();
    }

    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        auto tensor = base_->require_tensor_data(prefixed_name(name));
        tensor.metadata.name = std::string(name);
        return tensor;
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const override {
        return base_->require_f32(prefixed_name(name), expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const override {
        return base_->optional_f32(prefixed_name(name), expected_shape);
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        base_->set_backend_tensor(tensor, prefixed_name(name), storage_type, expected_shape);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        base_->set_backend_f32_tensor(tensor, prefixed_name(name), expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return base_->require_i64_scalar(prefixed_name(name));
    }

private:
    std::string prefixed_name(std::string_view name) const {
        return prefix_ + std::string(name);
    }

    std::shared_ptr<const assets::TensorSource> base_;
    std::string prefix_;
};

}  // namespace

std::shared_ptr<const assets::TensorSource> make_higgs_codec_tensor_source(
    std::shared_ptr<const assets::TensorSource> source) {
    return std::make_shared<PrefixedTensorSource>(std::move(source), std::string(kHiggsCodecWeightPrefix));
}

}  // namespace engine::models::higgs_tts
