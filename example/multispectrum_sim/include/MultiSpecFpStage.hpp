#pragma once
#include "SpectralProjectionPipeline.hpp"
namespace yk::spectral {
// Real source stage: owns backend lifetime, prepares physical inputs, generates
// validated batches, and seeds host signal/reference state for downstream stages.
// The backend retains its fused CUDA kernels and device buffers.
class MultiSpecFpStage final : public IProjectionSource {
public:
    explicit MultiSpecFpStage(std::unique_ptr<IProjectionBackend>);
    ~MultiSpecFpStage() override;
    const pipeline::PluginInfo& info() const noexcept override;
    pipeline::SourceContract<ProjectionSignalDomain> contract() const override;
    void prepare(const SpectralSourceContext&) override;
    int batchCapacity() const override;
    void generate(int first, int count, std::vector<ProjectionFrame>&) override;
    void ensureLocation(std::vector<ProjectionFrame>&, ProjectionMemoryLocation) override;
    void finalizeBatch(std::vector<ProjectionFrame>&) override;
    void synchronize() override;
    ProjectionViewGeometry geometryForDisplay(int view) const override;
    void sampleScatterNoise(ProjectionFrame&) override;
    void reset() noexcept override;

private:
    std::unique_ptr<IProjectionBackend> backend_;
    int views_ = 0, width_ = 0, height_ = 0, capacity_ = 0;
    bool prepared_ = false, resources_started_ = false;
    void requirePrepared() const;
};
}
