#pragma once
#include "SpectralProjectionPipeline.hpp"

namespace yk::spectral::primary_detail {
class CudaProjectionBackend final : public IProjectionBackend {
public:
    CudaProjectionBackend();
    ~CudaProjectionBackend() override;
    ProjectionSignalDomain outputDomain() const override {
        return ProjectionSignalDomain::EnergyKeV;
    }
    void prepare(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&) override;
    void project(int view, ProjectionFrame&) override;
    void sampleScatterNoise(ProjectionFrame&) override;
    void reset() noexcept override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yk::spectral::primary_detail
