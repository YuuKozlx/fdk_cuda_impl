#pragma once
#include "SpectralProjectionPipeline.hpp"

namespace yk::spectral::primary_detail {
ProjectionSourceRegistry makeCudaProjectionSourceRegistry();
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
    bool supportsDeviceFrames()const override{return true;}
    void ensureLocation(std::vector<ProjectionFrame>&,ProjectionMemoryLocation) override;
    void finalizeBatch(std::vector<ProjectionFrame>&) override;
    void synchronize() override;
    int batchCapacity() const override;
    void projectBatch(int first_view,int count,std::vector<ProjectionFrame>&) override;
    ProjectionViewGeometry geometryForDisplay(int view) const override;
    void sampleScatterNoise(ProjectionFrame&) override;
    void reset() noexcept override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yk::spectral::primary_detail
