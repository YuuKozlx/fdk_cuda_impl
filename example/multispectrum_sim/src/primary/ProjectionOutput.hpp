#pragma once
#include "SpectralProjectionPipeline.hpp"
#include <fstream>
namespace yk::spectral::primary_detail {
// Writes energy-integrating signals. A photon-counting sink can independently
// preserve channel counts, bin thresholds, and per-channel air calibration.
class EnergyProjectionOutput final : public IProjectionOutputSink {
public:
    ProjectionSignalDomain domain() const override { return ProjectionSignalDomain::EnergyKeV; }
    void prepare(const SimulationConfig&, const std::filesystem::path&) override;
    void write(const ProjectionFrame&) override;
    void finish() override;
    void reset() noexcept override;
private:
    void writeBuffer(std::ofstream&, const std::vector<float>&);
    ProjectionConfig config_;
    std::filesystem::path output_;
    int width_ = 0, height_ = 0, views_ = 0, written_ = 0;
    std::ofstream attenuation_, energy_, air_;
    std::vector<float> attenuation_buffer_;
};
} // namespace yk::spectral::primary_detail
