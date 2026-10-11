#pragma once

#include "config/SimConfig.hpp"
#include "SpectralModel.hpp"
#include "../../../visualization/LiveView.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace yk::spectral {
using ProjectionViewGeometry = yk::visualization::FrameGeometry;

// Architecture:
//   backend.prepare/project -> ordered signal stages -> domain-specific sink
// The backend owns CUDA resources; stages own temporal/random state; the sink
// owns normalization and file I/O. Pipeline resets all three after each run.
// Extensions are registered in C++, not reordered through TOML. Stages in the
// same physical phase execute in registration order after the built-in stage.
// Current execution uses host frames at the backend boundary. Device stages
// need a device-frame contract before they can be registered here.
// A photon-counting implementation supplies a count backend, count-domain
// stages (threshold response, dead time/pile-up), and a count output sink.

// Energy-integrating signals and photon counts have different physical units.
enum class ProjectionSignalDomain { EnergyKeV, PhotonCounts };
enum class ProjectionMemoryLocation { Host, Device };
enum class ProjectionStagePhase {
    Scatter, QuantumNoise, OpticalCrosstalk, Afterglow, ElectronicCrosstalk, Readout
};

struct PhotonCountFrame {
    // Optional [energy][row][column] incident/transmitted expectations.
    std::vector<float> energies_keV;
    std::vector<float> expected_counts;
    // Measured detector channels, distinct from the simulation energy grid.
    // N thresholds define N-1 bins; counts are [bin][row][column].
    std::vector<float> bin_edges_keV;
    std::vector<float> channel_counts;
    std::vector<float> air_channel_counts;
    std::vector<float> reference_channel_counts;
};

struct ProjectionFrame {
    int view = 0, width = 0, height = 0;
    ProjectionSignalDomain domain = ProjectionSignalDomain::EnergyKeV;
    ProjectionMemoryLocation location = ProjectionMemoryLocation::Host;
    std::vector<float> expected_primary_energy, sampled_primary_energy;
    std::vector<float> air_energy, signal_energy, reference_energy;
    std::vector<unsigned char> object_mask;
    PhotonCountFrame photon_counts;
    void resize(int columns, int rows);
};

// A backend owns device resources and may fuse ray integration, spectral
// transport and detector response into one kernel.
class IProjectionBackend {
public:
    virtual ~IProjectionBackend() = default;
    virtual ProjectionSignalDomain outputDomain() const = 0;
    virtual void prepare(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&) = 0;
    virtual void project(int view, ProjectionFrame&) = 0;
    virtual ProjectionViewGeometry geometryForDisplay(int view) const;
    virtual void sampleScatterNoise(ProjectionFrame&) = 0;
    virtual void reset() noexcept = 0;
};

struct ProjectionStageContext {
    const SimulationConfig& config;
    IProjectionBackend& backend;
};

class IProjectionStage {
public:
    virtual ~IProjectionStage() = default;
    virtual const char* name() const = 0;
    virtual ProjectionStagePhase phase() const = 0;
    virtual ProjectionSignalDomain domain() const = 0;
    virtual ProjectionMemoryLocation location() const {
        return ProjectionMemoryLocation::Host;
    }
    virtual void prepare(const SimulationConfig&) {}
    virtual void process(ProjectionFrame&, const ProjectionStageContext&) = 0;
    // Air is a deterministic reference. Stateful modules keep separate air state.
    virtual void processAir(ProjectionFrame&, const ProjectionStageContext&) {}
    virtual void reset() noexcept {}
};

class IProjectionOutputSink {
public:
    virtual ~IProjectionOutputSink() = default;
    virtual ProjectionSignalDomain domain() const = 0;
    virtual void prepare(const SimulationConfig&, const std::filesystem::path&) = 0;
    virtual void write(const ProjectionFrame&) = 0;
    virtual void finish() = 0;
    virtual void reset() noexcept = 0;
};

class SpectralProjectionPipeline {
public:
    explicit SpectralProjectionPipeline(std::unique_ptr<IProjectionBackend> backend);
    ~SpectralProjectionPipeline();
    void addStage(std::unique_ptr<IProjectionStage> stage);
    void setOutputSink(std::unique_ptr<IProjectionOutputSink> sink);
    void setVisualization(yk::visualization::LiveView*);
    bool run(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&, const std::filesystem::path& output,
        std::string& error);
private:
    std::unique_ptr<IProjectionBackend> backend_;
    std::vector<std::unique_ptr<IProjectionStage>> extensions_;
    std::unique_ptr<IProjectionOutputSink> sink_;
    bool running_ = false;
    yk::visualization::LiveView* visualization_ = nullptr;
};

} // namespace yk::spectral
