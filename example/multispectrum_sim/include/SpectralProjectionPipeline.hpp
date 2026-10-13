#pragma once

#include "../../../visualization/LiveView.hpp"
#include "SpectralModel.hpp"
#include "config/SimConfig.hpp"
#include "pipeline/Stage.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace yk::spectral {
using ProjectionViewGeometry = yk::visualization::FrameGeometry;

// Backend supplies borrowed Host/Device batch frames. Stages declare location;
// pipeline inserts whole-batch transfers only at actual CPU/Device boundaries.
// Built-ins run on CUDA for device backends, CPU for host backends. Extensions
// register with addStage(), ordered by physical phase and registration order.
// Stages execute over the batch in view order; temporal state persists across
// batches and is reset after each run. Device pointers belong to the backend,
// use the current device/default stream, and must not be retained across batches.
// A photon-counting implementation supplies a count backend, count-domain
// stages (threshold response, dead time/pile-up), and a count output sink.

// Energy-integrating signals and photon counts have different physical units.
enum class ProjectionSignalDomain { EnergyKeV,
    PhotonCounts };
using ProjectionMemoryLocation = pipeline::Location;
enum class ProjectionStagePhase {
    Scatter,
    QuantumNoise,
    OpticalCrosstalk,
    Afterglow,
    ElectronicCrosstalk,
    Readout
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

struct DeviceProjectionFrame {
    float *expected = nullptr, *sampled = nullptr, *air = nullptr, *signal = nullptr, *reference = nullptr, *logarithm = nullptr;
    unsigned char* mask = nullptr;
};
struct ProjectionFrame {
    int view = 0, width = 0, height = 0;
    ProjectionSignalDomain domain = ProjectionSignalDomain::EnergyKeV;
    ProjectionMemoryLocation location = ProjectionMemoryLocation::Host;
    std::vector<float> expected_primary_energy, sampled_primary_energy;
    std::vector<float> air_energy, signal_energy, reference_energy;
    std::vector<unsigned char> object_mask;
    PhotonCountFrame photon_counts;
    DeviceProjectionFrame device;
    std::vector<float> logarithm;
    bool logarithm_valid = false;
    void resize(int columns, int rows);
};

// A backend owns device resources and may fuse ray integration, spectral
// transport and detector response into one kernel.
class IProjectionBackend {
public:
    virtual ~IProjectionBackend() = default;
    virtual ProjectionSignalDomain outputDomain() const = 0;
    virtual void prepare(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&)
        = 0;
    virtual void project(int view, ProjectionFrame&) = 0;
    virtual bool supportsDeviceFrames() const { return false; }
    virtual void ensureLocation(std::vector<ProjectionFrame>&, ProjectionMemoryLocation);
    virtual void finalizeBatch(std::vector<ProjectionFrame>&) { }
    virtual void synchronize() { }
    virtual int batchCapacity() const { return 1; }
    virtual void projectBatch(int first_view, int count, std::vector<ProjectionFrame>&);
    virtual ProjectionViewGeometry geometryForDisplay(int view) const;
    virtual void sampleScatterNoise(ProjectionFrame&) = 0;
    virtual void reset() noexcept = 0;
};

struct SpectralSourceContext {
    const SimulationConfig& config;
    const std::vector<std::uint8_t>& labels;
    const SpectralTransmissionModel& model;
};
class IProjectionSource : public pipeline::ISource<ProjectionFrame, SpectralSourceContext, ProjectionSignalDomain> {
public:
    virtual void ensureLocation(std::vector<ProjectionFrame>&, ProjectionMemoryLocation) = 0;
    virtual void finalizeBatch(std::vector<ProjectionFrame>&) = 0;
    virtual void synchronize() = 0;
    virtual ProjectionViewGeometry geometryForDisplay(int view) const = 0;
    virtual void sampleScatterNoise(ProjectionFrame&) = 0;
};
using ProjectionSourceRegistry = pipeline::FactoryRegistry<IProjectionSource>;

struct ProjectionStageContext {
    const SimulationConfig& config;
    IProjectionSource& source;
};

class IProjectionStage : public pipeline::IStage<ProjectionFrame, ProjectionStageContext, ProjectionSignalDomain> {
public:
    virtual ProjectionStagePhase phase() const = 0;
    // Separate deterministic air state remains a projection-specific operation.
    virtual void processAir(ProjectionFrame&, const ProjectionStageContext&) { }
};

struct ProjectionOutputContext {
    const SimulationConfig& config;
    const std::filesystem::path& output_file;
};
using IProjectionOutputSink = pipeline::IOutput<ProjectionFrame, ProjectionOutputContext, ProjectionSignalDomain>;

class SpectralProjectionPipeline {
public:
    explicit SpectralProjectionPipeline(std::unique_ptr<IProjectionSource> source);
    const pipeline::PluginInfo& sourceInfo() const noexcept;
    ~SpectralProjectionPipeline();
    void addStage(std::unique_ptr<IProjectionStage> stage);
    void setOutputSink(std::unique_ptr<IProjectionOutputSink> sink);
    void setVisualization(yk::visualization::LiveView*);
    bool run(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&, const std::filesystem::path& output,
        std::string& error, TaskProgressCallback progress = {});

private:
    std::unique_ptr<IProjectionSource> source_;
    std::vector<std::unique_ptr<IProjectionStage>> extensions_;
    std::unique_ptr<IProjectionOutputSink> sink_;
    bool running_ = false;
    yk::visualization::LiveView* visualization_ = nullptr;
};

} // namespace yk::spectral
