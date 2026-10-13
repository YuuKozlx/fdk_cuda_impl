#pragma once
#include "TaskProgress.hpp"
#include "pipeline/Stage.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace yk::spectral::postprocess {
enum class Domain { AttenuationMmInv,
    HounsfieldUnits };
using Location = pipeline::Location;
struct Grid {
    std::array<int, 3> dimensions {};
    std::array<double, 3> spacing_mm { 1, 1, 1 }, origin_mm {};
    // Voxel-index axes in scanner world coordinates; not assumed to be patient axes.
    std::array<double, 9> direction { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
};
// Borrowed storage. location is authoritative; the other pointer may be stale.
// Plugins must not retain pointers or silently reinterpret attenuation as HU.
struct Volume {
    Grid grid;
    size_t elements = 0;
    float* host_data = nullptr;
    float* device_data = nullptr;
    Domain domain = Domain::AttenuationMmInv;
    Location location = Location::Host;
};
struct Context;
class IDataTransfer {
public:
    virtual ~IDataTransfer() = default;
    // Use caller-owned buffers/pool/stream. Update location only after success.
    virtual void ensure(Volume&, Location, const Context&) = 0;
};
struct Context {
    std::filesystem::path volume_file, slice_prefix;
    // Optional read-only inputs for methods that need projections/configuration.
    std::filesystem::path original_projection_file, reconstruction_projection_file, source_config_file;
    Grid grid; // Filled from the current volume before stage preparation.
    TaskProgressCallback progress;
    IDataTransfer* transfer = nullptr;
    void* stream = nullptr;
};
using IStage = pipeline::IStage<Volume, Context, Domain>;
struct ReadOnlyVolume {
    Grid grid;
    size_t elements;
    const float* host_data;
    const float* device_data;
    Domain domain;
    Location location;
    explicit ReadOnlyVolume(const Volume& v)
        : grid(v.grid)
        , elements(v.elements)
        , host_data(v.host_data)
        , device_data(v.device_data)
        , domain(v.domain)
        , location(v.location)
    {
    }
};
// Exporters read a completed volume; they cannot change its values or metadata.
using IExporter = pipeline::IOutput<ReadOnlyVolume, Context, Domain>;
struct StageTiming {
    std::string name;
    double milliseconds = 0;
};
class Pipeline {
public:
    void append(std::unique_ptr<IStage>);
    void appendExporter(std::unique_ptr<IExporter>);
    // One worker per instance. Fails explicitly when domain/input/transfer is missing.
    // Resets stages on success and failure; numerical processors may modify in place.
    void run(Volume&, const Context&);
    const std::vector<StageTiming>& timings() const { return timings_; }

private:
    std::vector<std::unique_ptr<IStage>> stages_;
    std::vector<std::unique_ptr<IExporter>> exporters_;
    std::vector<StageTiming> timings_;
    bool running_ = false;
};
using StageRegistry = pipeline::FactoryRegistry<IStage>;
using ExportRegistry = pipeline::FactoryRegistry<IExporter>;
std::unique_ptr<IStage> makeHuConversionStage(double water_mu_mm_inv);
std::unique_ptr<IExporter> makeRawFloatExporter();
std::unique_ptr<IExporter> makeOrthogonalBmpExporter();
} // namespace yk::spectral::postprocess
