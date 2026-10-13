#pragma once
#include "TaskProgress.hpp"
#include "pipeline/Stage.hpp"
#include "image/ImageGrid.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace yk::spectral::image { class Workspace; }
namespace yk::spectral::postprocess {
enum class Domain { AttenuationMmInv,
    HounsfieldUnits,
    Labels };
using Location = pipeline::Location;
using Grid = image::Grid;
struct LabelClass { int value=0; std::string name; };
struct LabelMapping { std::vector<double> thresholds_hu; std::vector<LabelClass> classes; };
// Borrowed storage. location is authoritative; the other pointer may be stale.
// Plugins must not retain pointers or silently reinterpret attenuation as HU.
struct Volume {
    Grid grid;
    size_t elements = 0;
    float* host_data = nullptr;
    float* device_data = nullptr;
    Domain domain = Domain::AttenuationMmInv;
    Location location = Location::Host;
    // Replacing stages transfer ownership here; output survives stage reset.
    std::shared_ptr<float> host_owner, device_owner;
    std::shared_ptr<const LabelMapping> label_mapping;
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
    image::Workspace* image_workspace = nullptr;
};
using IStage = pipeline::IStage<Volume, Context, Domain>;
struct ReadOnlyVolume {
    Grid grid;
    size_t elements;
    const float* host_data;
    const float* device_data;
    Domain domain;
    Location location;
    std::shared_ptr<const LabelMapping> label_mapping;
    explicit ReadOnlyVolume(const Volume& v)
        : grid(v.grid)
        , elements(v.elements)
        , host_data(v.host_data)
        , device_data(v.device_data)
        , domain(v.domain)
        , location(v.location)
        , label_mapping(v.label_mapping)
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
