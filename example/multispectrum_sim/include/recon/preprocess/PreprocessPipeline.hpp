#pragma once
#include "config/SimConfig.hpp"
#include "TaskProgress.hpp"
#include <memory>
#include <vector>
namespace yk::spectral::preprocess {
class CudaPixelOps;
enum class Domain { LogAttenuation, EnergySignal };
enum class Location { Host, Device };
// location identifies the authoritative copy. Other storage is stale after a stage.
// Storage is borrowed, valid only during process(); never retain frame pointers.
struct Frame {
    int view = 0, width = 0, height = 0;
    size_t pixels = 0;
    Domain domain = Domain::LogAttenuation;
    Location location = Location::Host;
    std::vector<float> *host_data = nullptr;
    float *device_data = nullptr;
    const float *device_air = nullptr;
};
struct Context {
    const SimulationConfig &config;
    CudaPixelOps &cuda;
    const TaskProgressCallback &progress;
};
class IStage {
  public:
    virtual ~IStage() = default;
    virtual const char *name() const = 0;
    virtual Domain inputDomain() const = 0;
    virtual Domain outputDomain() const = 0;
    virtual Location inputLocation() const { return Location::Device; }
    virtual Location outputLocation() const { return inputLocation(); }
    virtual void prepare(const Context &) = 0;
    virtual void process(Frame &, const Context &) = 0;
    virtual void reset() noexcept = 0;
};
struct Statistics {
    size_t image_uploads = 0, image_downloads = 0, air_uploads = 0;
    double total_ms = 0, init_ms = 0, fit_ms = 0, read_ms = 0, transfer_ms = 0, write_ms = 0;
    std::vector<std::pair<std::string, double>> stages_ms;
};
// Not thread-safe; one instance per reconstruction worker. append() installs stages
// after the built-in correction chain. Its final output must be log attenuation.
class Pipeline {
  public:
    void append(std::unique_ptr<IStage>);
    bool run(const SimulationConfig &, const std::filesystem::path &input,
             const std::filesystem::path &output, std::string &diagnostic,
             TaskProgressCallback progress = {});
    const Statistics &statistics() const { return statistics_; }

  private:
    std::vector<std::unique_ptr<IStage>> extensions_;
    Statistics statistics_;
    bool running_ = false;
};
} // namespace yk::spectral::preprocess
