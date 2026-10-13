#pragma once
#include "ImageGrid.hpp"
#include "pipeline/Stage.hpp"
#include <array>
#include <cstddef>
#include <memory>
#include <vector>
namespace yk::spectral::image {
using Location = pipeline::Location;
enum class Domain { Projection,
    Volume };
enum class Operation { Resample,
    Mean,
    Gaussian,
    Bilateral,
    Guided };
enum class Neighborhood { Slice2D,
    Spatial3D };
enum class Interpolation { Nearest,
    Linear };
// x contiguous, then y, z, batch. Projection angles are batches, never z neighbors.
struct Image {
    Grid grid;
    int batches = 1;
    Domain domain = Domain::Volume;
    Location location = Location::Host;
    float* data = nullptr;
    std::shared_ptr<float> owner; // optional for borrowed input; owns replaced output
    size_t elements() const;
};
struct Options {
    Operation operation = Operation::Gaussian;
    Neighborhood neighborhood = Neighborhood::Spatial3D;
    std::array<int, 3> radius { 1, 1, 1 };
    std::array<double, 3> sigma_mm { 1, 1, 1 };
    double range_sigma = 0.01; // same units as input values
    double epsilon = 1.e-4; // squared input units; self-guided filter
    std::array<int, 3> output_dimensions { 1, 1, 1 };
    Interpolation interpolation = Interpolation::Linear;
    bool antialias = true; // area averaging on shrinking axes for linear resampling
};
// Default CUDA stream only, matching DeviceMemoryPool event ownership.
// One workspace per worker, not thread safe. Scratch is reused across stages/runs.
class Workspace {
public:
    explicit Workspace(Location, size_t scratch_budget_bytes = 256ull * 1024 * 1024);
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Location location() const;
    size_t scratchBudget() const;
    float* scratch(int slot, size_t elements);
    void trim();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::shared_ptr<float> allocate(Location, size_t elements);
void copy(float* dst, Location, const float* src, Location, size_t elements);
void synchronize(Location);
void validate(const Image&);
void validate(const Options&);
Grid resampledGrid(const Grid&, const std::array<int, 3>&);
void ensureLocation(Image&, Location);
// Replaces storage, preserves domain/location. Output survives workspace/stage reset.
void process(Image&, const Options&, Workspace&);
struct Context {
    Workspace& workspace;
};
using IStage = pipeline::IStage<Image, Context, Domain>;
std::unique_ptr<IStage> makeStage(Options, Domain, Location = Location::Device);
using Registry = pipeline::FactoryRegistry<IStage>;
Registry makeRegistry(Domain, Location = Location::Device);
struct Timing {
    std::string id;
    double process_ms = 0, transfer_ms = 0;
};
class Pipeline {
public:
    void append(std::unique_ptr<IStage>);
    void run(Image&, Workspace&);
    const std::vector<Timing>& timings() const { return timings_; }
    size_t transfers() const { return transfers_; }

private:
    bool running_ = false;
    size_t transfers_ = 0;
    std::vector<std::unique_ptr<IStage>> stages_;
    std::vector<Timing> timings_;
};
} // namespace yk::spectral::image
