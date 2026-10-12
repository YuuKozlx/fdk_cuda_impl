#include "recon/preprocess/PreprocessPipeline.hpp"
#include "recon/preprocess/DeviceMemoryPool.hpp"
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace yk::spectral;
using namespace yk::spectral::preprocess;
namespace {
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
class Probe final : public IStage {
    Location location_;
    bool add_;

  public:
    Probe(Location l, bool add = false) : location_(l), add_(add) {}
    const char *name() const override { return "State probe"; }
    Domain inputDomain() const override { return Domain::LogAttenuation; }
    Domain outputDomain() const override { return Domain::LogAttenuation; }
    Location inputLocation() const override { return location_; }
    void prepare(const Context &) override {}
    void process(Frame &f, const Context &) override {
        require(f.location == location_, "wrong authoritative location at stage entry");
        if (add_)
            for (auto &v : *f.host_data)
                v += 2;
    }
    void reset() noexcept override {}
};
void write(const std::filesystem::path &p, const std::vector<float> &values) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
}
std::vector<float> read(const std::filesystem::path &p) {
    std::ifstream in(p, std::ios::binary);
    std::vector<float> v(12);
    in.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
    require(bool(in), "state test output missing");
    return v;
}
} // namespace
void testPreprocessState(const std::filesystem::path &root) {
    // A second reservation must reuse every block without another cudaMalloc.
    trimDevicePool();
    initializeDevicePool({4096, 8192});
    auto first = devicePoolStatistics();
    initializeDevicePool({4096, 8192});
    auto second = devicePoolStatistics();
    require(second.allocations == first.allocations && second.reuses >= first.reuses + 2,
            "pool initialization reallocates reserved blocks");
    void *lease1 = nullptr, *lease2 = nullptr;
    require(poolAllocate(&lease1, 4096) == cudaSuccess &&
                poolAllocate(&lease2, 4096) == cudaSuccess,
            "cannot borrow pooled buffers");
    require(lease1 != lease2, "pool reused an active lease");
    require(poolRelease(lease1) == cudaSuccess && poolRelease(lease2) == cudaSuccess,
            "cannot return pooled buffers");
    require(poolRelease(lease2) != cudaSuccess, "pool accepted a double return");
    size_t free_bytes = 0, total_bytes = 0;
    require(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess,
            "cannot inspect memory budget");
    void *oversized = nullptr;
    require(poolAllocate(&oversized, total_bytes) == cudaErrorMemoryAllocation && !oversized,
            "pool consumed VTK reserved memory");
    SimulationConfig config;
    auto &g = config.geometry.parameters;
    g.views = 2;
    g.detector_u = 3;
    g.detector_v = 2;
    auto &pp = config.reconstruction.preprocessing;
    pp.enabled = true;
    auto &bh = pp.beam_hardening;
    bh.enabled = true;
    bh.coefficients = {0, 1};
    bh.clamp_min = -20;
    bh.clamp_max = 20;
    const auto input = root / "state-input.raw", output = root / "state-output.raw";
    const std::vector<float> values = {-1, 0, .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1};
    write(input, values);
    std::string diagnostic;
    Pipeline resident;
    resident.append(std::make_unique<Probe>(Location::Device));
    resident.append(std::make_unique<Probe>(Location::Device));
    require(resident.run(config, input, output, diagnostic), diagnostic.c_str());
    require(resident.statistics().image_uploads == 2 && resident.statistics().image_downloads == 2,
            "consecutive device stages transferred full frames");
    require(read(output) == values, "resident pipeline changed identity values");
    const auto pool_after_first=devicePoolStatistics();
    require(pool_after_first.active_bytes==0,"completed pipeline retained active leases");
    require(resident.run(config,input,output,diagnostic),diagnostic.c_str());
    const auto pool_after_second=devicePoolStatistics();
    require(pool_after_second.allocations==pool_after_first.allocations,"same-size repeated pipeline allocates new device buffers");

    Pipeline mixed;
    mixed.append(std::make_unique<Probe>(Location::Host, true));
    mixed.append(std::make_unique<Probe>(Location::Device));
    mixed.append(std::make_unique<Probe>(Location::Device));
    require(mixed.run(config, input, output, diagnostic), diagnostic.c_str());
    require(mixed.statistics().image_uploads == 4 && mixed.statistics().image_downloads == 4,
            "host/device boundary did not transfer exactly once");
    auto result = read(output);
    for (size_t i = 0; i < values.size(); ++i)
        require(result[i] == values[i] + 2, "stale device copy survived CPU stage modification");
    // Compare the CUDA polynomial against double Horner, including clamp endpoints.
    for (int degree = 0; degree <= 16; ++degree) {
        bh.coefficients.resize(degree + 1);
        for (int i = 0; i <= degree; ++i)
            bh.coefficients[i] = (i % 2 ? -1 : 1) * .2 / (i + 1);
        bh.clamp_min = 0;
        bh.clamp_max = .3;
        Pipeline p;
        require(p.run(config, input, output, diagnostic), diagnostic.c_str());
        result = read(output);
        for (size_t i = 0; i < values.size(); ++i) {
            double expected = 0;
            for (auto it = bh.coefficients.rbegin(); it != bh.coefficients.rend(); ++it)
                expected = expected * values[i] + *it;
            expected = std::min(.3, std::max(0., expected));
            require(std::abs(result[i] - expected) < 1e-7,
                    "GPU BHC disagrees with double Horner reference");
        }
    }
    bh.coefficients = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    write(input, std::vector<float>(12, 2));
    Pipeline overflow;
    require(!overflow.run(config, input, output, diagnostic),
            "overflow was hidden by BHC clamping");
    bh.coefficients = {0, 1};
    write(input, values);
    require(overflow.run(config, input, output, diagnostic),
            "pipeline failed to reset after invalid frame");
}
