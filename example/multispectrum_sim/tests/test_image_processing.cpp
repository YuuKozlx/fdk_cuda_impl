#include "cuda/DeviceMemoryPool.hpp"
#include "image/ImagePluginInfo.hpp"
#include "image/ImageStages.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace yk::spectral;
namespace im = yk::spectral::image;
void require(bool p, const char* s)
{
    if (!p)
        throw std::runtime_error(s);
}
template <class F>
void rejects(F f)
{
    bool failed = false;
    try {
        f();
    } catch (const std::exception&) {
        failed = true;
    }
    require(failed, "invalid image request accepted");
}
im::Image source(int x, int y, int z, int batches = 1)
{
    im::Image v;
    v.grid.dimensions = { x, y, z };
    v.batches = batches;
    v.owner = im::allocate(im::Location::Host, v.elements());
    v.data = v.owner.get();
    return v;
}
void compare(const im::Image& a, const im::Image& b, double tol)
{
    require(a.elements() == b.elements(), "shape mismatch");
    for (size_t i = 0; i < a.elements(); ++i)
        if (std::abs(double(a.data[i]) - b.data[i]) > tol || !std::isfinite(b.data[i]))
            throw std::runtime_error("image numerical mismatch at " + std::to_string(i) + " " + std::to_string(a.data[i]) + " vs " + std::to_string(b.data[i]));
}
int clamp(int x, int n) { return std::max(0, std::min(n - 1, x)); }
im::Image naiveBox(const im::Image& in, int r, bool slices)
{
    auto out = source(in.grid.dimensions[0], in.grid.dimensions[1], in.grid.dimensions[2], in.batches);
    const int nx = out.grid.dimensions[0], ny = out.grid.dimensions[1], nz = out.grid.dimensions[2];
    for (int b = 0; b < in.batches; ++b)
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    double sum = 0;
                    int count = 0;
                    for (int dz = slices ? 0 : -r; dz <= (slices ? 0 : r); ++dz)
                        for (int dy = -r; dy <= r; ++dy)
                            for (int dx = -r; dx <= r; ++dx) {
                                sum += in.data[((size_t(b) * nz + clamp(z + dz, nz)) * ny + clamp(y + dy, ny)) * nx + clamp(x + dx, nx)];
                                ++count;
                            }
                    out.data[((size_t(b) * nz + z) * ny + y) * nx + x] = float(sum / count);
                }
    return out;
}
im::Image naiveGuided(const im::Image& input, double epsilon)
{
    auto mean = naiveBox(input, 1, false), square = source(9, 7, 11, 2);
    for (size_t i = 0; i < input.elements(); ++i)
        square.data[i] = input.data[i] * input.data[i];
    auto var = naiveBox(square, 1, false), a = source(9, 7, 11, 2), b = source(9, 7, 11, 2);
    for (size_t i = 0; i < input.elements(); ++i) {
        double v = std::max(0., double(var.data[i]) - double(mean.data[i]) * mean.data[i]);
        a.data[i] = float(v / (v + epsilon));
        b.data[i] = (1 - a.data[i]) * mean.data[i];
    }
    auto ma = naiveBox(a, 1, false), mb = naiveBox(b, 1, false);
    for (size_t i = 0; i < input.elements(); ++i)
        ma.data[i] = ma.data[i] * input.data[i] + mb.data[i];
    return ma;
}
class Probe final : public postprocess::IStage {
public:
    const pipeline::PluginInfo& info() const noexcept override
    {
        static const pipeline::PluginInfo p { "test.grid", "Grid probe", "", "1", pipeline::PluginKind::PostprocessStage };
        return p;
    }
    pipeline::Contract<postprocess::Domain> contract() const override { return { postprocess::Domain::AttenuationMmInv, postprocess::Domain::AttenuationMmInv, im::Location::Device, im::Location::Device }; }
    void validateContext(const postprocess::Context& c) const override { require(c.grid.dimensions == std::array<int, 3> { 5, 4, 6 }, "stage preparation received obsolete geometry"); }
    void process(postprocess::Volume&, const postprocess::Context&) override { }
};
int main(int argc, char** argv)
{
    try {
        int devices = 0;
        require(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0, "CUDA device required for image validation");
        im::Workspace cpu(im::Location::Host), gpu(im::Location::Device);
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            auto big = source(256, 256, 64);
            for (size_t i = 0; i < big.elements(); ++i)
                big.data[i] = float(.02 + .003 * std::sin(double(i) * .02));
            auto device = big;
            im::ensureLocation(device, im::Location::Device);
            for (int op = 1; op <= 4; ++op) {
                im::Options o;
                o.operation = im::Operation(op);
                o.neighborhood = im::Neighborhood::Spatial3D;
                auto warm = device;
                im::process(warm, o, gpu);
                auto host = big, actual = device;
                auto tick = std::chrono::steady_clock::now();
                im::process(host, o, cpu);
                double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick).count();
                tick = std::chrono::steady_clock::now();
                im::process(actual, o, gpu);
                double gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick).count();
                im::ensureLocation(actual, im::Location::Host);
                compare(host, actual, 1.e-6);
                std::cout << im::makeStage(o, im::Domain::Volume)->info().id << " shape=256x256x64 radius=1 cpu_ms=" << cpuMs << " cuda_ms=" << gpuMs << " speedup=" << cpuMs / gpuMs << '\n';
            }
            return 0;
        }
        auto input = source(9, 7, 11, 2);
        input.grid.spacing_mm = { .4, .7, 1.2 };
        for (size_t i = 0; i < input.elements(); ++i)
            input.data[i] = float(.2 * std::sin(double(i) * .19) + (i % 9 >= 4 ? .7 : 0));
        double max_error = 0;
        for (int op = 0; op < 5; ++op)
            for (auto neighborhood : { im::Neighborhood::Slice2D, im::Neighborhood::Spatial3D }) {
                im::Options o;
                o.operation = im::Operation(op);
                o.neighborhood = neighborhood;
                o.output_dimensions = { 5, 11, neighborhood == im::Neighborhood::Slice2D ? 11 : 6 };
                o.range_sigma = .2;
                o.epsilon = .03;
                auto host = input, device = input;
                im::process(host, o, cpu);
                im::ensureLocation(device, im::Location::Device);
                im::process(device, o, gpu);
                im::ensureLocation(device, im::Location::Host);
                compare(host, device, 3.e-6);
                for (size_t i = 0; i < host.elements(); ++i)
                    max_error = std::max(max_error, std::abs(double(host.data[i]) - device.data[i]));
                // Small scratch forces overlapping slabs; includes final partial slab and batch boundaries.
                if (op) {
                    im::Workspace tiled(im::Location::Device, size_t(9) * 7 * 5 * 8 * sizeof(float));
                    auto small = input;
                    im::ensureLocation(small, im::Location::Device);
                    im::process(small, o, tiled);
                    im::ensureLocation(small, im::Location::Host);
                    compare(host, small, 3.e-6);
                }
                if (op == 1)
                    compare(host, naiveBox(input, 1, neighborhood == im::Neighborhood::Slice2D), 2.e-6);
                if (op == 4 && neighborhood == im::Neighborhood::Spatial3D)
                    compare(host, naiveGuided(input, o.epsilon), 3.e-6);
            }
        // Slice-wise mode cannot mix adjacent layers, even for a 3D volume.
        for (int op = 1; op < 5; ++op) {
            auto layers = source(7, 5, 6);
            for (size_t i = 0; i < layers.elements(); ++i)
                layers.data[i] = float(i / 35) * 10;
            im::Options o;
            o.operation = im::Operation(op);
            o.neighborhood = im::Neighborhood::Slice2D;
            auto expected = layers;
            im::ensureLocation(layers, im::Location::Device);
            im::process(layers, o, gpu);
            im::ensureLocation(layers, im::Location::Host);
            compare(expected, layers, 1.e-5);
        }
        auto projection = source(7, 5, 1, 3);
        projection.domain = im::Domain::Projection;
        for (size_t i = 0; i < projection.elements(); ++i)
            projection.data[i] = float(i / 35) * 100;
        auto expectedProjection = projection;
        im::Pipeline chain;
        im::Options o;
        o.operation = im::Operation::Gaussian;
        chain.append(im::makeStage(o, im::Domain::Projection));
        o.operation = im::Operation::Guided;
        chain.append(im::makeStage(o, im::Domain::Projection));
        chain.run(projection, gpu);
        require(chain.transfers() == 1, "consecutive image stages transferred twice");
        im::ensureLocation(projection, im::Location::Host);
        compare(projection, expectedProjection, 1.e-5);
        // Area downsampling suppresses an alternating signal; nearest keeps exact labels.
        auto stripes = source(8, 1, 1);
        for (size_t i = 0; i < 8; ++i)
            stripes.data[i] = float(i % 2);
        o.operation = im::Operation::Resample;
        o.output_dimensions = { 4, 1, 1 };
        auto averaged = stripes;
        im::process(averaged, o, cpu);
        for (size_t i = 0; i < 4; ++i)
            require(std::abs(averaged.data[i] - .5f) < 1.e-6, "downsampling aliases alternating input");
        o.interpolation = im::Interpolation::Nearest;
        auto nearest = stripes;
        im::process(nearest, o, cpu);
        for (size_t i = 0; i < 4; ++i)
            require(nearest.data[i] == 0 || nearest.data[i] == 1, "nearest changed a label");
        auto rotated = input.grid;
        rotated.direction = { 0, -1, 0, 1, 0, 0, 0, 0, 1 };
        auto resized = im::resampledGrid(rotated, { 5, 4, 6 });
        for (int k = 0; k < 3; ++k) {
            double oldCenter = rotated.origin_mm[k], newCenter = resized.origin_mm[k];
            for (int a = 0; a < 3; ++a) {
                oldCenter += rotated.direction[3 * k + a] * rotated.spacing_mm[a] * (rotated.dimensions[a] - 1) * .5;
                newCenter += resized.direction[3 * k + a] * resized.spacing_mm[a] * (resized.dimensions[a] - 1) * .5;
            }
            require(std::abs(oldCenter - newCenter) < 1.e-12, "resampling moved physical center");
        }
        // Real volume pipeline starts on Device, keeps two filters resident, then transfers once.
        auto volumeInput = input;
        volumeInput.batches = 1;
        im::ensureLocation(volumeInput, im::Location::Device);
        postprocess::Volume volume;
        volume.grid = volumeInput.grid;
        volume.elements = volumeInput.elements();
        volume.device_data = volumeInput.data;
        volume.device_owner = volumeInput.owner;
        volume.location = im::Location::Device;
        im::VolumeTransfer transfer;
        postprocess::Context context;
        context.transfer = &transfer;
        postprocess::Pipeline post;
        o = {};
        o.operation = im::Operation::Resample;
        o.output_dimensions = { 5, 4, 6 };
        post.append(im::makeVolumeStage(o));
        post.append(std::make_unique<Probe>());
        o.operation = im::Operation::Gaussian;
        post.append(im::makeVolumeStage(o));
        post.run(volume, context);
        require(volume.location == im::Location::Device && volume.elements == 120, "volume output ownership/shape lost after stage reset");
        require(transfer.uploads == 0 && transfer.downloads == 0, "resident volume filter crossed PCIe");
        transfer.ensure(volume, im::Location::Host, context);
        require(transfer.downloads == 1, "volume export transfer count incorrect");
        for (size_t i = 0; i < volume.elements; ++i)
            require(std::isfinite(volume.host_data[i]), "volume output invalid after stage reset");
        auto bad = input;
        bad.domain = im::Domain::Projection;
        rejects([&] { im::validate(bad); });
        o.radius[0] = -1;
        rejects([&] { im::process(bad, o, cpu); });
        o = {};
        o.operation = im::Operation::Resample;
        rejects([&] { im::makeProjectionStage(o); });
        im::Workspace tooSmall(im::Location::Device, 64);
        o = {};
        rejects([&] {auto v=input;im::ensureLocation(v,im::Location::Device);im::process(v,o,tooSmall); });
        auto registry = im::makeRegistry(im::Domain::Volume);
        require(registry.list().size() == 5, "image registry missing operations");
        for (const auto& p : registry.list()) {
            pipeline::validatePluginInfo(p);
            require(registry.create(p.id)->info() == p, "image metadata mismatch");
        }
        // Repeated same-sized resident filters reuse pool buffers (after a warm-up).
        o = {};
        auto repeated = volumeInput;
        im::process(repeated, o, gpu);
        im::process(repeated, o, gpu);
        auto stats = cuda_resources::devicePoolStatistics();
        im::process(repeated, o, gpu);
        auto stats2 = cuda_resources::devicePoolStatistics();
        require(stats.allocations == stats2.allocations, "repeated image processing allocated new device buffers");
        std::cout << "CPU/CUDA max error=" << max_error << "; 2D/3D, slice isolation, independent batches, halos, geometry, ownership and residency passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
