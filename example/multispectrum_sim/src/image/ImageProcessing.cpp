#include "image/ImageProcessing.hpp"
#include "ImageBackend.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "image/ImagePluginInfo.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
namespace yk::spectral::image {
namespace {
    void require(bool ok, const char* s)
    {
        if (!ok)
            throw std::invalid_argument(s);
    }
    void checked(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    size_t checkedCount(const Grid& g, int batches)
    {
        require(batches > 0, "image batch count must be positive");
        size_t n = size_t(batches);
        for (int d : g.dimensions) {
            require(d > 0 && n <= std::numeric_limits<size_t>::max() / sizeof(float) / size_t(d), "invalid image dimensions");
            n *= size_t(d);
        }
        return n;
    }
    void gridValid(const Grid& g)
    {
        checkedCount(g, 1);
        for (int i = 0; i < 3; ++i)
            require(std::isfinite(g.spacing_mm[i]) && g.spacing_mm[i] > 0 && std::isfinite(g.origin_mm[i]), "invalid image grid");
        for (double d : g.direction)
            require(std::isfinite(d), "invalid image direction");
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double dot = 0;
                for (int k = 0; k < 3; ++k)
                    dot += g.direction[3 * k + i] * g.direction[3 * k + j];
                require(std::abs(dot - (i == j ? 1. : 0.)) < 1.e-6, "image direction must be orthonormal");
            }
    }
    detail::Shape shape(const Grid& g) { return { g.dimensions[0], g.dimensions[1], g.dimensions[2] }; }
    template <class F, class Cuda>
    void dispatch(Location l, float* out, size_t n, F f, Cuda cuda)
    {
        if (l == Location::Device)
            cuda(out, n, f);
        else
            for (size_t i = 0; i < n; ++i)
                out[i] = f(i);
    }
    void point(Location l, float* out, size_t n, detail::Pointwise f) { dispatch(l, out, n, f, detail::cudaPointwise); }
    void box(const float* in, float* out, detail::Shape s, const int* radii, const double* sigmas, bool gaussian, Workspace& w)
    {
        const auto loc = w.location();
        const float* current = in;
        for (int axis = 0; axis < 3; ++axis) {
            if (!radii[axis] || s.dim(axis) == 1)
                continue;
            float* next = w.scratch(current == w.scratch(1, s.count()) ? 2 : 1, s.count());
            dispatch(loc, next, s.count(), detail::AxisFilter { current, s, axis, radii[axis], sigmas[axis], gaussian }, detail::cudaAxis);
            current = next;
        }
        if (current != out)
            copy(out, loc, current, loc, s.count());
    }
    const pipeline::PluginInfo& descriptor(const Options& o, Domain d, Location l)
    {
        return pluginDescriptions()[(d == Domain::Volume ? 10 : 0) + (l == Location::Device ? 5 : 0) + int(o.operation)];
    }
    class Stage final : public IStage {
        Options options_;
        Domain domain_;
        Location location_;

    public:
        Stage(Options o, Domain d, Location l)
            : options_(o)
            , domain_(d)
            , location_(l)
        {
            validate(o);
        }
        const pipeline::PluginInfo& info() const noexcept override { return descriptor(options_, domain_, location_); }
        pipeline::Contract<Domain> contract() const override { return { domain_, domain_, location_, location_ }; }
        void validateContext(const Context& c) const override { require(c.workspace.location() == location_, "image stage/workspace location mismatch"); }
        void process(Image& data, const Context& c) override { image::process(data, options_, c.workspace); }
    };
}
size_t Image::elements() const { return checkedCount(grid, batches); }
void validate(const Image& i)
{
    gridValid(i.grid);
    i.elements();
    require(i.data, "missing image data");
    require(i.domain == Domain::Projection || i.domain == Domain::Volume, "invalid image domain");
    require(i.location == Location::Host || i.location == Location::Device, "invalid image location");
    require(i.domain != Domain::Projection || i.grid.dimensions[2] == 1, "projection angles must use batches, not spatial Z");
}
void validate(const Options& o)
{
    require(int(o.operation) >= 0 && int(o.operation) <= int(Operation::Guided), "invalid image operation");
    require(o.neighborhood == Neighborhood::Slice2D || o.neighborhood == Neighborhood::Spatial3D, "invalid neighborhood");
    require(o.interpolation == Interpolation::Nearest || o.interpolation == Interpolation::Linear, "invalid interpolation");
    for (int a = 0; a < 3; ++a) {
        require(o.radius[a] >= 0 && o.radius[a] <= 4096, "image radius outside [0,4096]");
        require(std::isfinite(o.sigma_mm[a]) && o.sigma_mm[a] > 0, "invalid spatial sigma");
        require(o.output_dimensions[a] > 0, "invalid resampling dimensions");
    }
    require(std::isfinite(o.range_sigma) && o.range_sigma > 0, "invalid range sigma");
    require(std::isfinite(o.epsilon) && o.epsilon > 0, "invalid guided epsilon");
}
Grid resampledGrid(const Grid& input, const std::array<int, 3>& dims)
{
    gridValid(input);
    Grid out = input;
    out.dimensions = dims;
    checkedCount(out, 1);
    // Preserve voxel-cell support, physical center and orientation, not only centers.
    for (int a = 0; a < 3; ++a) {
        out.spacing_mm[a] = input.spacing_mm[a] * input.dimensions[a] / dims[a];
        for (int k = 0; k < 3; ++k)
            out.origin_mm[k] += input.direction[3 * k + a] * (out.spacing_mm[a] - input.spacing_mm[a]) * 0.5;
    }
    gridValid(out);
    return out;
}
std::shared_ptr<float> allocate(Location l, size_t n)
{
    require(n > 0 && n <= std::numeric_limits<size_t>::max() / sizeof(float), "invalid image allocation");
    if (l == Location::Host)
        return { new float[n], std::default_delete<float[]>() };
    require(l == Location::Device, "invalid allocation location");
    int device = 0;
    checked(cudaGetDevice(&device));
    float* p = nullptr;
    checked(cuda_resources::poolAllocate(&p, n * sizeof(float)));
    return { p, [device](float* q) {int old=0;cudaGetDevice(&old);cudaSetDevice(device);cuda_resources::poolRelease(q);cudaSetDevice(old); } };
}
void copy(float* dst, Location dl, const float* src, Location sl, size_t n)
{
    if (!n || dst == src)
        return;
    if (dl == Location::Host && sl == Location::Host)
        std::memcpy(dst, src, n * sizeof(float));
    else
        checked(cudaMemcpyAsync(dst, src, n * sizeof(float), dl == Location::Device ? (sl == Location::Device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : cudaMemcpyDeviceToHost, 0));
    // Host storage may be reused/freed immediately by the caller.
    if (dl != sl)
        synchronize(Location::Device);
}
void synchronize(Location l)
{
    if (l == Location::Device)
        checked(cudaStreamSynchronize(0));
}
void ensureLocation(Image& data, Location l)
{
    validate(data);
    if (l == data.location)
        return;
    auto out = allocate(l, data.elements());
    copy(out.get(), l, data.data, data.location, data.elements());
    data.owner = std::move(out);
    data.data = data.owner.get();
    data.location = l;
}
struct Workspace::Impl {
    Location location;
    size_t budget;
    int device = -1;
    std::array<std::shared_ptr<float>, 8> buffers;
    std::array<size_t, 8> capacities {};
};
Workspace::Workspace(Location l, size_t b)
    : impl_(std::make_unique<Impl>())
{
    require(b >= 8 * sizeof(float), "image scratch budget too small");
    require(l == Location::Host || l == Location::Device, "invalid workspace location");
    impl_->location = l;
    impl_->budget = b;
    if (l == Location::Device)
        checked(cudaGetDevice(&impl_->device));
}
Workspace::~Workspace() = default;
Location Workspace::location() const { return impl_->location; }
size_t Workspace::scratchBudget() const { return impl_->budget; }
float* Workspace::scratch(int slot, size_t n)
{
    require(slot >= 0 && slot < 8 && n > 0, "invalid scratch request");
    if (location() == Location::Device) {
        int device = 0;
        checked(cudaGetDevice(&device));
        require(device == impl_->device, "workspace used on another CUDA device");
    }
    if (impl_->capacities[slot] < n) {
        size_t others = 0;
        for (int a = 0; a < 8; ++a)
            if (a != slot)
                others += impl_->capacities[a];
        require(n <= impl_->budget / sizeof(float) && others <= impl_->budget / sizeof(float) - n, "image scratch budget exceeded; trim workspace before larger geometry");
        impl_->buffers[slot].reset();
        impl_->capacities[slot] = 0;
        impl_->buffers[slot] = allocate(location(), n);
        impl_->capacities[slot] = n;
    }
    return impl_->buffers[slot].get();
}
void Workspace::trim()
{
    const auto status = location() == Location::Device ? cudaStreamSynchronize(0) : cudaSuccess;
    for (auto& b : impl_->buffers)
        b.reset();
    impl_->capacities.fill(0);
    checked(status);
}
void process(Image& image, const Options& options, Workspace& workspace)
{
    validate(image);
    validate(options);
    require(image.location == workspace.location(), "image/workspace location mismatch");
    const auto loc = image.location;
    const auto s = shape(image.grid);
    if (options.operation == Operation::Resample) {
        require(image.domain != Domain::Projection || options.output_dimensions[2] == 1, "cannot resample projection angle axis");
        require(options.neighborhood != Neighborhood::Slice2D || options.output_dimensions[2] == s.z, "slice-wise resampling must preserve slice count");
        const Grid finalGrid = resampledGrid(image.grid, options.output_dimensions);
        // Each axis has its own output lease; obsolete intermediates return to pool.
        Image current = image;
        for (int axis = 0; axis < 3; ++axis) {
            if (current.grid.dimensions[axis] == options.output_dimensions[axis])
                continue;
            Image next = current;
            next.grid.dimensions[axis] = options.output_dimensions[axis];
            next.owner = allocate(loc, next.elements());
            next.data = next.owner.get();
            const auto src = shape(current.grid), dst = shape(next.grid);
            for (int b = 0; b < image.batches; ++b)
                dispatch(loc, next.data + size_t(b) * dst.count(), dst.count(), detail::ResizeAxis { current.data + size_t(b) * src.count(), src, dst, axis, options.interpolation == Interpolation::Nearest, options.antialias }, detail::cudaResize);
            current = std::move(next);
        }
        synchronize(loc);
        current.grid = finalGrid;
        image = std::move(current);
        return;
    }
    int radius[3];
    double sigma[3];
    for (int a = 0; a < 3; ++a) {
        radius[a] = image.grid.dimensions[a] == 1 ? 0 : options.radius[a];
        sigma[a] = options.sigma_mm[a] / image.grid.spacing_mm[a];
        require(std::isfinite(sigma[a]) && sigma[a] > 0, "spatial sigma/grid ratio invalid");
    }
    if (options.neighborhood == Neighborhood::Slice2D || image.domain == Domain::Projection)
        radius[2] = 0;
    const int halo = radius[2] * (options.operation == Operation::Guided ? 2 : 1);
    const size_t plane = size_t(s.x) * s.y;
    const size_t maxDepth = workspace.scratchBudget() / sizeof(float) / 8 / plane;
    require(maxDepth >= size_t(std::min(s.z, 2 * halo + 1)), "scratch budget cannot hold one slab and its filter halo");
    const int step = maxDepth >= size_t(s.z) ? s.z : int(maxDepth) - 2 * halo;
    auto output = allocate(loc, image.elements());
    for (int batch = 0; batch < image.batches; ++batch)
        for (int z = 0; z < s.z; z += step) {
            const int end = std::min(s.z, z + step), lo = std::max(0, z - halo), hi = std::min(s.z, end + halo);
            detail::Shape tile { s.x, s.y, hi - lo };
            const size_t n = tile.count();
            float* input = workspace.scratch(0, n);
            float* result = workspace.scratch(3, n);
            const float* reference = image.data + size_t(batch) * s.count();
            copy(input, loc, reference + size_t(lo) * plane, loc, n);
            if (options.operation == Operation::Mean || options.operation == Operation::Gaussian) {
                box(input, result, tile, radius, sigma, options.operation == Operation::Gaussian, workspace);
            } else if (options.operation == Operation::Bilateral) {
                dispatch(loc, result, n, detail::Bilateral { input, tile, radius[0], radius[1], radius[2], sigma[0], sigma[1], sigma[2], options.range_sigma }, detail::cudaBilateral);
            } else {
                // Shift moments to reduce loss of precision on large constant HU backgrounds.
                point(loc, input, n, { input, reference, nullptr, 4, 0 });
                float* mean = result;
                float* a = workspace.scratch(4, n);
                float* b = workspace.scratch(5, n);
                float* ma = workspace.scratch(6, n);
                float* mb = workspace.scratch(7, n);
                box(input, mean, tile, radius, sigma, false, workspace);
                point(loc, a, n, { input, nullptr, nullptr, 0, 0 });
                box(a, ma, tile, radius, sigma, false, workspace);
                point(loc, a, n, { ma, mean, nullptr, 1, options.epsilon });
                point(loc, b, n, { a, mean, nullptr, 2, 0 });
                box(a, ma, tile, radius, sigma, false, workspace);
                box(b, mb, tile, radius, sigma, false, workspace);
                point(loc, a, n, { ma, input, mb, 3, 0 });
                point(loc, a, n, { a, reference, nullptr, 5, 0 });
                result = a;
            }
            copy(output.get() + size_t(batch) * s.count() + size_t(z) * plane, loc, result + size_t(z - lo) * plane, loc, size_t(end - z) * plane);
        }
    synchronize(loc);
    image.owner = std::move(output);
    image.data = image.owner.get();
}
std::unique_ptr<IStage> makeStage(Options o, Domain d, Location l)
{
    require(d == Domain::Projection || d == Domain::Volume, "invalid stage domain");
    require(l == Location::Host || l == Location::Device, "invalid stage location");
    return std::make_unique<Stage>(o, d, l);
}
Registry makeRegistry(Domain d, Location l)
{
    require(d == Domain::Projection || d == Domain::Volume, "invalid registry domain");
    require(l == Location::Host || l == Location::Device, "invalid registry backend");
    Registry r;
    for (int k = 0; k < 5; ++k) {
        Options o;
        o.operation = Operation(k);
        if (d == Domain::Projection)
            o.neighborhood = Neighborhood::Slice2D;
        r.add(descriptor(o, d, l), [=] { return makeStage(o, d, l); });
    }
    return r;
}
void Pipeline::append(std::unique_ptr<IStage> stage)
{
    require(!running_ && bool(stage), "invalid image stage");
    stages_.push_back(std::move(stage));
}
void Pipeline::run(Image& image, Workspace& w)
{
    validate(image);
    pipeline::RunScope scope(running_);
    Context c { w };
    auto domain = image.domain;
    for (auto& stage : stages_) {
        domain = pipeline::advanceDomain(domain, stage->contract(), stage->info().name.c_str());
        stage->validateContext(c);
    }
    timings_.clear();
    transfers_ = 0;
    auto cleanup = pipeline::onExit([&] { pipeline::resetReverse(stages_); });
    for (auto& stage : stages_) {
        stage->prepare(c);
        const auto t = pipeline::execute(stage->info().name.c_str(), stage->contract(), &image, 1, [&](Location l) { ensureLocation(image, l); }, [&](Image& v) { stage->process(v, c); }, [&] { synchronize(w.location()); });
        validate(image);
        transfers_ += t.transfers;
        timings_.push_back({ stage->info().id, t.process_ms, t.transfer_ms });
    }
}
} // namespace
