#include "cuda/DeviceMemoryPool.hpp"
#include "recon/postprocess/stages/correction/EmpiricalBoneCorrection.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
namespace yk::spectral::postprocess {
namespace {
    void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    struct Buffer {
        float* p = nullptr;
        ~Buffer() { cuda_resources::poolRelease(p); }
        void upload(const std::vector<float>& v)
        {
            check(cuda_resources::poolAllocate(&p, v.size() * 4));
            check(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        }
    };
    size_t validate(const EbhcBasis& b, const Grid& g)
    {
        size_t n = 1;
        for (int k = 0; k < 3; ++k) {
            if (g.dimensions[k] < 2 || !std::isfinite(g.spacing_mm[k]) || g.spacing_mm[k] <= 0)
                throw std::invalid_argument("Invalid EBHC grid");
            n *= g.dimensions[k];
        }
        for (auto& v : b) {
            if (!v || v->size() != n)
                throw std::invalid_argument("Invalid EBHC basis size");
            for (float x : *v)
                if (!std::isfinite(x))
                    throw std::invalid_argument("Nonfinite EBHC basis");
        }
        return n;
    }
    __device__ double value(size_t i, const float* f, const float* a, const float* b, const float* c, EbhcCoefficients t) { return double(f[i]) + t.linear * a[i] + t.cross * b[i] + t.square * c[i]; }
    __global__ void tv(const float* f, const float* a, const float* b, const float* c, size_t n, int nx, int ny, double sx, double sy, double sz, EbhcCoefficients t, double* blocks)
    {
        __shared__ double sums[256];
        double sum = 0;
        const size_t plane = size_t(nx) * ny;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
            const double v = value(i, f, a, b, c, t);
            const double x = i % nx ? (v - value(i - 1, f, a, b, c, t)) / sx : 0;
            const double y = (i / nx) % ny ? (v - value(i - nx, f, a, b, c, t)) / sy : 0;
            const double z = i / plane ? (v - value(i - plane, f, a, b, c, t)) / sz : 0;
            sum += sqrt(x * x + y * y + z * z);
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int k = 128; k; k /= 2) {
            if (threadIdx.x < k)
                sums[threadIdx.x] += sums[threadIdx.x + k];
            __syncthreads();
        }
        if (!threadIdx.x)
            blocks[blockIdx.x] = sums[0];
    }
    __global__ void combine(float* f, const float* a, const float* b, const float* c, size_t n, EbhcCoefficients t)
    {
        size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n)
            f[i] = float(value(i, f, a, b, c, t));
    }
    class Stage final : public IStage {
        EbhcCoefficients coefficients_;
        EbhcBasis basis_;
        Grid grid_;
        Buffer buffers_[3];
        size_t n_ = 0;

    public:
        Stage(EbhcCoefficients c, EbhcBasis b, Grid g)
            : coefficients_(c)
            , basis_(std::move(b))
            , grid_(g)
        {
            validate(basis_, g);
            if (!std::isfinite(c.linear) || !std::isfinite(c.cross) || !std::isfinite(c.square))
                throw std::invalid_argument("Invalid EBHC coefficients");
        }
        const pipeline::PluginInfo& info() const noexcept override { return ebhcVolumeInfo(); }
        pipeline::Contract<Domain> contract() const override { return { Domain::AttenuationMmInv, Domain::AttenuationMmInv, Location::Device, Location::Device }; }
        void prepare(const Context& c) override
        {
            reset();
            if (c.grid.dimensions != grid_.dimensions || c.grid.spacing_mm != grid_.spacing_mm || c.grid.origin_mm != grid_.origin_mm || c.grid.direction != grid_.direction)
                throw std::invalid_argument("EBHC context grid mismatch");
            n_ = validate(basis_, grid_);
            try {
                for (int k = 0; k < 3; ++k)
                    buffers_[k].upload(*basis_[k]);
            } catch (...) {
                reset();
                throw;
            }
        }
        void process(Volume& v, const Context& c) override
        {
            if (!n_ || !v.device_data || v.location != Location::Device || v.domain != Domain::AttenuationMmInv || v.elements != n_ || v.grid.dimensions != grid_.dimensions || v.grid.spacing_mm != grid_.spacing_mm || v.grid.origin_mm != grid_.origin_mm || v.grid.direction != grid_.direction)
                throw std::invalid_argument("Invalid EBHC volume");
            combine<<<unsigned((n_ + 255) / 256), 256, 0, static_cast<cudaStream_t>(c.stream)>>>(v.device_data, buffers_[0].p, buffers_[1].p, buffers_[2].p, n_, coefficients_);
            check(cudaGetLastError());
        }
        void reset() noexcept override
        {
            for (auto& b : buffers_) {
                cuda_resources::poolRelease(b.p);
                b.p = nullptr;
            }
            n_ = 0;
        }
    };
}
EbhcTvResult fitEbhcFullVolumeTv(const std::vector<float>& f, const EbhcBasis& b, const Grid& g, std::function<void(int, double)> progress)
{
    const size_t n = validate(b, g);
    if (f.size() != n)
        throw std::invalid_argument("EBHC baseline size mismatch");
    for (float v : f)
        if (!std::isfinite(v))
            throw std::invalid_argument("Nonfinite baseline");
    Buffer df, db[3];
    df.upload(f);
    for (int k = 0; k < 3; ++k)
        db[k].upload(*b[k]);
    struct Reduction {
        double* p = nullptr;
        ~Reduction() { cuda_resources::poolRelease(p); }
    } reduction;
    const int count = 1024;
    check(cuda_resources::poolAllocate(&reduction.p, count * sizeof(double)));
    std::vector<double> host(count);
    EbhcTvResult result;
    using Point = std::array<double, 3>;
    struct Vertex {
        Point x;
        double y;
    };
    auto evaluate = [&](Point x) {EbhcCoefficients c{x[0],x[1],x[2]};tv<<<count,256>>>(df.p,db[0].p,db[1].p,db[2].p,n,g.dimensions[0],g.dimensions[1],g.spacing_mm[0],g.spacing_mm[1],g.spacing_mm[2],c,reduction.p);check(cudaGetLastError());check(cudaMemcpy(host.data(),reduction.p,count*sizeof(double),cudaMemcpyDeviceToHost));double sum=0;for(auto v:host)sum+=v;sum/=n;if(!std::isfinite(sum))throw std::runtime_error("Nonfinite TV");++result.evaluations;result.trace.push_back({x[0],x[1],x[2],sum});if(progress && result.evaluations%10==0)progress(result.evaluations,sum);return Vertex{x,sum}; };
    std::array<Vertex, 4> vertices { evaluate({ 0, 0, 0 }), evaluate({ .05, 0, 0 }), evaluate({ 0, .01, 0 }), evaluate({ 0, 0, .01 }) };
    result.initial_tv = vertices[0].y;
    auto move = [](Point center, Point worst, double factor) {Point p;for(int k=0;k<3;++k)p[k]=center[k]+factor*(center[k]-worst[k]);return p; };
    while (result.evaluations < 600) {
        std::sort(vertices.begin(), vertices.end(), [](auto a, auto b) { return a.y < b.y; });
        double width = 0;
        for (int j = 1; j < 4; ++j)
            for (int k = 0; k < 3; ++k)
                width = std::max(width, std::abs(vertices[j].x[k] - vertices[0].x[k]));
        if (width < 1e-5 && vertices[3].y - vertices[0].y < 1e-7 * std::max(result.initial_tv, 1e-20)) {
            result.converged = true;
            break;
        }
        Point center {};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                center[k] += vertices[j].x[k] / 3;
        auto reflected = evaluate(move(center, vertices[3].x, 1));
        if (reflected.y < vertices[0].y) {
            auto expanded = evaluate(move(center, vertices[3].x, 2));
            vertices[3] = expanded.y < reflected.y ? expanded : reflected;
        } else if (reflected.y < vertices[2].y)
            vertices[3] = reflected;
        else {
            bool outside = reflected.y < vertices[3].y;
            auto contracted = evaluate(move(center, vertices[3].x, outside ? .5 : -.5));
            if (contracted.y < (outside ? reflected.y : vertices[3].y))
                vertices[3] = contracted;
            else
                for (int j = 1; j < 4; ++j) {
                    Point p;
                    for (int k = 0; k < 3; ++k)
                        p[k] = (vertices[0].x[k] + vertices[j].x[k]) * .5;
                    vertices[j] = evaluate(p);
                }
        }
    }
    std::sort(vertices.begin(), vertices.end(), [](auto a, auto b) { return a.y < b.y; });
    result.coefficients = { vertices[0].x[0], vertices[0].x[1], vertices[0].x[2] };
    result.final_tv = vertices[0].y;
    return result;
}
std::unique_ptr<IStage> makeEbhcVolumeStage(EbhcCoefficients c, EbhcBasis b, Grid g) { return std::make_unique<Stage>(c, std::move(b), g); }
}
