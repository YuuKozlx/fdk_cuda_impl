#include "cuda/DeviceMemoryPool.hpp"
#include "recon/preprocess/correction/EmpiricalBhcStage.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
namespace yk::spectral::preprocess {
void EmpiricalBhcParameters::validate() const
{
    if (!std::isfinite(square) || !std::isfinite(cross) || !std::isfinite(path_scale_mm) || path_scale_mm <= 0)
        throw std::invalid_argument("Invalid EBHC coefficients/path scale");
}
namespace {
    void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    __global__ void correct(float* p, const float* paths, size_t n, double a, double b, double scale)
    {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        const double v = p[i], q = paths[i] / scale;
        // Invalid paths remain observable to the pipeline validator. No clipping:
        // clipping would break equivalence with the reconstructed basis combination.
        p[i] = isfinite(v) && isfinite(q) && q >= 0 ? float(v + a * q * q + b * v * q) : nanf("");
    }
    class Stage final : public IStage {
        EmpiricalBhcParameters parameters_;
        std::shared_ptr<IBoneThicknessProvider> provider_;
        float* paths_ = nullptr;
        int width_ = 0, height_ = 0, views_ = 0;

    public:
        Stage(EmpiricalBhcParameters p, std::shared_ptr<IBoneThicknessProvider> provider)
            : parameters_(p)
            , provider_(std::move(provider))
        {
            p.validate();
            if (!provider_)
                throw std::invalid_argument("Missing EBHC path provider");
        }
        ~Stage() override { reset(); }
        const pipeline::PluginInfo& info() const noexcept override { return empiricalBhcInfo(); }
        pipeline::Contract<Domain> contract() const override { return { Domain::LogAttenuation, Domain::LogAttenuation, Location::Device, Location::Device }; }
        void validateContext(const Context& c) const override
        {
            const auto& g = c.config.geometry.parameters;
            if (g.detector_u <= 0 || g.detector_v <= 0 || g.views <= 0)
                throw std::invalid_argument("Invalid EBHC geometry");
        }
        void prepare(const Context& c) override
        {
            reset();
            validateContext(c);
            const auto& g = c.config.geometry.parameters;
            width_ = g.detector_u;
            height_ = g.detector_v;
            views_ = g.views;
            try {
                check(cuda_resources::poolAllocate(&paths_, size_t(width_) * height_ * sizeof(float)));
                provider_->prepare(c);
            } catch (...) {
                reset();
                throw;
            }
        }
        void process(Frame& f, const Context&) override
        {
            if (!paths_ || f.width != width_ || f.height != height_ || f.pixels != size_t(width_) * height_ || f.view < 0 || f.view >= views_ || !f.device_data || f.location != Location::Device || f.domain != Domain::LogAttenuation)
                throw std::invalid_argument("Invalid EBHC frame");
            provider_->writeDevice(f.view, width_, height_, paths_);
            correct<<<unsigned((f.pixels + 255) / 256), 256>>>(f.device_data, paths_, f.pixels, parameters_.square, parameters_.cross, parameters_.path_scale_mm);
            check(cudaGetLastError());
        }
        void reset() noexcept override
        {
            provider_->reset();
            cuda_resources::poolRelease(paths_);
            paths_ = nullptr;
            width_ = height_ = views_ = 0;
        }
    };
}
std::unique_ptr<IStage> makeEmpiricalBhcStage(EmpiricalBhcParameters p, std::shared_ptr<IBoneThicknessProvider> provider) { return std::make_unique<Stage>(p, std::move(provider)); }
}
