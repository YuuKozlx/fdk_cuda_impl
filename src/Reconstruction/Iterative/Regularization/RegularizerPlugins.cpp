#include "HuberTvLaunch.cuh"
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "Reconstruction/Iterative/Common/YkReductionWorkspace.hpp"
#include "Reconstruction/Iterative/Common/kernels/YkIterLaunch.cuh"
#include "Reconstruction/Iterative/Common/kernels/YkTvRegularizationLaunch.cuh"
#include "RegularizationValueLaunch.cuh"
#include "RegularizerPlugin.hpp"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>
namespace YK::Iter {
namespace {
    bool valid(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c)
    {
        return std::isfinite(c.lower_bound) && std::isfinite(c.upper_bound) && c.lower_bound <= c.upper_bound && (!c.non_negative || c.upper_bound >= 0) && p.volume.Nx > 0 && p.volume.Ny > 0 && p.volume.Nz > 0 && p.volume.voxelX_mm > 0 && p.volume.voxelY_mm > 0 && p.volume.voxelZ_mm > 0 && std::isfinite(p.volume.voxelX_mm) && std::isfinite(p.volume.voxelY_mm) && std::isfinite(p.volume.voxelZ_mm) && std::isfinite(c.strength) && c.strength >= 0 && c.inner_iterations > 0 && std::isfinite(c.strength_reduction) && c.strength_reduction > 0 && (c.tv_dimensionality == ETvDimensionality::Slice2D || c.tv_dimensionality == ETvDimensionality::Volume3D);
    }
    double differenceNormBound(const SReconstructionParams& p, int dimensions)
    {
        const double hx = p.volume.voxelX_mm, hy = p.volume.voxelY_mm, hz = p.volume.voxelZ_mm;
        return 4 * (1 / (hx * hx) + 1 / (hy * hy) + (dimensions == 3 ? 1 / (hz * hz) : 0));
    }
    bool evaluatePenalty(const float* x, double& value, const SReconstructionParams& p, float shape, int dims, int kind,
        ReductionWorkspace& reduction, cudaStream_t stream)
    {
        if (!x)
            return false;
        regularization_value_launch(x, p.volume.Nx, p.volume.Ny, p.volume.Nz,
            p.volume.voxelX_mm, p.volume.voxelY_mm, p.volume.voxelZ_mm, shape, dims, kind, reduction.data(), stream);
        return cudaPeekAtLastError() == cudaSuccess && reduction.read(stream, value);
    }
    class SmoothedTv final : public IAlgebraicRegularizer {
        ReductionWorkspace reduction_;
        SReconstructionParams p_;
        AlgebraicRegularizationConfig c_;
        cudaStream_t stream_ = nullptr;
        float* gradient_ = nullptr;
        float* reference_ = nullptr;
        size_t n_ = 0;
        float strength_ = 0;

    public:
        ~SmoothedTv() override { release(); }
        size_t workspaceBytes(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c) const override { return c.gradient_only ? 8192 : size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz * sizeof(float) * (c.require_prox ? 2 : 1) + 8192; }
        bool prepare(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c, cudaStream_t s, int device) override
        {
            release();
            if (!valid(p, c) || !std::isfinite(c.epsilon) || c.epsilon <= 0 || !std::isfinite(c.step_size) || c.step_size < 0 || cudaSetDevice(device) != cudaSuccess)
                return false;
            if (!detail::fitsReconstructionBudget(workspaceBytes(p, c), "smoothed TV plugin"))
                return false;
            p_ = p;
            c_ = c;
            stream_ = s;
            n_ = size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
            strength_ = c.strength;
            if (!reduction_.prepare())
                return false;
            if (c.gradient_only)
                return true;
            if (cudaMalloc(&gradient_, n_ * sizeof(float)) != cudaSuccess)
                return false;
            return !c.require_prox || cudaMalloc(&reference_, n_ * sizeof(float)) == cudaSuccess;
        }
        bool apply(float* x, const AlgebraicRegularizationContext&) override
        {
            if (!x || !gradient_)
                return false;
            for (int j = 0; j < c_.inner_iterations; ++j) {
                tv_gradient_launch(x, gradient_, p_.volume.Nx, p_.volume.Ny, p_.volume.Nz,
                    p_.volume.voxelX_mm, p_.volume.voxelY_mm, p_.volume.voxelZ_mm, c_.epsilon, int(c_.tv_dimensionality), stream_);
                axpy_launch(x, gradient_, -strength_, n_, stream_);
            }
            strength_ *= c_.strength_reduction;
            return cudaPeekAtLastError() == cudaSuccess;
        }
        RegularizerCapabilities capabilities() const override { return { true, true, true }; }
        bool prox(const float* z, float* u, float gamma) override
        {
            if (!z || !u || !std::isfinite(gamma) || gamma < 0)
                return false;
            if (gamma > 0 && (!reference_ || !gradient_))
                return false;
            if (gamma > 0 && cudaMemcpyAsync(reference_, z, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            if (u != z && cudaMemcpyAsync(u, z, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            const float lo = c_.non_negative ? std::max(0.f, c_.lower_bound) : c_.lower_bound;
            clamp_launch(u, n_, lo, c_.upper_bound, stream_);
            if (gamma == 0)
                return cudaPeekAtLastError() == cudaSuccess;
            const float safe = float(1 / (1 + gamma * gradientLipschitz()));
            const float eta = c_.step_size > 0 ? std::min(c_.step_size, safe) : safe;
            for (int j = 0; j < c_.inner_iterations; ++j) {
                if (!gradient(u, gradient_))
                    return false;
                bounded_prox_update_launch(u, reference_, gradient_, n_, gamma, eta, lo, c_.upper_bound, stream_);
            }
            return cudaPeekAtLastError() == cudaSuccess;
        }
        bool gradient(const float* x, float* out) override
        {
            if (!x || !out)
                return false;
            tv_gradient_launch(x, out, p_.volume.Nx, p_.volume.Ny, p_.volume.Nz, p_.volume.voxelX_mm,
                p_.volume.voxelY_mm, p_.volume.voxelZ_mm, c_.epsilon, int(c_.tv_dimensionality), stream_);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        bool value(const float* x, double& v) override { return evaluatePenalty(x, v, p_, c_.epsilon, int(c_.tv_dimensionality), 1, reduction_, stream_); }
        double gradientLipschitz() const override { return differenceNormBound(p_, int(c_.tv_dimensionality)) / c_.epsilon; }
        void reset() override { strength_ = c_.strength; }
        void release() override
        {
            if (gradient_) {
                cudaStreamSynchronize(stream_);
                cudaFree(gradient_);
            }
            if (reference_) {
                cudaStreamSynchronize(stream_);
                cudaFree(reference_);
            }
            reference_ = nullptr;
            gradient_ = nullptr;
            reduction_.release();
        }
    };
    class HuberTvProx final : public IAlgebraicRegularizer {
        ReductionWorkspace reduction_;
        SReconstructionParams p_;
        AlgebraicRegularizationConfig c_;
        cudaStream_t stream_ = nullptr;
        float *reference_ = nullptr, *gradient_ = nullptr;
        size_t n_ = 0;
        float gamma_ = 0;
        bool prepared_ = false;

    public:
        ~HuberTvProx() override { release(); }
        size_t workspaceBytes(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c) const override
        {
            return c.gradient_only || c.strength == 0 ? 8192 : size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz * sizeof(float) * 2 + 8192;
        }
        bool prepare(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c, cudaStream_t s, int device) override
        {
            release();
            if (!valid(p, c) || !std::isfinite(c.huber_delta) || c.huber_delta <= 0 || !std::isfinite(c.step_size) || c.step_size < 0 || cudaSetDevice(device) != cudaSuccess)
                return false;
            p_ = p;
            c_ = c;
            stream_ = s;
            n_ = size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
            gamma_ = c.strength;
            const double hx = p.volume.voxelX_mm, hy = p.volume.voxelY_mm, hz = p.volume.voxelZ_mm;
            const double d = 4 * (1 / (hx * hx) + 1 / (hy * hy) + (c.tv_dimensionality == ETvDimensionality::Volume3D ? 1 / (hz * hz) : 0));
            const double bound = 1.0 + gamma_ * d / c.huber_delta;
            if (!c.require_prox && c.step_size > 0 && c.step_size > 1.0 / bound * (1.0 + 1e-6))
                return false;
            if (gamma_ > 0 && !c.gradient_only) {
                if (!detail::fitsReconstructionBudget(workspaceBytes(p, c), "Huber TV prox plugin"))
                    return false;
                if (cudaMalloc(&reference_, n_ * sizeof(float)) != cudaSuccess || cudaMalloc(&gradient_, n_ * sizeof(float)) != cudaSuccess) {
                    release();
                    return false;
                }
            }
            if (!reduction_.prepare()) {
                release();
                return false;
            }
            prepared_ = true;
            return true;
        }
        bool apply(float* u, const AlgebraicRegularizationContext&) override
        {
            if (!prox(u, u, gamma_))
                return false;
            gamma_ *= c_.strength_reduction;
            return true;
        }
        bool prox(const float* z, float* u, float gamma) override
        {
            if (!prepared_ || !z || !u || !std::isfinite(gamma) || gamma < 0)
                return false;
            if (gamma == 0) {
                if (u != z && cudaMemcpyAsync(u, z, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                    return false;
                clamp_launch(u, n_, c_.non_negative ? std::max(0.f, c_.lower_bound) : c_.lower_bound, c_.upper_bound, stream_);
                return cudaPeekAtLastError() == cudaSuccess;
            }
            if (c_.gradient_only || !reference_ || !gradient_)
                return false;
            if (cudaMemcpyAsync(reference_, z, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            if (u != z && cudaMemcpyAsync(u, z, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            const double d = differenceNormBound(p_, int(c_.tv_dimensionality));
            const float safe = float(1 / (1 + gamma * d / c_.huber_delta));
            const float eta = c_.step_size == 0 ? safe : std::min(c_.step_size, safe);
            for (int j = 0; j < c_.inner_iterations; ++j) {
                huber_tv_gradient_launch(u, gradient_, p_.volume.Nx, p_.volume.Ny, p_.volume.Nz, p_.volume.voxelX_mm,
                    p_.volume.voxelY_mm, p_.volume.voxelZ_mm, c_.huber_delta, int(c_.tv_dimensionality), stream_);
                bounded_prox_update_launch(u, reference_, gradient_, n_, gamma, eta, c_.non_negative ? std::max(0.f, c_.lower_bound) : c_.lower_bound, c_.upper_bound, stream_);
                if (cudaPeekAtLastError() != cudaSuccess)
                    return false;
            }
            return true;
        }
        RegularizerCapabilities capabilities() const override { return { true, true, true }; }
        bool gradient(const float* x, float* out) override
        {
            if (!prepared_ || !x || !out)
                return false;
            huber_tv_gradient_launch(x, out, p_.volume.Nx, p_.volume.Ny, p_.volume.Nz, p_.volume.voxelX_mm,
                p_.volume.voxelY_mm, p_.volume.voxelZ_mm, c_.huber_delta, int(c_.tv_dimensionality), stream_);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        bool value(const float* x, double& v) override { return prepared_ && evaluatePenalty(x, v, p_, c_.huber_delta, int(c_.tv_dimensionality), 0, reduction_, stream_); }
        double gradientLipschitz() const override { return differenceNormBound(p_, int(c_.tv_dimensionality)) / c_.huber_delta; }
        void reset() override { gamma_ = c_.strength; }
        void release() override
        {
            if (reference_ || gradient_)
                cudaStreamSynchronize(stream_);
            if (reference_)
                cudaFree(reference_);
            if (gradient_)
                cudaFree(gradient_);
            reference_ = gradient_ = nullptr;
            prepared_ = false;
            reduction_.release();
        }
    };
    class SimplePenalty final : public IAlgebraicRegularizer {
        int kind_;
        SReconstructionParams p_ {};
        AlgebraicRegularizationConfig c_ {};
        cudaStream_t stream_ = nullptr;
        ReductionWorkspace reduction_;

    public:
        explicit SimplePenalty(int kind)
            : kind_(kind)
        {
        }
        bool prepare(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c, cudaStream_t stream, int device) override
        {
            release();
            if (!valid(p, c) || cudaSetDevice(device) != cudaSuccess)
                return false;
            p_ = p;
            c_ = c;
            stream_ = stream;
            return reduction_.prepare();
        }
        RegularizerCapabilities capabilities() const override { return { true, kind_ == 2, kind_ == 2 }; }
        bool gradient(const float* x, float* out) override
        {
            if (kind_ != 2 || !x || !out)
                return false;
            return cudaMemcpyAsync(out, x, size_t(p_.volume.Nx) * p_.volume.Ny * p_.volume.Nz * sizeof(float), cudaMemcpyDeviceToDevice, stream_) == cudaSuccess;
        }
        bool value(const float* x, double& v) override { return evaluatePenalty(x, v, p_, 1, int(c_.tv_dimensionality), kind_, reduction_, stream_); }
        double gradientLipschitz() const override { return kind_ == 2 ? 1 : 0; }
        bool prox(const float* z, float* out, float gamma) override
        {
            if (kind_ != 2 || !z || !out || !std::isfinite(gamma) || gamma < 0)
                return false;
            const size_t n = size_t(p_.volume.Nx) * p_.volume.Ny * p_.volume.Nz;
            if (z != out && cudaMemcpyAsync(out, z, n * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            scale_launch(out, 1 / (1 + gamma), n, stream_);
            clamp_launch(out, n, c_.non_negative ? std::max(0.f, c_.lower_bound) : c_.lower_bound, c_.upper_bound, stream_);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        bool apply(float* x, const AlgebraicRegularizationContext&) override { return prox(x, x, c_.strength); }
        size_t workspaceBytes(const SReconstructionParams&, const AlgebraicRegularizationConfig&) const override { return 8192; }
        void reset() override { }
        void release() override
        {
            if (stream_)
                cudaStreamSynchronize(stream_);
            reduction_.release();
            stream_ = nullptr;
        }
    };

    class StandardTv final : public IAlgebraicRegularizer {
        SReconstructionParams p_ {};
        AlgebraicRegularizationConfig c_ {};
        cudaStream_t stream_ = nullptr;
        size_t n_ = 0;
        float *reference_ = nullptr, *bar_ = nullptr, *dual_ = nullptr;
        ReductionWorkspace reduction_;

    public:
        ~StandardTv() override { release(); }
        RegularizerCapabilities capabilities() const override { return { true, false, true }; }
        size_t workspaceBytes(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c) const override
        {
            return 8192 + (c.gradient_only || c.strength == 0 ? 0 : size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz * sizeof(float) * (2 + int(c.tv_dimensionality)));
        }
        bool prepare(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c, cudaStream_t stream, int device) override
        {
            release();
            if (!valid(p, c) || !std::isfinite(c.step_size) || c.step_size < 0 || cudaSetDevice(device) != cudaSuccess)
                return false;
            p_ = p;
            c_ = c;
            stream_ = stream;
            n_ = size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
            if (c.step_size > 0 && c.step_size * c.step_size * differenceNormBound(p, int(c.tv_dimensionality)) >= 1)
                return false;
            if (!detail::fitsReconstructionBudget(workspaceBytes(p, c), "standard TV prox") || !reduction_.prepare())
                return false;
            if (c.gradient_only || c.strength == 0)
                return true;
            if (cudaMalloc(&reference_, n_ * 4) != cudaSuccess || cudaMalloc(&bar_, n_ * 4) != cudaSuccess || cudaMalloc(&dual_, n_ * 4 * int(c.tv_dimensionality)) != cudaSuccess) {
                release();
                return false;
            }
            return true;
        }
        bool value(const float* x, double& v) override { return evaluatePenalty(x, v, p_, 1, int(c_.tv_dimensionality), 3, reduction_, stream_); }
        bool prox(const float* z, float* u, float gamma) override
        {
            if (!z || !u || !std::isfinite(gamma) || gamma < 0 || !n_)
                return false;
            if (gamma > 0 && (!reference_ || !bar_ || !dual_))
                return false;
            if (gamma > 0 && cudaMemcpyAsync(reference_, z, n_ * 4, cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            if (u != z && cudaMemcpyAsync(u, z, n_ * 4, cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            const float lo = c_.non_negative ? std::max(0.f, c_.lower_bound) : c_.lower_bound;
            clamp_launch(u, n_, lo, c_.upper_bound, stream_);
            if (gamma == 0)
                return cudaPeekAtLastError() == cudaSuccess;
            if (cudaMemcpyAsync(bar_, u, n_ * 4, cudaMemcpyDeviceToDevice, stream_) != cudaSuccess || cudaMemsetAsync(dual_, 0, n_ * 4 * int(c_.tv_dimensionality), stream_) != cudaSuccess)
                return false;
            const float step = c_.step_size > 0 ? c_.step_size : float(.9 / std::sqrt(differenceNormBound(p_, int(c_.tv_dimensionality))));
            const auto& v = p_.volume;
            for (int j = 0; j < c_.inner_iterations; ++j) {
                tv_dual_launch(bar_, dual_, v.Nx, v.Ny, v.Nz, v.voxelX_mm, v.voxelY_mm, v.voxelZ_mm, int(c_.tv_dimensionality), step, gamma, stream_);
                tv_primal_launch(u, bar_, reference_, dual_, v.Nx, v.Ny, v.Nz, v.voxelX_mm, v.voxelY_mm, v.voxelZ_mm, int(c_.tv_dimensionality), step, lo, c_.upper_bound, stream_);
            }
            return cudaPeekAtLastError() == cudaSuccess;
        }
        bool apply(float* x, const AlgebraicRegularizationContext&) override { return prox(x, x, c_.strength); }
        void reset() override { }
        void release() override
        {
            if (reference_ || bar_ || dual_)
                cudaStreamSynchronize(stream_);
            if (reference_)
                cudaFree(reference_);
            if (bar_)
                cudaFree(bar_);
            if (dual_)
                cudaFree(dual_);
            reference_ = bar_ = dual_ = nullptr;
            n_ = 0;
            reduction_.release();
        }
    };
    struct Registry {
        std::mutex mutex;
        std::unordered_map<std::string, RegularizerFactory> factories;
        Registry()
        {
            factories.emplace("smoothed_tv", [] { return std::make_unique<SmoothedTv>(); });
            factories.emplace("huber_tv", [] { return std::make_unique<HuberTvProx>(); });
            factories.emplace("huber_tv_prox", [] { return std::make_unique<HuberTvProx>(); });
            factories.emplace("tikhonov", [] { return std::make_unique<SimplePenalty>(2); });
            factories.emplace("tv", [] { return std::make_unique<StandardTv>(); });
        }
    };
    Registry& registry()
    {
        static Registry r;
        return r;
    }
}
bool registerRegularizerPlugin(const std::string& id, RegularizerFactory factory)
{
    if (id.empty() || id == "none" || !factory)
        return false;
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.factories.emplace(id, std::move(factory)).second;
}
std::unique_ptr<IAlgebraicRegularizer> createRegularizerPlugin(const std::string& id)
{
    RegularizerFactory factory;
    {
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        const auto i = r.factories.find(id);
        if (i == r.factories.end())
            return {};
        factory = i->second;
    }
    try {
        return factory();
    } catch (...) {
        return {};
    }
}
std::string regularizerPluginId(const AlgebraicRegularizationConfig& c)
{
    if (!c.plugin_id.empty())
        return c.plugin_id == "none" ? "" : c.plugin_id;
    return c.type == EAlgebraicRegularizer::SmoothedTv ? "smoothed_tv" : c.type == EAlgebraicRegularizer::HuberTvProx ? "huber_tv_prox"
                                                                                                                      : "";
}
}
