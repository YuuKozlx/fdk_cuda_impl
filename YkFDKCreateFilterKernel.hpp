#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cmath>
#include <utility>

#include "YkGlobals.h"
#include "YkFFT.hpp"

namespace YK {
    // ============================================================
    // Device helpers
    // ============================================================
    static __device__ __forceinline__ float yk_sinc_pi_device(float x)
    {
        const float pi = CUDA_PI;
        float t = pi * x;
        if (fabsf(t) < 1e-8f) return 1.0f;
        return sinf(t) / t;
    }

    static __device__ __forceinline__ float yk_window_shape_device(float x, EFilterKernel kind)
    {
        // x in [0,1]
        const float pi = CUDA_PI;
        x = fminf(fmaxf(x, 0.0f), 1.0f);

        switch (kind) {
        case EFilterKernel::None:
        case EFilterKernel::RamLak:
            return 1.0f;

        case EFilterKernel::SheppLogan:
            // common: sinc(f/(2*fc)) -> with x=f/fc => sinc(x/2)
            return yk_sinc_pi_device(0.5f * x);

        case EFilterKernel::Cosine:
            return cosf(0.5f * pi * x);

        case EFilterKernel::Hann:
            return 0.5f * (1.0f + cosf(pi * x));

        case EFilterKernel::Hamming:
            return 0.54f + 0.46f * cosf(pi * x);

        case EFilterKernel::Blackman: {
            // single-sided form consistent with pi*x
            const float a0 = 0.42f;
            const float a1 = 0.5f;
            const float a2 = 0.08f;
            return a0 + a1 * cosf(pi * x) + a2 * cosf(2.0f * pi * x);
        }

        default:
            return 1.0f;
        }
    }

    // ============================================================
    // Kernels (static to avoid multiple definition across TUs)
    // ============================================================

    // (0) Identity weights: w[k] = gain * (bake_invN ? 1/N : 1)
    static __global__ void kernel_fill_identity_weights(
        float* __restrict__ w,
        int n_complex,
        int N,
        float gain,
        bool bake_invN)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_complex) return;
        float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
        w[k] = gain * invN;
    }

    // (1) Analytic frequency-domain build (direct)
    // w[k] = gain * ramp(f or k) * window(x) * (optional 1/N), cutoff applied
    static __global__ void kernel_build_weights_analytic_freq(
        float* __restrict__ w,
        int n_complex,
        int N,
        SFilterKernelDesc desc,
        bool bake_invN)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_complex) return;

        // None => identity (do not force DC unless user asked)
        if (desc.kind == EFilterKernel::None) {
            float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
            w[k] = desc.gain * invN;
            return;
        }

        float f = (N > 0) ? ((float)k / (float)N) : 0.0f; // [0,0.5]

        if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; return; }

        float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
        if (f > cc) { w[k] = 0.0f; return; }

        float ramp = f;
        float x = (cc > 0.0f) ? (f / cc) : 0.0f;
        float shape = yk_window_shape_device(x, desc.kind);

        float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
        w[k] = desc.gain * ramp * shape * invN;
    }

    // (2) Spatial discrete Ram-Lak kernel (DU=1 convention)
    static __global__ void kernel_gen_spatial_rl_kernel_du1(
        float* __restrict__ h,
        int N,
        bool bake_invN)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u >= N) return;

        int n = (u <= N / 2) ? u : (u - N);
        int an = (n < 0) ? -n : n;

        float val = 0.0f;
        if (n == 0) {
            val = 1.0f / 4.0f;
        }
        else if (an & 1) {
            const float pi = 3.14159265358979323846f;
            float fn = (float)n;
            val = -1.0f / (pi * pi * fn * fn);
        }
        else {
            val = 0.0f;
        }

        if (bake_invN && N > 0) val *= (1.0f / (float)N);
        h[u] = val;
    }

    // (3) Extract ramp weights from FFT(RL)
    static __global__ void kernel_extract_weights_from_fft(
        const cufftComplex* __restrict__ src,
        float* __restrict__ dst,
        int n_complex,
        int mode,             // 0=RealPart, 1=Magnitude
        bool force_dc_zero)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_complex) return;

        float re = src[k].x;
        float im = src[k].y;
        float v = (mode == 1) ? sqrtf(re * re + im * im) : re;

        if (force_dc_zero && k == 0) v = 0.0f;
        dst[k] = v;
    }

    // (4) Apply window/cutoff/gain/DC on ramp weights (in-place)
    static __global__ void kernel_apply_window_to_weights_inplace(
        float* __restrict__ w,
        int n_complex,
        int N,
        SFilterKernelDesc desc)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_complex) return;

        // None should not normally come here, but keep safe
        if (desc.kind == EFilterKernel::None) {
            w[k] = w[k] * desc.gain;
            return;
        }

        float f = (N > 0) ? ((float)k / (float)N) : 0.0f;

        if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; return; }

        float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
        if (f > cc) { w[k] = 0.0f; return; }

        float x = (cc > 0.0f) ? (f / cc) : 0.0f;
        float shape = yk_window_shape_device(x, desc.kind);

        w[k] = w[k] * (desc.gain * shape);
    }

    // ============================================================
    // FilterKernelFFT: NEW single public API, but keeps both build paths
    // ============================================================
    class FilterKernelFFT {
    public:


        FilterKernelFFT() = default;
        ~FilterKernelFFT() { release(); }

        FilterKernelFFT(const FilterKernelFFT&) = delete;
        FilterKernelFFT& operator=(const FilterKernelFFT&) = delete;

        FilterKernelFFT(FilterKernelFFT&& o) noexcept { move_from(o); }
        FilterKernelFFT& operator=(FilterKernelFFT&& o) noexcept {
            if (this != &o) { release(); move_from(o); }
            return *this;
        }

        void prepare(int paddedN, cudaStream_t stream = 0)
        {
            if (paddedN <= 0) {
                YK_ASSERT(false && "paddedN must be > 0");
                return;
            }
            if (ready_ && paddedN_ == paddedN) {
                setStream(stream);
                return;
            }

            release();

            paddedN_ = paddedN;
            n_complex_ = paddedN_ / 2 + 1;
            stream_ = stream;

            YK_CUDA_CHECK(cudaMalloc(&d_tmp_fft_, (size_t)n_complex_ * sizeof(cufftComplex)));

            bool ok = fft_r2c_.init(paddedN_, /*batch=*/1, YK::CudaFFT::EPlanMode::R2COnly, stream_);
            YK_ASSERT(ok && "CudaFFT R2COnly init failed");
            ready_ = ok;
        }

        void setStream(cudaStream_t s)
        {
            stream_ = s;
            if (ready_) fft_r2c_.setStream(stream_);
        }

        void release()
        {
            if (d_tmp_fft_) {
                YK_CUDA_CHECK(cudaFree(d_tmp_fft_));
                d_tmp_fft_ = nullptr;
            }
            fft_r2c_.release();

            paddedN_ = 0;
            n_complex_ = 0;
            stream_ = 0;
            ready_ = false;
        }

        int paddedN() const { return paddedN_; }
        int n_complex() const { return n_complex_; }
        cudaStream_t stream() const { return stream_; }

        float* alloc_weights() const
        {
            YK_ASSERT(ready_);
            float* d_w = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_w, (size_t)n_complex_ * sizeof(float)));
            return d_w;
        }

        // ============================================================
        // THE SINGLE PUBLIC BUILD API (NEW)
        // Keeps BOTH:
        //  - desc.source == AnalyticFreq     : direct frequency build
        //  - desc.source == DiscreteRLFFT    : spatial RL -> FFT -> ramp -> window
        //
        // bake_invN:
        //  - true : bake 1/N into weights (or RL kernel) to cancel cuFFT C2R gain N
        //
        // mode:
        //  - only affects DiscreteRLFFT (extract from RL spectrum)
        // ============================================================
        void build_weights(
            float* d_weights_fft,               // [n_complex]
            const SFilterKernelDesc& desc,
            bool bake_invN = true) const
        {
            YK_ASSERT(ready_);
            YK_ASSERT(d_weights_fft);

            dim3 block(256, 1);
            dim3 gridC((n_complex_ + block.x - 1) / block.x, 1);

            // kind=None is identity regardless of source
            if (desc.kind == EFilterKernel::None) {
                kernel_fill_identity_weights << <gridC, block, 0, stream_ >> > (
                    d_weights_fft, n_complex_, paddedN_, desc.gain, bake_invN);
                YK_CUDA_KERNEL_CHECK();
                return;
            }

            if (desc.source == EWeightsBuildSource::AnalyticFreq) {
                // direct frequency-domain generation (kept!)
                kernel_build_weights_analytic_freq << <gridC, block, 0, stream_ >> > (
                    d_weights_fft, n_complex_, paddedN_, desc, bake_invN);
                YK_CUDA_KERNEL_CHECK();
                return;
            }

            // DiscreteRLFFT path (kept!)
            float* d_spatial = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_spatial, (size_t)paddedN_ * sizeof(float)));

            dim3 gridN((paddedN_ + block.x - 1) / block.x, 1);
            kernel_gen_spatial_rl_kernel_du1 << <gridN, block, 0, stream_ >> > (
                d_spatial, paddedN_, bake_invN);
            YK_CUDA_KERNEL_CHECK();

            // RL -> spectrum
            fft_r2c_.fft(d_spatial, d_tmp_fft_);

            auto mode = desc.extract_mode;

            // extract ramp
            kernel_extract_weights_from_fft << <gridC, block, 0, stream_ >> > (
                d_tmp_fft_, d_weights_fft, n_complex_, (int)mode, desc.force_dc_zero);
            YK_CUDA_KERNEL_CHECK();

            YK_CUDA_CHECK(cudaFree(d_spatial));

            // apply window/cutoff/gain/DC
            kernel_apply_window_to_weights_inplace << <gridC, block, 0, stream_ >> > (
                d_weights_fft, n_complex_, paddedN_, desc);
            YK_CUDA_KERNEL_CHECK();
        }

    private:
        void move_from(FilterKernelFFT& o) noexcept
        {
            paddedN_ = o.paddedN_;   o.paddedN_ = 0;
            n_complex_ = o.n_complex_; o.n_complex_ = 0;
            stream_ = o.stream_;    o.stream_ = 0;
            ready_ = o.ready_;     o.ready_ = false;

            d_tmp_fft_ = o.d_tmp_fft_; o.d_tmp_fft_ = nullptr;
            fft_r2c_ = std::move(o.fft_r2c_);
        }

    private:
        int paddedN_ = 0;
        int n_complex_ = 0;
        cudaStream_t stream_ = 0;
        bool ready_ = false;

        cufftComplex* d_tmp_fft_ = nullptr;
        YK::CudaFFT fft_r2c_; // R2C only
    };

} // namespace YK
