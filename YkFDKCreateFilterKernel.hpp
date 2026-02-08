#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cmath>

#include "YkGlobals.h"
#include "YkFFT.hpp"

namespace YK {

    // ============================================================
    // Filter options
    // ============================================================
    enum class EFilterKernel {
        RamLak,
        SheppLogan,
        Cosine,
        Hann,
        Hamming
    };

    struct FilterKernelDesc {
        EFilterKernel kind = EFilterKernel::RamLak;

        // IMPORTANT:
        // cutoff is in DFT-normalized frequency:
        //   f = k / N   in [0, 0.5], Nyquist = 0.5
        // So default cutoff=0.5 means full Nyquist.
        float cutoff = 0.5f;

        float gain = 1.0f;

        // true  => ramp = f   (f = k/N, DU=1 convention)
        // false => ramp = k   (index ramp, still DU=1)
        bool normalized_ramp = true;

        // Make DC consistent between analytic & discrete
        bool force_dc_zero = true;
    };

    // ============================================================
    // Device helpers
    // ============================================================
    __device__ __forceinline__ float yk_sinc_pi_device(float x) {
        const float pi = 3.14159265358979323846f;
        float t = pi * x;
        if (fabsf(t) < 1e-8f) return 1.0f;
        return sinf(t) / t;
    }

    // ============================================================
    // (A) Analytic weights in frequency domain (DU=1 convention)
    //
    // Use DFT-normalized frequency: f = k / N  in [0, 0.5]
    // This matches the DFT of the discrete Ram-Lak kernel (DU=1).
    //
    // w[k] = gain * ramp * windowShape * (optional 1/N)
    // NOTE: NO du scaling here!
    // NOTE: DO NOT multiply by 2 for "symmetric spectrum" when using cuFFT R2C half-spectrum.
    // ============================================================
    __global__ void kernel_build_filter_weights_fft(
        float* __restrict__ w,
        int n_complex,
        int N,                 // N == paddedN
        FilterKernelDesc desc,
        bool bake_invN)        // bake 1/N into w (to cancel cuFFT C2R gain N)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_complex) return;

        // DFT-normalized frequency: f = k/N, Nyquist = 0.5
        float f = (N > 0) ? ((float)k / (float)N) : 0.0f; // [0, 0.5]

        if (desc.force_dc_zero && k == 0) {
            w[k] = 0.0f;
            return;
        }

        if (f > desc.cutoff) {
            w[k] = 0.0f;
            return;
        }

        float ramp = desc.normalized_ramp ? f : (float)k;

        float shape = 1.0f;
        const float pi = 3.14159265358979323846f;

        // normalize x to [0,1] over [0, cutoff]
        float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
        float x = f / cc;

        switch (desc.kind) {
        case EFilterKernel::RamLak:     shape = 1.0f; break;
        case EFilterKernel::SheppLogan: shape = yk_sinc_pi_device(x * 0.5f); break; // sinc(pi*x/2)
        case EFilterKernel::Cosine:     shape = cosf(0.5f * pi * x); break;
        case EFilterKernel::Hann:       shape = 0.5f * (1.0f + cosf(pi * x)); break;
        case EFilterKernel::Hamming:    shape = 0.54f + 0.46f * cosf(pi * x); break;
        default:                        shape = 1.0f; break;
        }

        float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
        w[k] = desc.gain * ramp * shape * invN;
    }

    // ============================================================
    // (B) Generate spatial discrete Ram-Lak kernel (DU=1 convention)
    //     h[0]      = 1/4
    //     h[n odd]  = -1/(pi^2 n^2)
    //     h[n even] = 0
    //
    // circular shift: n=0 at idx0 (so FFT directly yields non-negative bins)
    // optional: bake 1/N into h to cancel cuFFT C2R scaling (N gain)
    //
    // NOTE: NO du scaling here!
    // ============================================================
    __global__ void kernel_gen_spatial_rl_kernel_du1(
        float* __restrict__ h,
        int N,           // paddedN
        bool bake_invN)  // include 1/N in h
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u >= N) return;

        int n = (u <= N / 2) ? u : (u - N);
        int an = (n < 0) ? -n : n;

        float val = 0.0f;
        if (n == 0) {
            val = 1.0f / 4.0f;
        }
        else if (an & 1) { // odd
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

    // ============================================================
    // Extract weights from FFT(kernel)
    //   - RealPart:   dst[k]=Re
    //   - Magnitude:  dst[k]=sqrt(Re^2+Im^2)
    // Optionally force DC to 0 for alignment with analytic ramp.
    // ============================================================
    __global__ void kernel_extract_weights_from_fft(
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

    // ============================================================
    // FilterKernelFFT: build weights for FilterManager
    //  - All weights are DU=1 convention (no du)
    //  - bake_invN=true recommended if FilterManager does NOT divide by N after IFFT
    // ============================================================
    class FilterKernelFFT {
    public:
        enum class EKernelToWeightsMode {
            RealPart = 0,
            Magnitude = 1
        };

        FilterKernelFFT() = default;
        ~FilterKernelFFT() { release(); }

        FilterKernelFFT(const FilterKernelFFT&) = delete;
        FilterKernelFFT& operator=(const FilterKernelFFT&) = delete;

        FilterKernelFFT(FilterKernelFFT&& o) noexcept { move_from(o); }
        FilterKernelFFT& operator=(FilterKernelFFT&& o) noexcept {
            if (this != &o) { release(); move_from(o); }
            return *this;
        }

        void prepare(int paddedN, cudaStream_t stream = 0) {
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

        void setStream(cudaStream_t s) {
            stream_ = s;
            if (ready_) fft_r2c_.setStream(stream_);
        }

        void release() {
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

        float* alloc_weights() const {
            YK_ASSERT(ready_);
            float* d_w = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_w, (size_t)n_complex_ * sizeof(float)));
            return d_w;
        }

        // Analytic (DU=1, DFT-normalized f=k/N). bake_invN recommended.
        void build_analytic(float* d_weights_fft, const FilterKernelDesc& desc, bool bake_invN = true) const {
            YK_ASSERT(ready_);
            YK_ASSERT(d_weights_fft);

            dim3 block(256, 1);
            dim3 grid((n_complex_ + block.x - 1) / block.x, 1);

            YK::kernel_build_filter_weights_fft << <grid, block, 0, stream_ >> > (
                d_weights_fft, n_complex_, paddedN_, desc, bake_invN);
            YK_CUDA_KERNEL_CHECK();
        }

        // Discrete RL (DU=1) -> FFT -> extract.
        // bake_invN should match analytic path.
        // force_dc_zero should match desc.force_dc_zero in analytic path.
        void build_discrete_ramlak_du1(
            float* d_weights_fft,
            bool bake_invN = true,
            EKernelToWeightsMode mode = EKernelToWeightsMode::RealPart,
            bool force_dc_zero = false) const
        {
            YK_ASSERT(ready_);
            YK_ASSERT(d_weights_fft);

            float* d_spatial = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_spatial, (size_t)paddedN_ * sizeof(float)));

            dim3 block(256, 1);
            dim3 gridN((paddedN_ + block.x - 1) / block.x, 1);

            YK::kernel_gen_spatial_rl_kernel_du1 << <gridN, block, 0, stream_ >> > (
                d_spatial, paddedN_, bake_invN);
            YK_CUDA_KERNEL_CHECK();

            // R2C FFT
            fft_r2c_.fft(d_spatial, d_tmp_fft_);

            dim3 gridC((n_complex_ + block.x - 1) / block.x, 1);
            YK::kernel_extract_weights_from_fft << <gridC, block, 0, stream_ >> > (
                d_tmp_fft_, d_weights_fft, n_complex_, (int)mode, force_dc_zero);
            YK_CUDA_KERNEL_CHECK();

            YK_CUDA_CHECK(cudaFree(d_spatial));
        }

        int paddedN()   const { return paddedN_; }
        int n_complex() const { return n_complex_; }
        cudaStream_t stream() const { return stream_; }

    private:
        void move_from(FilterKernelFFT& o) noexcept {
            paddedN_ = o.paddedN_; o.paddedN_ = 0;
            n_complex_ = o.n_complex_; o.n_complex_ = 0;
            stream_ = o.stream_; o.stream_ = 0;
            ready_ = o.ready_; o.ready_ = false;

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
