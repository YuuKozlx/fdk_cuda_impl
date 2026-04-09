#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <utility>


#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../Filter/YkFFT.hpp"
#include "YkFilterKernelKernels.cuh"

namespace YK {
    namespace Filter {

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

            // --------------------------------------------------------
            // Lifecycle
            // --------------------------------------------------------

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

            // --------------------------------------------------------
            // Accessors
            // --------------------------------------------------------

            int          paddedN()   const { return paddedN_; }
            int          n_complex() const { return n_complex_; }
            cudaStream_t stream()    const { return stream_; }

            /// Allocate a device buffer sized for the weight array.
            /// Caller takes ownership; free with cudaFree().
            float* alloc_weights() const
            {
                YK_ASSERT(ready_);
                float* d_w = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_w, (size_t)n_complex_ * sizeof(float)));
                return d_w;
            }

            // --------------------------------------------------------
            // Build
            //
            // Fills d_weights_fft[n_complex] according to desc.
            //
            // bake_invN = true  : bakes 1/N into weights to cancel the
            //                     cuFFT C2R gain of N downstream.
            //
            // Two build paths (selected by desc.source):
            //   AnalyticFreq   – direct frequency-domain construction
            //   DiscreteRLFFT  – spatial RL kernel -> FFT -> window
            // --------------------------------------------------------
            void build_weights(
                float* d_weights_fft,
                const SFilterKernelDesc& desc,
                bool bake_invN = true) const
            {
                YK_ASSERT(ready_);
                YK_ASSERT(d_weights_fft);

                using namespace detail;

                dim3 block(256, 1);
                dim3 gridC((n_complex_ + block.x - 1) / block.x, 1);

                if (desc.kind == EFilterKernel::None) {
                    kernel_fill_identity_weights << <gridC, block, 0, stream_ >> > (
                        d_weights_fft, n_complex_, paddedN_, desc.gain, bake_invN);
                    YK_CUDA_KERNEL_CHECK();
                    return;
                }

                if (desc.source == EWeightsBuildSource::AnalyticFreq) {
                    kernel_build_weights_analytic_freq << <gridC, block, 0, stream_ >> > (
                        d_weights_fft, n_complex_, paddedN_, desc, bake_invN);
                    YK_CUDA_KERNEL_CHECK();
                    return;
                }

                // DiscreteRLFFT path
                float* d_spatial = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_spatial, (size_t)paddedN_ * sizeof(float)));

                dim3 gridN((paddedN_ + block.x - 1) / block.x, 1);
                kernel_gen_spatial_rl_kernel_du1 << <gridN, block, 0, stream_ >> > (
                    d_spatial, paddedN_, bake_invN);
                YK_CUDA_KERNEL_CHECK();

                fft_r2c_.fft(d_spatial, d_tmp_fft_);

                kernel_extract_weights_from_fft << <gridC, block, 0, stream_ >> > (
                    d_tmp_fft_, d_weights_fft, n_complex_,
                    (int)desc.extract_mode, desc.force_dc_zero);
                YK_CUDA_KERNEL_CHECK();

                YK_CUDA_CHECK(cudaFree(d_spatial));

                kernel_apply_window_to_weights_inplace << <gridC, block, 0, stream_ >> > (
                    d_weights_fft, n_complex_, paddedN_, desc);
                YK_CUDA_KERNEL_CHECK();
            }

        private:
            void move_from(FilterKernelFFT& o) noexcept
            {
                paddedN_ = o.paddedN_;    o.paddedN_ = 0;
                n_complex_ = o.n_complex_;  o.n_complex_ = 0;
                stream_ = o.stream_;     o.stream_ = 0;
                ready_ = o.ready_;      o.ready_ = false;
                d_tmp_fft_ = o.d_tmp_fft_;  o.d_tmp_fft_ = nullptr;
                fft_r2c_ = std::move(o.fft_r2c_);
            }

        private:
            int           paddedN_ = 0;
            int           n_complex_ = 0;
            cudaStream_t  stream_ = 0;
            bool          ready_ = false;

            cufftComplex* d_tmp_fft_ = nullptr;
            YK::CudaFFT   fft_r2c_;   // R2C only
        };

    } // namespace Filter
} // namespace YK
