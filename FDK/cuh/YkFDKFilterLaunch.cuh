#pragma once
#include <cufft.h>
#include <cuda_runtime.h>

#include "../../Filter/YkConv.hpp"               // _kernel_pointwise_mul
#include "../../Filter/YkFFT.hpp"                // CudaFFT
#include "../../FDK/YkFdkPipelineContext.hpp" // SKernelLaunchPolicy
#include "../../global/YkMacro.hpp"              // YK_CUDA_KERNEL_CHECK



namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // fp_launchPad
            //   把 [K, Nv, Nu] 的 d_src 零填充到 [K, Nv, paddedN] 的 d_padded。
            // ----------------------------------------------------------------
            void fp_launchPad(
                const float* d_src,
                float* d_padded,
                const int* d_startu,
                int Nu, int Nv, int paddedN, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream);

            // ----------------------------------------------------------------
            // fp_launchFilter
            //   对 d_padded（[K*Nv, paddedN]）执行 FFT → 逐点乘权重 → IFFT，
            //   并在 postScale != 1.f 时做原地缩放。
            // ----------------------------------------------------------------
            void fp_launchFilter(
                float* d_padded,
                cufftComplex* d_complex_buf,
                const float* d_weights,
                int             n_cmplx,
                int             paddedN,
                int             K,
                int             Nv,
                float           postScale,
                CudaFFT& fft_batch,
                cudaStream_t    stream);


            // ----------------------------------------------------------------
            // fp_launchCrop
            //   把 [K, Nv, paddedN] 的 d_padded 裁回 [K, Nv, Nu] 的 d_dst。
            // ----------------------------------------------------------------
            void fp_launchCrop(
                const float* d_padded,
                float* d_dst,
                const int* d_startu,
                int Nu, int Nv, int paddedN, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream);


        }; // namespace YK::Fdk::detail
    };
};
