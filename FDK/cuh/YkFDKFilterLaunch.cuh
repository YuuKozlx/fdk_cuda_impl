#pragma once
#include <cufft.h>
#include <cuda_runtime.h>

#include "../../Filter/YkConv.hpp"               // _kernel_pointwise_mul
#include "../../Filter/YkFFT.hpp"                // CudaFFT
#include "../../FDK/YkFdkPipelineContext.hpp" // SKernelLaunchPolicy
#include "../../global/YkMacro.hpp"              // YK_CUDA_KERNEL_CHECK

#include "YkFDKFilterKernels.cuh"


namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // fp_launchPad
            //   把 [K, Nv, Nu] 的 d_src 零填充到 [K, Nv, paddedN] 的 d_padded。
            // ----------------------------------------------------------------
            inline void fp_launchPad(
                const float* d_src,
                float* d_padded,
                const int* d_startu,
                int Nu, int Nv, int paddedN, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream)
            {
                const int bk = policy.block_threads;          // 已归一化，必为 32 的倍数
                const int wpb = bk / 32;
                const int blk = (K * Nv + wpb - 1) / wpb;

                _fp_pad_kernel << <blk, bk, 0, stream >> > (
                    d_src, d_padded, d_startu,
                    Nu, Nv, paddedN, K,
                    policy.bounds_check ? 1 : 0);
                YK_CUDA_KERNEL_CHECK();
            }

            // ----------------------------------------------------------------
            // fp_launchFilter
            //   对 d_padded（[K*Nv, paddedN]）执行 FFT → 逐点乘权重 → IFFT，
            //   并在 postScale != 1.f 时做原地缩放。
            // ----------------------------------------------------------------
            inline void fp_launchFilter(
                float* d_padded,
                cufftComplex* d_complex_buf,
                const float* d_weights,
                int             n_cmplx,
                int             paddedN,
                int             K,
                int             Nv,
                float           postScale,
                CudaFFT& fft_batch,
                cudaStream_t    stream)
            {
                const int batch = K * Nv;

                fft_batch.fft(d_padded, d_complex_buf);

                {
                    dim3 blk(256);
                    dim3 grd((n_cmplx + 255) / 256, batch);
                    _kernel_pointwise_mul << <grd, blk, 0, stream >> > (
                        d_complex_buf, d_weights, n_cmplx, batch);
                    YK_CUDA_KERNEL_CHECK();
                }

                fft_batch.ifft(d_complex_buf, d_padded);

                if (postScale != 1.0f) {
                    const int total = batch * paddedN;
                    _fp_scale_inplace << <(total + 255) / 256, 256, 0, stream >> > (
                        d_padded, total, postScale);
                    YK_CUDA_KERNEL_CHECK();
                }
            }

            // ----------------------------------------------------------------
            // fp_launchCrop
            //   把 [K, Nv, paddedN] 的 d_padded 裁回 [K, Nv, Nu] 的 d_dst。
            // ----------------------------------------------------------------
            inline void fp_launchCrop(
                const float* d_padded,
                float* d_dst,
                const int* d_startu,
                int Nu, int Nv, int paddedN, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream)
            {
                const int bk = policy.block_threads;
                const int wpb = bk / 32;
                const int blk = (K * Nv + wpb - 1) / wpb;

                _fp_crop_kernel << <blk, bk, 0, stream >> > (
                    d_padded, d_dst, d_startu,
                    Nu, Nv, paddedN, K,
                    policy.bounds_check ? 1 : 0);
                YK_CUDA_KERNEL_CHECK();
            }

        }; // namespace YK::Fdk::detail
    };
};
