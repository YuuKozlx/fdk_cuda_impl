#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "Filter/YkConv.hpp"
#include "global/YkWarpStrideCtx.cuh"
#include "YkFDKFilterLaunch.cuh"



namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // _fp_scale_inplace
            //   原地将 data[0..n) 乘以标量 s
            // ----------------------------------------------------------------
            static __global__ void _fp_scale_inplace(float* data, int n, float s)
            {
                WarpStrideCtx<EWarpStrideAxis::X> ctx;

                for (int base = ctx.warp_global * 32; base < n; base += ctx.n_warps * 32)
                {
                    int i = base + ctx.lane;
                    if (i >= n) break;
                    data[i] *= s;
                }
            }

            // ----------------------------------------------------------------
            // _fp_pad_kernel — 改用 WarpStrideCtx<RowWarp>
            // ----------------------------------------------------------------
            __global__ void _fp_pad_kernel(
                const float* __restrict__ src,
                float* __restrict__       dst,
                const int* __restrict__   startu,
                int Nu, int Nv, int paddedN, int K, int bounds_check)
            {
                WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;

                for (int row = ctx.warp_global; row < K * Nv; row += ctx.n_warps)
                {
                    const int i = row / Nv;
                    const int v = row - i * Nv;
                    if (i >= K) continue;

                    const int    sU = startu[i];
                    const size_t base_src = (static_cast<size_t>(i) * Nv + v) * Nu;
                    const size_t base_dst = (static_cast<size_t>(i) * Nv + v) * paddedN;

                    for (int u = ctx.lane; u < paddedN; u += 32)
                    {
                        const int su = u - sU;
                        float val = 0.0f;
                        if (!bounds_check)
                            val = src[base_src + su];
                        else if (static_cast<unsigned>(su) < static_cast<unsigned>(Nu))
                            val = src[base_src + su];
                        dst[base_dst + u] = val;
                    }
                }
            }

            // ----------------------------------------------------------------
            // _fp_crop_kernel — 改用 WarpStrideCtx<RowWarp>
            // ----------------------------------------------------------------
            __global__ void _fp_crop_kernel(
                const float* __restrict__ src,
                float* __restrict__       dst,
                const int* __restrict__   startu,
                int Nu, int Nv, int paddedN, int K, int bounds_check)
            {
                WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;

                for (int row = ctx.warp_global; row < K * Nv; row += ctx.n_warps)
                {
                    const int i = row / Nv;
                    const int v = row - i * Nv;
                    if (i >= K) continue;

                    const int    sU = startu[i];
                    const size_t base_src = (static_cast<size_t>(i) * Nv + v) * paddedN;
                    const size_t base_dst = (static_cast<size_t>(i) * Nv + v) * Nu;

                    for (int u = ctx.lane; u < Nu; u += 32)
                    {
                        const int su = sU + u;
                        float val = 0.0f;
                        if (!bounds_check)
                            val = src[base_src + su];
                        else if (static_cast<unsigned>(su) < static_cast<unsigned>(paddedN))
                            val = src[base_src + su];
                        dst[base_dst + u] = val;
                    }
                }
            }


            void fp_launchPad(
                const float* d_src, float* d_padded, const int* d_startu,
                int Nu, int Nv, int paddedN, int K,
                const SKernelLaunchPolicy& policy, cudaStream_t stream)
            {
                const auto launch = policy.makeRowWarp((size_t)K * Nv);

                _fp_pad_kernel << <launch.grid, launch.block, 0, stream >> > (
                    d_src, d_padded, d_startu,
                    Nu, Nv, paddedN, K,
                    policy.bounds_check ? 1 : 0);
                YK_CUDA_KERNEL_CHECK();
            }

            void fp_launchFilter(
                float* d_padded, cufftComplex* d_complex_buf,
                const float* d_weights,
                int n_cmplx, int paddedN, int K, int Nv,
                CudaFFT& fft_batch, cudaStream_t stream)
            {
                const int batch = K * Nv;

                fft_batch.fft(d_padded, d_complex_buf);

                YK::Filter::launch_pointwise_mul(d_complex_buf, d_weights, n_cmplx, batch, stream);
                YK_CUDA_KERNEL_CHECK();

                fft_batch.ifft(d_complex_buf, d_padded);
            }

            void fp_launchCrop(
                const float* d_padded, float* d_dst, const int* d_startu,
                int Nu, int Nv, int paddedN, int K,
                const SKernelLaunchPolicy& policy, cudaStream_t stream)
            {
                const auto launch = policy.makeRowWarp((size_t)K * Nv);

                _fp_crop_kernel << <launch.grid, launch.block, 0, stream >> > (
                    d_padded, d_dst, d_startu,
                    Nu, Nv, paddedN, K,
                    policy.bounds_check ? 1 : 0);
                YK_CUDA_KERNEL_CHECK();
            }
        }; // namespace YK::Fdk::detail
    };
};

//namespace YK {
//    namespace Fdk {
//        namespace detail {
//
//            namespace cg = cooperative_groups;
//
//            // ----------------------------------------------------------------
//            // _fp_scale_inplace
//            //   原地将 data[0..n) 乘以标量 s
//            // ----------------------------------------------------------------
//            static __global__ void _fp_scale_inplace(float* data, int n, float s)
//            {
//                const int i = blockIdx.x * blockDim.x + threadIdx.x;
//                if (i < n) data[i] *= s;
//            }
//
//            // ----------------------------------------------------------------
//            // _fp_pad_kernel
//            //   将 [K, Nv, Nu] 的投影拷贝到 [K, Nv, paddedN] 的 zero-padded 缓冲。
//            //   每个 warp 负责一行 (i, v)；startu[i] 给出该视角的列偏移。
//            //   bounds_check != 0 时跳过越界读取（非对称截断时使用）。
//            // ----------------------------------------------------------------
//            __global__ void _fp_pad_kernel(
//                const float* __restrict__ src,
//                float* __restrict__ dst,
//                const int* __restrict__ startu,
//                int Nu, int Nv, int paddedN, int K, int bounds_check)
//            {
//                cg::thread_block              tb = cg::this_thread_block();
//                cg::thread_block_tile<32>     warp = cg::tiled_partition<32>(tb);
//
//                const int wpb = static_cast<int>(tb.size() / 32);
//                const int warp_global = static_cast<int>(blockIdx.x) * wpb
//                    + static_cast<int>(tb.thread_rank() / 32);
//                if (warp_global >= K * Nv) return;
//
//                const int i = warp_global / Nv;
//                const int v = warp_global - i * Nv;
//                const int sU = startu[i];
//
//                const size_t base_src = (static_cast<size_t>(i) * Nv + v) * Nu;
//                const size_t base_dst = (static_cast<size_t>(i) * Nv + v) * paddedN;
//
//                for (int u = static_cast<int>(warp.thread_rank()); u < paddedN; u += 32) {
//                    const int su = u - sU;
//                    float val = 0.0f;
//                    if (!bounds_check)
//                        val = src[base_src + su];
//                    else if (static_cast<unsigned>(su) < static_cast<unsigned>(Nu))
//                        val = src[base_src + su];
//                    dst[base_dst + u] = val;
//                }
//            }
//
//            // ----------------------------------------------------------------
//            // _fp_crop_kernel
//            //   将 [K, Nv, paddedN] 的滤波结果裁回 [K, Nv, Nu]。
//            //   每个 warp 负责一行 (i, v)；startu[i] 给出列偏移。
//            // ----------------------------------------------------------------
//            __global__ void _fp_crop_kernel(
//                const float* __restrict__ src,
//                float* __restrict__ dst,
//                const int* __restrict__ startu,
//                int Nu, int Nv, int paddedN, int K, int bounds_check)
//            {
//                cg::thread_block              tb = cg::this_thread_block();
//                cg::thread_block_tile<32>     warp = cg::tiled_partition<32>(tb);
//
//                const int wpb = static_cast<int>(tb.size() / 32);
//                const int warp_global = static_cast<int>(blockIdx.x) * wpb
//                    + static_cast<int>(tb.thread_rank() / 32);
//                if (warp_global >= K * Nv) return;
//
//                const int i = warp_global / Nv;
//                const int v = warp_global - i * Nv;
//                const int sU = startu[i];
//
//                const size_t base_src = (static_cast<size_t>(i) * Nv + v) * paddedN;
//                const size_t base_dst = (static_cast<size_t>(i) * Nv + v) * Nu;
//
//                for (int u = static_cast<int>(warp.thread_rank()); u < Nu; u += 32) {
//                    const int su = sU + u;
//                    float val = 0.0f;
//                    if (!bounds_check)
//                        val = src[base_src + su];
//                    else if (static_cast<unsigned>(su) < static_cast<unsigned>(paddedN))
//                        val = src[base_src + su];
//                    dst[base_dst + u] = val;
//                }
//            }
//
//
//            // ----------------------------------------------------------------
//           // fp_launchPad
//           //   把 [K, Nv, Nu] 的 d_src 零填充到 [K, Nv, paddedN] 的 d_padded。
//           // ----------------------------------------------------------------
//            void fp_launchPad(
//                const float* d_src,
//                float* d_padded,
//                const int* d_startu,
//                int Nu, int Nv, int paddedN, int K,
//                const SKernelLaunchPolicy& policy,
//                cudaStream_t               stream)
//            {
//                const int bk = policy.block_threads;          // 已归一化，必为 32 的倍数
//                const int wpb = bk / 32;
//                const int blk = (K * Nv + wpb - 1) / wpb;
//
//                _fp_pad_kernel << <blk, bk, 0, stream >> > (
//                    d_src, d_padded, d_startu,
//                    Nu, Nv, paddedN, K,
//                    policy.bounds_check ? 1 : 0);
//                YK_CUDA_KERNEL_CHECK();
//            }
//
//            // ----------------------------------------------------------------
//            // fp_launchFilter
//            //   对 d_padded（[K*Nv, paddedN]）执行 FFT → 逐点乘权重 → IFFT，
//            //   并在 postScale != 1.f 时做原地缩放。
//            // ----------------------------------------------------------------
//            void fp_launchFilter(
//                float* d_padded,
//                cufftComplex* d_complex_buf,
//                const float* d_weights,
//                int             n_cmplx,
//                int             paddedN,
//                int             K,
//                int             Nv,
//                float           postScale,
//                CudaFFT& fft_batch,
//                cudaStream_t    stream)
//            {
//                const int batch = K * Nv;
//
//                fft_batch.fft(d_padded, d_complex_buf);
//
//                {
//
//                    YK::Filter::launch_pointwise_mul(d_complex_buf, d_weights, n_cmplx, batch, stream);
//                    YK_CUDA_KERNEL_CHECK();
//                }
//
//                fft_batch.ifft(d_complex_buf, d_padded);
//
//                if (postScale != 1.0f) {
//                    const int total = batch * paddedN;
//                    _fp_scale_inplace << <(total + 255) / 256, 256, 0, stream >> > (
//                        d_padded, total, postScale);
//                    YK_CUDA_KERNEL_CHECK();
//                }
//            }
//
//            // ----------------------------------------------------------------
//            // fp_launchCrop
//            //   把 [K, Nv, paddedN] 的 d_padded 裁回 [K, Nv, Nu] 的 d_dst。
//            // ----------------------------------------------------------------
//            void fp_launchCrop(
//                const float* d_padded,
//                float* d_dst,
//                const int* d_startu,
//                int Nu, int Nv, int paddedN, int K,
//                const SKernelLaunchPolicy& policy,
//                cudaStream_t               stream)
//            {
//                const int bk = policy.block_threads;
//                const int wpb = bk / 32;
//                const int blk = (K * Nv + wpb - 1) / wpb;
//
//                _fp_crop_kernel << <blk, bk, 0, stream >> > (
//                    d_padded, d_dst, d_startu,
//                    Nu, Nv, paddedN, K,
//                    policy.bounds_check ? 1 : 0);
//                YK_CUDA_KERNEL_CHECK();
//            }
//
//        }; // namespace YK::Fdk::detail
//    };
//};
