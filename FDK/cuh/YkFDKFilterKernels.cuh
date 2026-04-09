#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace YK {
    namespace Fdk {
        namespace detail {

            namespace cg = cooperative_groups;

            // ----------------------------------------------------------------
            // _fp_scale_inplace
            //   原地将 data[0..n) 乘以标量 s
            // ----------------------------------------------------------------
            static __global__ void _fp_scale_inplace(float* data, int n, float s)
            {
                const int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) data[i] *= s;
            }

            // ----------------------------------------------------------------
            // _fp_pad_kernel
            //   将 [K, Nv, Nu] 的投影拷贝到 [K, Nv, paddedN] 的 zero-padded 缓冲。
            //   每个 warp 负责一行 (i, v)；startu[i] 给出该视角的列偏移。
            //   bounds_check != 0 时跳过越界读取（非对称截断时使用）。
            // ----------------------------------------------------------------
            __global__ void _fp_pad_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int* __restrict__ startu,
                int Nu, int Nv, int paddedN, int K, int bounds_check)
            {
                cg::thread_block              tb = cg::this_thread_block();
                cg::thread_block_tile<32>     warp = cg::tiled_partition<32>(tb);

                const int wpb = static_cast<int>(tb.size() / 32);
                const int warp_global = static_cast<int>(blockIdx.x) * wpb
                    + static_cast<int>(tb.thread_rank() / 32);
                if (warp_global >= K * Nv) return;

                const int i = warp_global / Nv;
                const int v = warp_global - i * Nv;
                const int sU = startu[i];

                const size_t base_src = (static_cast<size_t>(i) * Nv + v) * Nu;
                const size_t base_dst = (static_cast<size_t>(i) * Nv + v) * paddedN;

                for (int u = static_cast<int>(warp.thread_rank()); u < paddedN; u += 32) {
                    const int su = u - sU;
                    float val = 0.0f;
                    if (!bounds_check)
                        val = src[base_src + su];
                    else if (static_cast<unsigned>(su) < static_cast<unsigned>(Nu))
                        val = src[base_src + su];
                    dst[base_dst + u] = val;
                }
            }

            // ----------------------------------------------------------------
            // _fp_crop_kernel
            //   将 [K, Nv, paddedN] 的滤波结果裁回 [K, Nv, Nu]。
            //   每个 warp 负责一行 (i, v)；startu[i] 给出列偏移。
            // ----------------------------------------------------------------
            __global__ void _fp_crop_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int* __restrict__ startu,
                int Nu, int Nv, int paddedN, int K, int bounds_check)
            {
                cg::thread_block              tb = cg::this_thread_block();
                cg::thread_block_tile<32>     warp = cg::tiled_partition<32>(tb);

                const int wpb = static_cast<int>(tb.size() / 32);
                const int warp_global = static_cast<int>(blockIdx.x) * wpb
                    + static_cast<int>(tb.thread_rank() / 32);
                if (warp_global >= K * Nv) return;

                const int i = warp_global / Nv;
                const int v = warp_global - i * Nv;
                const int sU = startu[i];

                const size_t base_src = (static_cast<size_t>(i) * Nv + v) * paddedN;
                const size_t base_dst = (static_cast<size_t>(i) * Nv + v) * Nu;

                for (int u = static_cast<int>(warp.thread_rank()); u < Nu; u += 32) {
                    const int su = sU + u;
                    float val = 0.0f;
                    if (!bounds_check)
                        val = src[base_src + su];
                    else if (static_cast<unsigned>(su) < static_cast<unsigned>(paddedN))
                        val = src[base_src + su];
                    dst[base_dst + u] = val;
                }
            }

        }; // namespace YK::Fdk::detail
    };
};