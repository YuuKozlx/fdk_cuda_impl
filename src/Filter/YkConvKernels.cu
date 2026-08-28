#include <cuda_runtime_api.h>
#include "../global/YkGlobals.h"
#include "YkConv.hpp"
#include <algorithm>
#include "global/YkMacro.hpp"
#include "global/YkKernelLaunchPolicy.hpp"

namespace YK {
    namespace Filter {
        namespace detail {
            // =============================================================================
            // pointwise_mul kernels
            // 功能：对 batch 组复数数据的每个点乘以对应的实数权重
            //       data[b][u] *= weights[u]   (u = 0 .. n_complex-1, b = 0 .. batch-1)
            //
            // 提供四个版本，性能依次提升：
            //   v1 - 基础版       : 1线程处理1个复数，直接映射
            //   v2 - warp stride  : warp对齐访存，stride loop覆盖任意尺寸
            //   v3 - float2 向量化: 单次64-bit LD/ST，减少内存事务
            //   v4 - float4 向量化: 单次128-bit LD/ST，每线程处理2个复数
            // =============================================================================


                // -----------------------------------------------------------------------------
                // v1: 基础版
                // 线程映射：blockIdx.x * blockDim.x + threadIdx.x → 直接对应 u（复数下标）
                //           blockIdx.y                              → 对应 batch 维度 b
                // 缺点：n_complex 较大时需要多个 block，grid 随数据尺寸变化
                // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v1(
                cufftComplex* data,        // [batch, n_complex] 复数数组（in/out）
                const float* weights,     // [n_complex]        实数权重（只读）
                int            n_complex,   // 每条数据的复数点数
                int            batch)       // 数据条数
            {
                int u = blockIdx.x * blockDim.x + threadIdx.x;  // 复数下标
                int b = blockIdx.y;                              // batch 下标

                if (u < n_complex && b < batch)
                {
                    int idx = b * n_complex + u;    // 展平后的线性下标
                    float w = weights[u];
                    data[idx].x *= w;               // 实部
                    data[idx].y *= w;               // 虚部
                }
            }


#define WARP_STRIDE_INIT()                                                   \
    const int lane          = threadIdx.x & 31;                              \
    const int warp_in_blk   = threadIdx.x >> 5;                              \
    const int warps_per_blk = (blockDim.x + 31) >> 5;                        \
    const int warp_global   = blockIdx.x * warps_per_blk + warp_in_blk;      \
    const int n_warps       = gridDim.x  * warps_per_blk;
#undef WARP_STRIDE_INIT



#define WARP_STRIDE_INIT_BATCH()                                              \
    const int lane          = threadIdx.x & 31;                               \
    const int warp_in_blk   = threadIdx.x >> 5;                               \
    const int warps_per_blk = (blockDim.x + 31) >> 5;                         \
    const int warp_global   = blockIdx.x * warps_per_blk + warp_in_blk;       \
    const int n_warps       = gridDim.x  * warps_per_blk;
            // -----------------------------------------------------------------------------
            // v2: warp stride 版
            // 核心改动：以 warp（32线程）为单位做 stride loop
            //   - warp 内 32 个线程处理连续的 32 个复数 → 保证 coalesced access
            //   - stride = n_warps * 32，每轮步进一个 block 的覆盖宽度
            //   - grid.x 固定为 1，不随 n_complex 变化
            //
            // 线程角色：
            //   lane    = threadIdx.x % 32   → warp 内偏移（0~31）
            //   warp_id = threadIdx.x / 32   → 本 block 内第几个 warp
            // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v2(
                cufftComplex* data,
                const float* weights,
                int           n_complex,
                int           batch)
            {
                WARP_STRIDE_INIT_BATCH()

                    for (int b = blockIdx.y; b < batch; b += gridDim.y)
                    {
                        for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                        {
                            int u = base + lane;
                            if (u < n_complex)
                            {
                                int   idx = b * n_complex + u;
                                float w = weights[u];
                                data[idx].x *= w;
                                data[idx].y *= w;
                            }
                        }
                    }
            }

            // -----------------------------------------------------------------------------
            // v3: float2 向量化版
            // cufftComplex 本质上就是 float2（x=实部, y=虚部），
            // 将指针重解释为 float2* 后，编译器可生成单条 64-bit 指令：
            //   LDG.E.64（一次读 8 字节） / STG.E.64（一次写 8 字节）
            // 相比 v2 的两次 32-bit 访问，内存事务减少 50%
            //
            // 其余 stride loop 逻辑与 v2 完全相同
            // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v3(
                float2* data,
                const float* weights,
                int          n_complex,
                int          batch)
            {
                WARP_STRIDE_INIT_BATCH()

                    for (int b = blockIdx.y; b < batch; b += gridDim.y)
                    {
                        for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                        {
                            int u = base + lane;
                            if (u < n_complex)
                            {
                                int    idx = b * n_complex + u;
                                float2 c = data[idx];
                                float  w = weights[u];
                                c.x *= w;
                                c.y *= w;
                                data[idx] = c;
                            }
                        }
                    }
            }



            // -----------------------------------------------------------------------------
            // v4: float4 向量化版
            // float4 = 16字节 = 2个 cufftComplex
            // 将 data 重解释为 float4* 后，单次 128-bit LD/ST 覆盖两个复数：
            //   float4.x, .y → 第一个复数（实部, 虚部）
            //   float4.z, .w → 第二个复数（实部, 虚部）
            //
            // 因此所有下标都以 n2 = n_complex/2 为基准，n_complex 必须为偶数
            //
            // 权重索引：
            //   u 对应 float4 下标，覆盖原始复数 [u*2, u*2+1]
            //   → weights[u*2]   用于 .x .y
            //   → weights[u*2+1] 用于 .z .w
            // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v4(
                float4* data,
                const float* weights,
                int          n2,
                int          batch)
            {
                WARP_STRIDE_INIT_BATCH()

                    for (int b = blockIdx.y; b < batch; b += gridDim.y)
                    {
                        for (int base = warp_global * 32; base < n2; base += n_warps * 32)
                        {
                            int u = base + lane;
                            if (u < n2)
                            {
                                int    idx = b * n2 + u;
                                float4 c = data[idx];
                                float  w0 = weights[u * 2];
                                float  w1 = weights[u * 2 + 1];
                                c.x *= w0; c.y *= w0;
                                c.z *= w1; c.w *= w1;
                                data[idx] = c;
                            }
                        }
                    }
            }

#undef WARP_STRIDE_INIT_BATCH

        } // namespace detail

        void launch_pointwise_mul(
            cufftComplex* data,
            const float* weights,
            int            n_complex,
            int            batch,
            cudaStream_t   stream)
        {
            // Empty work is a valid no-op; CUDA rejects a zero-dimensional
            // grid, so return before computing the launch geometry.
            if (n_complex <= 0 || batch <= 0)
                return;

            SKernelLaunchPolicy policy;
            const int block_threads = policy.normalizedBlockThreads();
            dim3 block(block_threads, 1);

            // fft点数若为 2 的倍数 则调用f4版，否则调用f2版
            const int n_elem = (n_complex % 2 == 0) ? n_complex / 2 : n_complex;
            const int warps_per_blk = block_threads / 32;
            const int warps_need = (n_elem + 31) / 32;
            const int grid_x = std::min((warps_need + warps_per_blk - 1) / warps_per_blk, 8);
            const int grid_y = std::min(batch, 65535);
            dim3 grid(grid_x, grid_y);

            YK::validateWarpLaunch(block, grid);

            if (n_complex % 2 == 0)
                detail::_kernel_pointwise_mul_v4 << <grid, block, 0, stream >> > (
                    reinterpret_cast<float4*>(data), weights, n_complex / 2, batch);
            else
                detail::_kernel_pointwise_mul_v3 << <grid, block, 0, stream >> > (
                    reinterpret_cast<float2*>(data), weights, n_complex, batch);

            YK_CUDA_KERNEL_CHECK();

        }

    }
}
