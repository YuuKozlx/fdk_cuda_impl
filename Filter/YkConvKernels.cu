#include <cuda_runtime_api.h>
#include "../global/YkGlobals.h"
#include "YkConv.hpp"

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
                int            n_complex,
                int            batch)
            {
                int b = blockIdx.y;
                if (b >= batch) return;

                const int lane = threadIdx.x & 31;           // 等价于 % 32，位运算更快
                const int warp_id = threadIdx.x >> 5;           // 等价于 / 32
                const int n_warps = blockDim.x >> 5;           // 本 block 内 warp 总数

                // 每个 warp 负责以 32 为粒度的 stride loop
                // base  : 本 warp 当前轮次的起始下标
                // stride: 每轮跳过所有 warp 覆盖的宽度 = n_warps * 32
                for (int base = warp_id * 32; base < n_complex; base += n_warps * 32)
                {
                    int u = base + lane;                        // 本线程负责的复数下标
                    if (u < n_complex)                          // 尾部边界保护
                    {
                        int idx = b * n_complex + u;
                        float w = weights[u];
                        data[idx].x *= w;
                        data[idx].y *= w;
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
                float2* data,        // 与 cufftComplex* 等价，显式用 float2 触发向量化
                const float* weights,
                int            n_complex,
                int            batch)
            {
                int b = blockIdx.y;
                if (b >= batch) return;

                const int lane = threadIdx.x & 31;
                const int warp_id = threadIdx.x >> 5;
                const int n_warps = blockDim.x >> 5;

                for (int base = warp_id * 32; base < n_complex; base += n_warps * 32)
                {
                    int u = base + lane;
                    if (u < n_complex)
                    {
                        int    idx = b * n_complex + u;

                        float2 c = data[idx];           // 64-bit LD：一次读取整个复数
                        float  w = weights[u];

                        c.x *= w;                       // 实部
                        c.y *= w;                       // 虚部

                        data[idx] = c;                  // 64-bit ST：一次写回整个复数
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
                float4* data,        // 重解释后的指针，每元素覆盖2个复数
                const float* weights,     // 仍是原始 weights[n_complex]，按需取两个
                int            n2,          // = n_complex / 2，float4 元素总数
                int            batch)
            {
                int b = blockIdx.y;
                if (b >= batch) return;

                const int lane = threadIdx.x & 31;
                const int warp_id = threadIdx.x >> 5;
                const int n_warps = blockDim.x >> 5;

                // stride loop 与 v2/v3 相同，只是元素单位从"1个复数"变为"2个复数"
                for (int base = warp_id * 32; base < n2; base += n_warps * 32)
                {
                    int u = base + lane;                // float4 下标
                    if (u < n2)
                    {
                        int    idx = b * n2 + u;

                        float4 c = data[idx];          // 128-bit LD：一次读取 2 个复数

                        // u 对应原始复数下标 u*2 和 u*2+1，各取一个权重
                        float  w0 = weights[u * 2];     // 第一个复数的权重
                        float  w1 = weights[u * 2 + 1]; // 第二个复数的权重

                        c.x *= w0;  c.y *= w0;          // 第一个复数：实部、虚部
                        c.z *= w1;  c.w *= w1;          // 第二个复数：实部、虚部

                        data[idx] = c;                  // 128-bit ST：一次写回 2 个复数
                    }
                }
            }

        } // namespace detail

          // =============================================================================
            // 统一 launch 入口
            // 根据 n_complex 自动选择最优 kernel：
            //   偶数 → v4（float4，128-bit，每线程2复数）
            //   奇数 → v3（float2，64-bit，每线程1复数）
            // grid.x 固定为 1，不随 n_complex 增大而扩张
            // =============================================================================
        void launch_pointwise_mul(
            cufftComplex* data,
            const float* weights,
            int            n_complex,
            int            batch,
            cudaStream_t   stream)
        {
            // 256线程 = 8 warp，对 512点 数据单轮覆盖，对更大数据自动 stride
            SKernelLaunchPolicy policy;
            dim3 block(policy.block_threads, 1);
            dim3 grid(1, batch);

            if (n_complex % 2 == 0)
            {
                // float4 路径：n2 = n_complex/2 个 float4 元素
                // 注意：偏移运算必须在 reinterpret 之前完成，
                //       否则 float4* + 1 步进 16 字节而非 8 字节
                detail::_kernel_pointwise_mul_v4
                    << <grid, block, 0, stream >> > (
                        reinterpret_cast<float4*>(data),    // cufftComplex* → float4*
                        weights,
                        n_complex / 2,
                        batch);
            }
            else
            {
                // float2 fallback：n_complex 为奇数时无法对齐到 float4
                detail::_kernel_pointwise_mul_v3
                    << <grid, block, 0, stream >> > (
                        reinterpret_cast<float2*>(data),    // cufftComplex* → float2*
                        weights,
                        n_complex,
                        batch);
            }

            YK_CUDA_CHECK(cudaGetLastError());
        }

    }
}
