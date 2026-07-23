// =============================================================================
// YkWarpStrideCtx.cuh
//
// Warp Stride Loop 上下文模板
//
// 用途：统一管理 CUDA kernel 中的 warp 身份计算，避免重复手写
//       根据数据维度选择对应特化，编译期零开销
// =============================================================================
#pragma once

#include <cassert>
#include <cuda_runtime.h>

namespace YK {

    // -----------------------------------------------------------------------------
    // 方向枚举
    //
    //   X       : 1D 连续数组，数据在 x 方向
    //             适用：滤波权重填充、scale、window 等 1D kernel
    //
    //   XY      : 2D，batch 在 blockIdx.y，数据在 x 方向 stride loop
    //             适用：pointwise_mul 等有 batch 维度的 kernel
    //
    //   RowWarp : 行级，每个 warp 独占一行，外层 stride loop 跨行
    //             适用：preweight 等 per-row 计算 kernel
    // -----------------------------------------------------------------------------
    enum class EWarpStrideAxis { X, XY, RowWarp };


    // 主模板（未特化时不可用）
    template<EWarpStrideAxis Axis>
    struct WarpStrideCtx;


    // -----------------------------------------------------------------------------
    // 特化：X — 1D kernel
    //
    // 【Launch 范式 — 小数据（n <= 2048，一次性初始化）】
    //   // 单 block 足够，stride loop 自动多轮覆盖
    //   dim3 block(256, 1, 1);
    //   dim3 grid(1, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n);
    //
    //   适用场景：
    //     滤波权重初始化（n = 512~2048）
    //     非热路径，启动开销比计算本身更显著
    //     每线程跑 n/256 轮，单 block 串行覆盖
    //
    // 【Launch 范式 — 大数据（n > 2048，热路径）】
    //   // 按 SM 数量固定 grid，避免过度启动
    //   static int sm_count = 0;
    //   if (sm_count == 0)
    //       cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    //
    //   const int warps_per_blk = 256 / 32;                               // = 8
    //   const int warps_need    = (n + 31) / 32;
    //   const int blocks_need   = (warps_need + warps_per_blk - 1) / warps_per_blk;
    //   const int grid_x        = min(blocks_need, sm_count * 2);         // 上限
    //   dim3 block(256, 1, 1);
    //   dim3 grid(grid_x, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n);
    //
    //   适用场景：
    //     大规模逐元素操作（n = 几十万以上）
    //     热路径，需要充分利用所有 SM
    //     每线程跑 ceil(n / (grid_x*256)) 轮
    //
    // 【Kernel 范式】
    //   __global__ void kernel(..., int n)
    //   {
    //       WarpStrideCtx<EWarpStrideAxis::X> ctx;
    //
    //       for (int base = ctx.warp_global * 32; base < n; base += ctx.n_warps * 32)
    //       {
    //           int k = base + ctx.lane;
    //           if (k >= n) break;        // 尾部边界保护，跳出本轮
    //           // ... 计算 ...
    //       }
    //   }
    //
    // 【边界说明】
    //   - 循环条件 base < n     : 整轮越界时不进入循环体
    //   - if (k >= n) break     : 尾部 warp 内部分 lane 越界时跳出
    //   - 不需要顶层 return 保护: stride loop 天然处理 warp 数超过数据量的情况
    // -----------------------------------------------------------------------------
    template<>
    struct WarpStrideCtx<EWarpStrideAxis::X>
    {
        int lane;
        int warp_global;
        int n_warps;

        __device__ __forceinline__ WarpStrideCtx()
        {
#ifndef NDEBUG
            assert(blockDim.x >= 32);
            assert(blockDim.x <= 1024);
            assert((blockDim.x & 31) == 0);
            assert(blockDim.y == 1 && blockDim.z == 1);
#endif
            lane = static_cast<int>(threadIdx.x) & 31;
            int warp_in = static_cast<int>(threadIdx.x) >> 5;
            int nw_blk = (static_cast<int>(blockDim.x) + 31) >> 5;
            warp_global = static_cast<int>(blockIdx.x) * nw_blk + warp_in;
            n_warps = static_cast<int>(gridDim.x) * nw_blk;
        }
    };


    // -----------------------------------------------------------------------------
    // 特化：XY — 2D batch kernel
    //
    // 【Launch 范式 — 小数据（n_complex <= 2048，batch 较小）】
    //   // x 方向单 block，y 方向对应 batch
    //   dim3 block(256, 1, 1);
    //   dim3 grid(1, batch, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n, batch);
    //
    //   适用场景：
    //     n_complex = 512~2048，batch = 几十到几百
    //     每线程在 x 方向跑 n/256 轮，y 方向 1:1 对应 batch
    //     batch 较小时 gridDim.y 不会超出硬件限制（65535）
    //
    // 【Launch 范式 — 大数据（n_complex > 2048 或 batch 极大）】
    //   static int sm_count = 0;
    //   if (sm_count == 0)
    //       cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    //
    //   const int warps_per_blk = 256 / 32;
    //   const int warps_need    = (n + 31) / 32;
    //   const int blocks_need   = (warps_need + warps_per_blk - 1) / warps_per_blk;
    //   const int grid_x        = min(blocks_need, sm_count * 2);         // x 上限
    //   const int grid_y        = min(batch, 65535);                      // y 上限
    //   dim3 block(256, 1, 1);
    //   dim3 grid(grid_x, grid_y, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n, batch);
    //
    //   batch 超出 gridDim.y 时，kernel 必须使用 b += gridDim.y 的 stride loop。
    //
    // 【Kernel 范式】
    //   __global__ void kernel(..., int n, int batch)
    //   {
    //       WarpStrideCtx<EWarpStrideAxis::XY> ctx;
    //       for (int b = ctx.b; b < batch; b += ctx.b_stride)
    //       for (int base = ctx.warp_global * 32; base < n; base += ctx.n_warps * 32)
    //       {
    //           int u = base + ctx.lane;
    //           if (u >= n) break;         // x 方向尾部越界
    //           int idx = ctx.b * n + u;
    //           // ... 计算 ...
    //       }
    //   }
    //
    // 【边界说明】
    //   - if (ctx.b >= batch) return : y 方向 gridDim.y 超出 batch 时，
    //                                  整个线程块无任务，直接退出
    //   - if (u >= n) break          : x 方向尾部 lane 越界，跳出循环
    //   - y 方向用 return，x 方向用 break
    //     原因：y 越界意味着整个线程无任何任务
    //           x 越界只是当前轮尾部，后续轮次可能仍有数据（stride loop）
    // -----------------------------------------------------------------------------
    template<>
    struct WarpStrideCtx<EWarpStrideAxis::XY>
    {
        int b;
        int b_stride;
        int lane;
        int warp_global;
        int n_warps;

        __device__ __forceinline__ WarpStrideCtx()
        {
#ifndef NDEBUG
            assert(blockDim.x >= 32);
            assert(blockDim.x <= 1024);
            assert((blockDim.x & 31) == 0);
            assert(blockDim.y == 1 && blockDim.z == 1);
#endif
            b = static_cast<int>(blockIdx.y);
            b_stride = static_cast<int>(gridDim.y);
            lane = static_cast<int>(threadIdx.x) & 31;
            int warp_in = static_cast<int>(threadIdx.x) >> 5;
            int nw_blk = (static_cast<int>(blockDim.x) + 31) >> 5;
            warp_global = static_cast<int>(blockIdx.x) * nw_blk + warp_in;
            n_warps = static_cast<int>(gridDim.x) * nw_blk;
        }
    };


    // -----------------------------------------------------------------------------
    // 特化：RowWarp — 行级 per-row kernel
    //
    // 【Launch 范式 — 小数据（K*Nv 较小，行数 <= sm_count*warps_per_blk）】
    //   // 精确分配，每行恰好一个 warp，无 stride
    //   const int total_rows    = K * Nv;
    //   const int warps_per_blk = 256 / 32;                               // = 8
    //   const int blocks        = (total_rows + warps_per_blk - 1) / warps_per_blk;
    //   dim3 block(256, 1, 1);
    //   dim3 grid(blocks, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(...);
    //
    //   适用场景：
    //     K*Nv 较小，block 数不会过多
    //     每个 warp 只跑一行，外层 stride loop 只执行一轮
    //     Nu 较大时行内 lane loop 仍有足够工作量
    //
    // 【Launch 范式 — 大数据（K*Nv 很大，需要限制 block 数）】
    //   static int sm_count = 0;
    //   if (sm_count == 0)
    //       cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    //
    //   const int total_rows    = K * Nv;
    //   const int warps_per_blk = 256 / 32;
    //   const int blocks_need   = (total_rows + warps_per_blk - 1) / warps_per_blk;
    //   const int blocks        = min(blocks_need, sm_count * 2);         // 上限
    //   dim3 block(256, 1, 1);
    //   dim3 grid(blocks, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(...);
    //
    //   适用场景：
    //     K = 几百帧，Nv = 几百行，K*Nv = 几万到几十万
    //     加上限后 block 数固定，stride loop 分摊剩余行
    //     geo[i] 反复读取但 K 较小，大概率命中 L1 cache
    //
    // 【Kernel 范式】
    //   __global__ void kernel(..., int Nu, int Nv, int K)
    //   {
    //       WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;
    //
    //       // 外层：跨行 stride loop，一个 warp 处理多行
    //       for (int row = ctx.warp_global; row < K * Nv; row += ctx.n_warps)
    //       {
    //           int i = row / Nv;
    //           int v = row - i * Nv;
    //
    //           if (i >= K) continue;      // 防御性检查，跳过异常行，继续后续行
    //
    //           // 每行重新加载行级参数（如 geo[i]）...
    //
    //           // 内层：行内 lane loop，步进固定 32
    //           for (int u = ctx.lane; u < Nu; u += 32)
    //           {
    //               // ... 计算 ...
    //           }                          // 行内无需显式越界检查，循环条件保证
    //       }
    //   }
    //
    // 【边界说明】
    //   - 外层循环条件 row < K*Nv : 行编号越界时不进入循环，自然退出
    //   - if (i >= K) continue    : 防御性保护，异常行跳过而非 return
    //                               原因：后续行仍需处理，return 会导致漏处理
    //   - 内层循环条件 u < Nu     : 行内天然边界，无需额外 break
    //   - 不在顶层做 return 保护  : 与 X/XY 不同，顶层 return 会导致
    //                               后续行漏处理
    //
    // 【小数据 vs 大数据的核心区别】
    //   小数据：blocks 精确覆盖，每 warp 只跑一行，stride loop 退化为单轮
    //   大数据：blocks 加上限，每 warp 跑多行，stride loop 分摊剩余
    //   两种情况 kernel 代码完全相同，只有 launch 侧的 blocks 计算不同
    //
    // 【与 X/XY 的关键区别】
    //   X/XY    : 元素级 stride，warp 处理多个独立元素，越界用 break
    //   RowWarp : 行级 stride，warp 处理多行，行内协作，越界用 continue
    // -----------------------------------------------------------------------------
    template<>
    struct WarpStrideCtx<EWarpStrideAxis::RowWarp>
    {
        int lane;
        int warp_global;
        int n_warps;

        __device__ __forceinline__ WarpStrideCtx()
        {
#ifndef NDEBUG
            assert(blockDim.x >= 32);
            assert(blockDim.x <= 1024);
            assert((blockDim.x & 31) == 0);
            assert(blockDim.y == 1 && blockDim.z == 1);
#endif
            lane = static_cast<int>(threadIdx.x) & 31;
            int warp_in = static_cast<int>(threadIdx.x) >> 5;
            int nw_blk = (static_cast<int>(blockDim.x) + 31) >> 5;
            warp_global = static_cast<int>(blockIdx.x) * nw_blk + warp_in;
            n_warps = static_cast<int>(gridDim.x) * nw_blk;
        }
    };

} // namespace YK
