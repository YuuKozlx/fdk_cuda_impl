
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"
#include "../YkVecGeo.hpp"             // SConeProjGeomVec, SFDKGeoParamPerView
#include "YkFDKPreWeightHelpers.cuh"
#include "YkFDKPreWeightLaunch.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            namespace cg = cooperative_groups;

            // =============================================================================
            // preweight_vec_chunk_rowwarp_kernel
            //
            // 功能：锥束 FDK 重建的余弦预加权（cosine pre-weighting）
            //   对每个投影像素乘以权重 w = DSD / |q|
            //   其中 q = 探测器像素位置 - 射线源位置（像素到源的向量）
            //       DSD = 源到探测器的距离
            //
            // 数据布局：src / dst 均为 [K, Nv, Nu] 的线性 device buffer
            //   K  = 投影帧数（views）
            //   Nv = 探测器行数（v 方向）
            //   Nu = 探测器列数（u 方向）
            //
            // 线程分配策略：一个 warp 负责一行 [i, v, :]
            //   warp_global = i * Nv + v  → 唯一对应一行
            //   warp 内 32 个 lane 以 stride=32 遍历该行的 Nu 个像素
            //
            // 几何参数从 geo[i] 读入 shared memory，再由 lane0 广播给整个 warp
            // =============================================================================
            __global__ void preweight_vec_chunk_rowwarp_kernel(
                const float* __restrict__              src,         // 输入投影 [K, Nv, Nu]
                float* __restrict__                    dst,         // 输出投影 [K, Nv, Nu]（支持原地）
                const SConeProjGeomVec* __restrict__   geo,         // 每帧的锥束几何（源点+探测器基向量）
                const SFDKGeoParamPerView* __restrict__ gv,         // 每帧的 FDK 参数（含 SDD）
                int Nu, int Nv, int K,                              // 探测器尺寸和帧数
                int bounds_check)                                   // 非零时启用额外边界检查
            {
                // ------------------------------------------------------------------
                // Step 1: 计算本 warp 的全局编号，映射到 (i, v)
                // ------------------------------------------------------------------
                const int lane = static_cast<int>(threadIdx.x) & 31;       // warp 内偏移 [0,31]
                const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;       // block 内第几个 warp
                const int warps_per_blk = static_cast<int>(blockDim.x) >> 5;       // 每 block 的 warp 数

                // 全局 warp 编号 = block偏移 + block内偏移
                const int warp_global = static_cast<int>(blockIdx.x) * warps_per_blk + warp_in_block;

                // 总任务数 = K * Nv（每帧每行分配一个 warp）
                if (warp_global >= K * Nv) return;

                // 将全局 warp 编号解码为 (帧下标 i, 行下标 v)
                const int i = warp_global / Nv;     // 投影帧下标
                const int v = warp_global - i * Nv; // 探测器行下标（等价于 % Nv，避免二次除法）

                if (bounds_check && i >= K) return; // 可选的额外越界保护

                // ------------------------------------------------------------------
                // Step 2: lane0 将 geo[i] 写入 shared memory，__syncwarp 后全 warp 可读
                //
                // shared memory 布局（每个 warp 占 kPreweightSlot 个 float）：
                //   ws[0..2]  = src(射线源坐标)    : src.x, src.y, src.z
                //   ws[3..5]  = detS(探测器原点)   : detS.x, detS.y, detS.z
                //   ws[6..8]  = detU(u方向单位向量): detU.x, detU.y, detU.z
                //   ws[9..11] = detV(v方向单位向量): detV.x, detV.y, detV.z
                //   ws[12]    = SDD（源到探测器距离）
                // ------------------------------------------------------------------
                extern __shared__ float smem[];

                // 每个 warp 使用 smem 中独立的一段，避免 warp 间踩踏
                float* ws = smem + warp_in_block * kPreweightSlot;

                if (lane == 0)
                {
                    const SConeProjGeomVec& g = geo[i];
                    const SFDKGeoParamPerView& gp = gv[i];

                    ws[0] = g.src.x;    ws[1] = g.src.y;    ws[2] = g.src.z;
                    ws[3] = g.detS.x;   ws[4] = g.detS.y;   ws[5] = g.detS.z;
                    ws[6] = g.detU.x;   ws[7] = g.detU.y;   ws[8] = g.detU.z;
                    ws[9] = g.detV.x;   ws[10] = g.detV.y;   ws[11] = g.detV.z;
                    ws[12] = gp.SDD_mm;
                }
                // lane0 写完后同步，确保 warp 内其他 lane 读到有效值
                // 注意：这里用 __syncwarp() 而非 __syncthreads()，
                //       因为只需要 warp 内同步，开销更小
                __syncwarp();

                // ------------------------------------------------------------------
                // Step 3: 所有 lane 从 shared memory 读取几何参数（寄存器广播）
                // ------------------------------------------------------------------
                const float src_x = ws[0], src_y = ws[1], src_z = ws[2];
                const float dS_x = ws[3], dS_y = ws[4], dS_z = ws[5];   // 探测器原点
                const float dU_x = ws[6], dU_y = ws[7], dU_z = ws[8];   // u 步进向量
                const float dV_x = ws[9], dV_y = ws[10], dV_z = ws[11];  // v 步进向量
                const float DSD = ws[12];

                // ------------------------------------------------------------------
                // Step 4: 预计算 v 行的基础向量 qs0 = detS + detV*v - src
                //
                // 探测器像素 (u,v) 的世界坐标：
                //   P(u,v) = detS + detU*u + detV*v
                //
                // 像素到源的向量：
                //   q(u,v) = P(u,v) - src
                //          = (detS - src) + detU*u + detV*v
                //          = qs0 + detU*u         ← qs0 只与 v 有关，提到循环外
                // ------------------------------------------------------------------
                const float fv = static_cast<float>(v);
                const float qs0_x = (dS_x + dV_x * fv) - src_x;   // q 在 u=0 时的 x 分量
                const float qs0_y = (dS_y + dV_y * fv) - src_y;   // q 在 u=0 时的 y 分量
                const float qs0_z = (dS_z + dV_z * fv) - src_z;   // q 在 u=0 时的 z 分量

                // ------------------------------------------------------------------
                // Step 5: warp stride loop 遍历当前行的所有 u，计算并写出加权结果
                //
                // 访存模式：lane 0,1,...,31 读取 u=0,1,...,31（连续），满足 coalesced access
                //           下一轮步进 32，继续保持对齐
                // ------------------------------------------------------------------
                const size_t base = (static_cast<size_t>(i) * Nv
                    + static_cast<size_t>(v)) * Nu;  // 当前行首地址偏移
                const float* src_row = src + base;
                float* dst_row = dst + base;

                for (int u = lane; u < Nu; u += 32)
                {
                    const float fu = static_cast<float>(u);

                    // 像素 (u,v) 到源的向量 q = qs0 + detU * u
                    const float qsx = qs0_x + dU_x * fu;
                    const float qsy = qs0_y + dU_y * fu;
                    const float qsz = qs0_z + dU_z * fu;

                    // |q|^2，fmaxf 防止除零（极端情况源点与像素重合）
                    const float r2 = qsx * qsx + qsy * qsy + qsz * qsz;

                    // 余弦权重 w = DSD / |q|
                    // rsqrtf(x) = 1/sqrt(x)，单指令，比 sqrtf+除法 快
                    const float w = DSD * rsqrtf(fmaxf(r2, 1e-20f));

                    dst_row[u] = src_row[u] * w;
                }
            }


            // =============================================================================
            // preweight_vec_chunk_rowwarp_shfl_kernel  (v2 - __shfl_sync 广播版)
            //
            // 与 v1 的唯一区别：用 __shfl_sync 替代 shared memory 广播
            //
            // 原理：
            //   lane0 从 global memory 读取几何参数，存入寄存器
            //   __shfl_sync(mask, var, 0) 将 lane0 的寄存器值广播给 warp 内所有 lane
            //   整个过程不经过 shared memory，也不需要 __syncwarp
            //
            // 优点：
            //   - 不占用 shared memory，可提升 occupancy
            //   - 省去 smem 读写延迟（约 20~30 cycle → ~4 cycle）
            //   - launch 时 smem_bytes = 0
            //
            // 缺点：
            //   - 参数较多时需要逐个 shfl，代码稍显繁琐
            //   - 仅适用于 warp 内广播，无法跨 warp
            // =============================================================================
            __global__ void preweight_vec_chunk_rowwarp_shfl_kernel(
                const float* __restrict__               src,
                float* __restrict__                     dst,
                const SConeProjGeomVec* __restrict__    geo,
                const SFDKGeoParamPerView* __restrict__ gv,
                int Nu, int Nv, int K,
                int bounds_check)
            {
                const int lane = static_cast<int>(threadIdx.x) & 31;
                const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
                const int warps_per_blk = static_cast<int>(blockDim.x) >> 5;
                const int warp_global = static_cast<int>(blockIdx.x) * warps_per_blk + warp_in_block;

                if (warp_global >= K * Nv) return;

                const int i = warp_global / Nv;
                const int v = warp_global - i * Nv;

                if (bounds_check && i >= K) return;

                // ------------------------------------------------------------------
                // __shfl_sync 广播几何参数
                //
                // 步骤：
                //   1. lane0 读取 geo[i] 到本地寄存器
                //   2. 每个参数调用一次 __shfl_sync(..., srcLane=0)
                //      → warp 内所有 lane 得到与 lane0 相同的值
                //   3. 无需 smem，无需 __syncwarp
                //
                // __shfl_sync 签名：
                //   T __shfl_sync(unsigned mask, T var, int srcLane, int width=32)
                //   mask = 0xFFFFFFFF 表示 warp 内 32 个 lane 全部参与
                // ------------------------------------------------------------------
                constexpr unsigned FULL_MASK = 0xFFFFFFFF;

                // lane0 读取，其余 lane 的 g_* 初始值无意义（马上被 shfl 覆盖）
                float g_src_x, g_src_y, g_src_z;
                float g_dS_x, g_dS_y, g_dS_z;
                float g_dU_x, g_dU_y, g_dU_z;
                float g_dV_x, g_dV_y, g_dV_z;
                float g_DSD;

                if (lane == 0)
                {
                    const SConeProjGeomVec& g = geo[i];
                    const SFDKGeoParamPerView& gp = gv[i];
                    g_src_x = g.src.x;   g_src_y = g.src.y;   g_src_z = g.src.z;
                    g_dS_x = g.detS.x;  g_dS_y = g.detS.y;  g_dS_z = g.detS.z;
                    g_dU_x = g.detU.x;  g_dU_y = g.detU.y;  g_dU_z = g.detU.z;
                    g_dV_x = g.detV.x;  g_dV_y = g.detV.y;  g_dV_z = g.detV.z;
                    g_DSD = gp.SDD_mm;
                }

                // 从 lane0 广播到 warp 内所有 lane，每条指令约 4 cycle
                // 广播后每个 lane 的寄存器都持有相同的几何参数
                const float src_x = __shfl_sync(FULL_MASK, g_src_x, 0);
                const float src_y = __shfl_sync(FULL_MASK, g_src_y, 0);
                const float src_z = __shfl_sync(FULL_MASK, g_src_z, 0);
                const float dS_x = __shfl_sync(FULL_MASK, g_dS_x, 0);
                const float dS_y = __shfl_sync(FULL_MASK, g_dS_y, 0);
                const float dS_z = __shfl_sync(FULL_MASK, g_dS_z, 0);
                const float dU_x = __shfl_sync(FULL_MASK, g_dU_x, 0);
                const float dU_y = __shfl_sync(FULL_MASK, g_dU_y, 0);
                const float dU_z = __shfl_sync(FULL_MASK, g_dU_z, 0);
                const float dV_x = __shfl_sync(FULL_MASK, g_dV_x, 0);
                const float dV_y = __shfl_sync(FULL_MASK, g_dV_y, 0);
                const float dV_z = __shfl_sync(FULL_MASK, g_dV_z, 0);
                const float DSD = __shfl_sync(FULL_MASK, g_DSD, 0);

                // ------------------------------------------------------------------
                // 以下与 v1 完全相同
                // ------------------------------------------------------------------
                const float fv = static_cast<float>(v);
                const float qs0_x = (dS_x + dV_x * fv) - src_x;
                const float qs0_y = (dS_y + dV_y * fv) - src_y;
                const float qs0_z = (dS_z + dV_z * fv) - src_z;

                const size_t base = (static_cast<size_t>(i) * Nv + static_cast<size_t>(v)) * Nu;
                const float* src_row = src + base;
                float* dst_row = dst + base;

                for (int u = lane; u < Nu; u += 32)
                {
                    const float fu = static_cast<float>(u);
                    const float qsx = qs0_x + dU_x * fu;
                    const float qsy = qs0_y + dU_y * fu;
                    const float qsz = qs0_z + dU_z * fu;
                    const float r2 = qsx * qsx + qsy * qsy + qsz * qsz;
                    const float w = DSD * rsqrtf(fmaxf(r2, 1e-20f));
                    dst_row[u] = src_row[u] * w;
                }
            }




            // =============================================================================
            // pw_launchPreweight
            //
            // 功能：配置并启动 preweight_vec_chunk_rowwarp_kernel
            //
            // Grid / Block 计算：
            //   总任务数  = K * Nv（每帧每行一个 warp）
            //   每 block  = blockThreads / 32 个 warp
            //   block 数  = ceil(K*Nv / warps_per_blk)
            //
            // Shared memory：
            //   每 block 分配 warps_per_blk * kPreweightSlot * sizeof(float) 字节
            //   各 warp 各用一段，互不干扰
            // =============================================================================
            void pw_launchPreweight(
                const float* d_src,     // 输入投影 device 指针 [K, Nv, Nu]
                float* d_dst,     // 输出投影 device 指针 [K, Nv, Nu]
                const SConeProjGeomVec* d_geo,     // 每帧锥束几何 device 指针 [K]
                const SFDKGeoParamPerView* d_gv,      // 每帧 FDK 参数 device 指针 [K]
                int Nu, int Nv, int K,
                const SKernelLaunchPolicy& policy,    // 包含 block_threads / bounds_check 等策略
                cudaStream_t               stream)
            {
                const int warps_per_blk = policy.block_threads / 32;
                const int total_warps = K * Nv;                                   // 总 warp 任务数

                // 向上取整：确保所有 warp 任务都有对应的 block
                const int blocks = (total_warps + warps_per_blk - 1) / warps_per_blk;

                // 动态 shared memory 大小：每个 warp 需要 kPreweightSlot 个 float
                const size_t smem_bytes = static_cast<size_t>(warps_per_blk)
                    * kPreweightSlot * sizeof(float);

                preweight_vec_chunk_rowwarp_shfl_kernel << <blocks, policy.block_threads, smem_bytes, stream >> > (
                    d_src, d_dst, d_geo, d_gv,
                    Nu, Nv, K,
                    policy.bounds_check ? 1 : 0);

                YK_CUDA_KERNEL_CHECK();
            }
        }
    }
} // namespace YK::Fdk::detail
