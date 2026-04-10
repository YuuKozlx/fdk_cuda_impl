
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

            // ----------------------------------------------------------------
            // preweight_vec_chunk_rowwarp_kernel
            //
            //   每个 warp 处理投影 [i, v, :] 的一行。
            //   几何量从 geo[i] 读入 shared memory，再广播给 warp 内各 lane。
            //   输出 dst[i,v,u] = src[i,v,u] * (DSD / |q|)，其中 q 为像素到源的向量。
            // ----------------------------------------------------------------
            __global__ void preweight_vec_chunk_rowwarp_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const SConeProjGeomVec* __restrict__ geo,
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

                extern __shared__ float smem[];
                float* ws = smem + warp_in_block * kPreweightSlot;

                if (lane == 0) {
                    const SConeProjGeomVec& g = geo[i];
                    ws[0] = g.src.x;   ws[1] = g.src.y;   ws[2] = g.src.z;
                    ws[3] = g.detS.x;  ws[4] = g.detS.y;  ws[5] = g.detS.z;
                    ws[6] = g.detU.x;  ws[7] = g.detU.y;  ws[8] = g.detU.z;
                    ws[9] = g.detV.x;  ws[10] = g.detV.y;  ws[11] = g.detV.z;
                    ws[12] = gv[i].SDD_mm;
                }
                __syncwarp();

                const float src_x = ws[0], src_y = ws[1], src_z = ws[2];
                const float dS_x = ws[3], dS_y = ws[4], dS_z = ws[5];
                const float dU_x = ws[6], dU_y = ws[7], dU_z = ws[8];
                const float dV_x = ws[9], dV_y = ws[10], dV_z = ws[11];
                const float DSD = ws[12];

                const float fv = static_cast<float>(v);
                const float qs0_x = (dS_x + dV_x * fv) - src_x;
                const float qs0_y = (dS_y + dV_y * fv) - src_y;
                const float qs0_z = (dS_z + dV_z * fv) - src_z;

                const size_t base = (static_cast<size_t>(i) * Nv + static_cast<size_t>(v)) * Nu;
                const float* src_row = src + base;
                float* dst_row = dst + base;

                for (int u = lane; u < Nu; u += 32) {
                    const float fu = static_cast<float>(u);
                    const float qsx = qs0_x + dU_x * fu;
                    const float qsy = qs0_y + dU_y * fu;
                    const float qsz = qs0_z + dU_z * fu;
                    const float r2 = qsx * qsx + qsy * qsy + qsz * qsz;
                    const float w = DSD * rsqrtf(fmaxf(r2, 1e-20f));
                    dst_row[u] = src_row[u] * w;
                }
            }



            // ----------------------------------------------------------------
            // pw_launchPreweight
            //   启动锥束余弦预加权 kernel。
            //   d_src / d_dst 均为 [K, Nv, Nu] device buffer（支持原地）。
            // ----------------------------------------------------------------
            void pw_launchPreweight(
                const float* d_src,
                float* d_dst,
                const SConeProjGeomVec* d_geo,
                const SFDKGeoParamPerView* d_gv,
                int Nu, int Nv, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream)
            {
                const int blockThreads = policy.block_threads;
                const int warps_per_blk = blockThreads / 32;
                const int total_warps = K * Nv;
                const int blocks = (total_warps + warps_per_blk - 1) / warps_per_blk;
                const size_t smem_bytes = static_cast<size_t>(warps_per_blk)
                    * kPreweightSlot * sizeof(float);

                preweight_vec_chunk_rowwarp_kernel << <blocks, blockThreads, smem_bytes, stream >> > (
                    d_src, d_dst, d_geo, d_gv,
                    Nu, Nv, K,
                    policy.bounds_check ? 1 : 0);
                YK_CUDA_KERNEL_CHECK();
            }
        }
    }
} // namespace YK::Fdk::detail
