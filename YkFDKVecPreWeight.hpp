#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"
#include "YkFDKVecPreWeight2.hpp"

namespace YK {
    namespace cg = cooperative_groups;

    //inline SKernelLaunchPolicy normalizePreweightPolicy(SKernelLaunchPolicy p) {
    //    if (p.block_threads < 32) p.block_threads = 32;
    //    p.block_threads = (p.block_threads + 31) & ~31;
    //    p.block_threads = std::min(p.block_threads, 1024);
    //    return p;
    //}

    //// smem slot 布局（16 float = 64 字节，无 bank conflict）
    //// [0..2]  src.xyz
    //// [3..5]  detS.xyz
    //// [6..8]  detU.xyz
    //// [9..11] detV.xyz
    //// [12]    DSD
    //// [13..15] padding
    //constexpr int kValidFloats = 13;
    //constexpr int kPreweightSlot = (kValidFloats + 15) & ~15;  // = 16

    //__global__ void preweight_vec_chunk_rowwarp_kernel(
    //    const float* __restrict__ src,
    //    float* __restrict__ dst,
    //    const SConeProjectionVec* __restrict__ geo,  // 已偏移
    //    const SFDKGeoParamPerView* __restrict__ gv,   // 已偏移
    //    int Nu, int Nv, int K,
    //    int bounds_check)
    //{
    //    const int lane = (int)threadIdx.x & 31;
    //    const int warp_in_block = (int)threadIdx.x >> 5;
    //    const int warps_per_blk = (int)(blockDim.x >> 5);
    //    const int warp_global = (int)blockIdx.x * warps_per_blk + warp_in_block;

    //    const int total_warps = K * Nv;
    //    if (warp_global >= total_warps) return;

    //    const int i = warp_global / Nv;
    //    const int v = warp_global - i * Nv;

    //    if (bounds_check && i >= K) return;

    //    // ── smem：每 warp 一个 slot ──────────────────────────────────────
    //    extern __shared__ float smem[];
    //    float* ws = smem + warp_in_block * kPreweightSlot;

    //    if (lane == 0) {
    //        const SConeProjectionVec& g = geo[i];
    //        ws[0] = g.src.x;   ws[1] = g.src.y;   ws[2] = g.src.z;
    //        ws[3] = g.detS.x;  ws[4] = g.detS.y;  ws[5] = g.detS.z;
    //        ws[6] = g.detU.x;  ws[7] = g.detU.y;  ws[8] = g.detU.z;
    //        ws[9] = g.detV.x;  ws[10] = g.detV.y;  ws[11] = g.detV.z;
    //        ws[12] = gv[i].SDD_mm;
    //    }
    //    __syncwarp();

    //    // ── 从 smem 读出标量 ─────────────────────────────────────────────
    //    const float src_x = ws[0], src_y = ws[1], src_z = ws[2];
    //    const float dS_x = ws[3], dS_y = ws[4], dS_z = ws[5];
    //    const float dU_x = ws[6], dU_y = ws[7], dU_z = ws[8];
    //    const float dV_x = ws[9], dV_y = ws[10], dV_z = ws[11];
    //    const float DSD = ws[12];

    //    // ── hoist v 相关常量出 u-loop ────────────────────────────────────
    //    const float fv = (float)v;
    //    const float qs0_x = (dS_x + dV_x * fv) - src_x;
    //    const float qs0_y = (dS_y + dV_y * fv) - src_y;
    //    const float qs0_z = (dS_z + dV_z * fv) - src_z;

    //    const float inv_dsd = (DSD > 0.0f) ? DSD : 0.0f;

    //    const size_t base = ((size_t)i * Nv + (size_t)v) * Nu;
    //    const float* __restrict__ src_row = src + base;
    //    float* __restrict__ dst_row = dst + base;

    //    // ── 主循环 ───────────────────────────────────────────────────────
    //    for (int u = lane; u < Nu; u += 32) {
    //        const float fu = (float)u;
    //        const float qsx = qs0_x + dU_x * fu;
    //        const float qsy = qs0_y + dU_y * fu;
    //        const float qsz = qs0_z + dU_z * fu;

    //        const float r2 = qsx * qsx + qsy * qsy + qsz * qsz;
    //        const float w = inv_dsd * rsqrtf(fmaxf(r2, 1e-20f));

    //        dst_row[u] = src_row[u] * w;
    //    }
    //}

    class PreweightManagerVec {
    public:
        PreweightManagerVec() { setPolicy(SKernelLaunchPolicy{}); }
        explicit PreweightManagerVec(const SKernelLaunchPolicy& p) { setPolicy(p); }

        void setPolicy(const SKernelLaunchPolicy& p) { policy_ = normalizePreweightPolicy(p); }
        const SKernelLaunchPolicy& policy() const { return policy_; }

        // geo/gv 由调用者负责偏移（+ base_a）后传入
        void applyChunk(const SDimensions3D& dims,
            const float* d_src_chunk, float* d_dst_chunk,
            const SConeProjectionVec* d_geo,
            const SFDKGeoParamPerView* d_gv,
            int K,
            cudaStream_t stream = (cudaStream_t)0) const
        {
            if (!d_src_chunk || !d_dst_chunk || !d_geo || !d_gv) return;

            const int Nu = (int)dims.iPU;
            const int Nv = (int)dims.iPV;
            if (Nu <= 0 || Nv <= 0 || K <= 0) return;

            const int blockThreads = policy_.block_threads;
            const int warps_per_blk = blockThreads / 32;
            const int total_warps = K * Nv;
            const int blocks = (total_warps + warps_per_blk - 1) / warps_per_blk;
            const size_t smem_bytes = (size_t)warps_per_blk * kPreweightSlot * sizeof(float);

            preweight_vec_chunk_rowwarp_kernel << <blocks, blockThreads, smem_bytes, stream >> > (
                d_src_chunk, d_dst_chunk,
                d_geo, d_gv,
                Nu, Nv, K,
                policy_.bounds_check ? 1 : 0);

            YK_CUDA_KERNEL_CHECK();
        }

    private:
        SKernelLaunchPolicy policy_{};
    };

} // namespace YK