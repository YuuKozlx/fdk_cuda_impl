#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

#include <cmath>
#include <crt/host_defines.h>
#include <driver_types.h>
#include <vector_types.h>
#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"

namespace YK {
    namespace cg = cooperative_groups;


    // normalize policy (host)
    inline SKernelLaunchPolicy normalizePreweightPolicy(SKernelLaunchPolicy p) {
        if (p.block_threads < 32) p.block_threads = 32;
        p.block_threads = (p.block_threads + 31) & ~31; // multiple of 32
        p.block_threads = std::min(p.block_threads, 1024);
        return p;
    }

    // ---------------------- kernel ----------------------
    __global__ void preweight_vec_chunk_rowwarp_kernel(
        const float* __restrict__ src,                 // [K*Nv*Nu]
        float* __restrict__ dst,                       // [K*Nv*Nu]
        const SConeProjectionVec* __restrict__ geo,    // [Ang]
        const SFDKGeoParamPerView* __restrict__ gv,  // [Ang]
        int Nu, int Nv, int K, int base_a, int Ang,
        int bounds_check)                               // 0/1
    {
        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

        const int warps_per_block = int(tb.size() / 32);
        const int warp_id_in_block = int(tb.thread_rank() / 32);
        const int warp_global = int(blockIdx.x) * warps_per_block + warp_id_in_block;

        const int total_warps = K * Nv;
        if (warp_global >= total_warps) return;

        const int i = warp_global / Nv;
        const int v = warp_global - i * Nv;
        const int a = base_a + i;

        if (bounds_check && (a < 0 || a >= Ang)) return;

        const SConeProjectionVec g = geo[a];
        const float DSD_n_abs = gv[a].SDD_mm;

        const float3 detSv = f3_add(g.detS, f3_mul(g.detV, (float)v));
        const size_t base = ((size_t)i * (size_t)Nv + (size_t)v) * (size_t)Nu;

        for (int u = (int)warp.thread_rank(); u < Nu; u += 32) {
            const float3 Q = f3_add(detSv, f3_mul(g.detU, (float)u));
            const float3 QS = f3_sub(Q, g.src);

            const float r2 = f3_dot(QS, QS);
            const float r = sqrtf(fmaxf(r2, 1e-20f));

            const float w = (DSD_n_abs > 0.0f) ? (DSD_n_abs / r) : 0.0f;

            const size_t idx = base + (size_t)u;
            dst[idx] = src[idx] * w;
        }
    }

    // ---------------------- manager (name kept) ----------------------
    class PreweightManagerVec {
    public:
        PreweightManagerVec() { setPolicy(SKernelLaunchPolicy{}); }
        explicit PreweightManagerVec(const SKernelLaunchPolicy& p) { setPolicy(p); }

        void setPolicy(const SKernelLaunchPolicy& p) { policy_ = normalizePreweightPolicy(p); }
        const SKernelLaunchPolicy& policy() const { return policy_; }

        void applyChunk(const SDimensions3D& dims,
            const float* d_src_chunk, float* d_dst_chunk,
            const SConeProjectionVec* d_geo,
            const SFDKGeoParamPerView* d_gv,
            int K, int base_a,
            cudaStream_t stream = (cudaStream_t)0 ) const
        {
            if (!d_src_chunk || !d_dst_chunk || !d_geo || !d_gv) return;

            const int Nu = (int)dims.iProjU;
            const int Nv = (int)dims.iProjV;
            const int Ang = (int)dims.iProjAngles;

            if (Nu <= 0 || Nv <= 0 || Ang <= 0 || K <= 0) return;

            const int blockThreads = policy_.block_threads;
            const int warps_per_block = blockThreads / 32;
            const int total_warps = K * Nv;
            const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

            preweight_vec_chunk_rowwarp_kernel << <blocks, blockThreads, 0, stream >> > (
                d_src_chunk, d_dst_chunk, d_geo, d_gv,
                Nu, Nv, K, base_a, Ang,
                policy_.bounds_check ? 1 : 0);

            YK_CUDA_KERNEL_CHECK();
        }

    private:
        SKernelLaunchPolicy policy_{};
    };

} // namespace YK
