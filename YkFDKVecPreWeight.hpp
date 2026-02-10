#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"

namespace YK {

    // ============================================================
    // Vec geometry preweight (chunk batch, power=1)
    // w_pre(u,v) = |(detS-src)·n_hat| / |(Q-src)|
    // Q = detS + u*detU + v*detV
    //
    // layout: [K][Nv][Nu] contiguous
    // ============================================================
    __global__ void _kernel_preweight_vec_chunk_warprow(
        const float* __restrict__ src,              // [K*Nv*Nu]
        float* __restrict__ dst,                    // [K*Nv*Nu]
        const SConeProjectionVec* __restrict__ geo, // [Ang]
        int Nu, int Nv,
        int K, int base_a)
    {
        // one warp processes one detector row (fixed v) of one view i
        int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // /32
        int lane = threadIdx.x & 31;

        int total_warps = K * Nv;
        if (warp_global >= total_warps) return;

        int i = warp_global / Nv;       // view index within chunk
        int v = warp_global - i * Nv;   // row
        int a = base_a + i;

        SConeProjectionVec g = geo[a];
        float3 U = g.detU;
        float3 V = g.detV;

        float3 n = f3_cross(U, V);
        float nlen2 = f3_dot(n, n);
        if (nlen2 < 1e-20f) return;

        float invn = rsqrtf(nlen2);
        float3 nh = make_float3(n.x * invn, n.y * invn, n.z * invn);

        float DSD_n = f3_dot(f3_sub(g.detS, g.src), nh);
        if (DSD_n < 0.0f) DSD_n = -DSD_n;

        // precompute detS + v*V
        float3 detSv = f3_add(g.detS, f3_mul(V, (float)v));

        size_t base = ((size_t)i * (size_t)Nv + (size_t)v) * (size_t)Nu;

        for (int u = lane; u < Nu; u += 32) {
            float3 Q = f3_add(detSv, f3_mul(U, (float)u));
            float3 QS = f3_sub(Q, g.src);

            float r2 = f3_dot(QS, QS);
            float r = sqrtf(fmaxf(r2, 1e-20f));

            float w = DSD_n / r;
            dst[base + (size_t)u] = src[base + (size_t)u] * w;
        }
    }

    class PreweightManagerVec {
    public:
        // init: only needs Nu/Nv and default launch policy
        bool init(int Nu, int Nv,
            int blockThreads = 256,   // must be multiple of 32
            cudaStream_t stream = 0)
        {
            Nu_ = Nu; Nv_ = Nv;
            stream_ = stream;

            if (blockThreads < 32 || (blockThreads % 32) != 0) blockThreads = 256;
            blockThreads_ = blockThreads;

            inited_ = (Nu_ > 0 && Nv_ > 0);
            return inited_;
        }

        void setStream(cudaStream_t s) { stream_ = s; }

        // apply chunk: src/dst are [K*Nv*Nu] contiguous
        // geo is device pointer [Ang]
        void applyChunk(const float* d_src_chunk,
            float* d_dst_chunk,
            const SConeProjectionVec* d_geo,
            int K, int base_a) const
        {
            if (!inited_ || K <= 0) return;

            const int warps = K * Nv_;
            const int warpsPerBlock = blockThreads_ / 32;
            const int blocks = (warps + warpsPerBlock - 1) / warpsPerBlock;

            _kernel_preweight_vec_chunk_warprow << <blocks, blockThreads_, 0, stream_ >> > (
                d_src_chunk, d_dst_chunk, d_geo, Nu_, Nv_, K, base_a);

            YK_CUDA_KERNEL_CHECK();
        }

        int Nu() const { return Nu_; }
        int Nv() const { return Nv_; }
        int blockThreads() const { return blockThreads_; }

    private:
        int Nu_ = 0, Nv_ = 0;
        int blockThreads_ = 256;
        cudaStream_t stream_ = 0;
        bool inited_ = false;
    };

    // 统一风格 helper（可选）
    inline void executeFdkPreweightVecChunk(
        PreweightManagerVec& pw,
        const float* d_in_chunk, float* d_out_chunk,
        const SConeProjectionVec* d_geo,
        int K, int base_a,
        cudaStream_t stream = 0)
    {
        pw.setStream(stream);
        pw.applyChunk(d_in_chunk, d_out_chunk, d_geo, K, base_a);
    }

} // namespace YK

