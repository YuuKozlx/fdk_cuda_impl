#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"
#include "IProcessor.hpp"
#include "YkFdkFilterContext.hpp"

namespace YK {
    namespace cg = cooperative_groups;

    inline SKernelLaunchPolicy normalizePreweightPolicy(SKernelLaunchPolicy p) {
        if (p.block_threads < 32) p.block_threads = 32;
        p.block_threads = (p.block_threads + 31) & ~31;
        p.block_threads = std::min(p.block_threads, 1024);
        return p;
    }

    constexpr int kValidFloats = 13;
    constexpr int kPreweightSlot = (kValidFloats + 15) & ~15;  // = 16

    __global__ void preweight_vec_chunk_rowwarp_kernel(
        const float* __restrict__ src,
        float* __restrict__ dst,
        const SConeProjectionVec* __restrict__ geo,
        const SFDKGeoParamPerView* __restrict__ gv,
        int Nu, int Nv, int K,
        int bounds_check)
    {
        const int lane = (int)threadIdx.x & 31;
        const int warp_in_block = (int)threadIdx.x >> 5;
        const int warps_per_blk = (int)(blockDim.x >> 5);
        const int warp_global = (int)blockIdx.x * warps_per_blk + warp_in_block;

        if (warp_global >= K * Nv) return;

        const int i = warp_global / Nv;
        const int v = warp_global - i * Nv;

        if (bounds_check && i >= K) return;

        extern __shared__ float smem[];
        float* ws = smem + warp_in_block * kPreweightSlot;

        if (lane == 0) {
            const SConeProjectionVec& g = geo[i];
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

        const float fv = (float)v;
        const float qs0_x = (dS_x + dV_x * fv) - src_x;
        const float qs0_y = (dS_y + dV_y * fv) - src_y;
        const float qs0_z = (dS_z + dV_z * fv) - src_z;
        const float inv_dsd = (DSD > 0.0f) ? DSD : 0.0f;

        const size_t base = ((size_t)i * Nv + (size_t)v) * Nu;
        const float* src_row = src + base;
        float* dst_row = dst + base;

        for (int u = lane; u < Nu; u += 32) {
            const float fu = (float)u;
            const float qsx = qs0_x + dU_x * fu;
            const float qsy = qs0_y + dU_y * fu;
            const float qsz = qs0_z + dU_z * fu;
            const float r2 = qsx * qsx + qsy * qsy + qsz * qsz;
            const float w = inv_dsd * rsqrtf(fmaxf(r2, 1e-20f));
            dst_row[u] = src_row[u] * w;
        }
    }



    // ============================================================
    // PreweightProcessor : IProcessor
    // ============================================================
    class PreweightProcessor : public IProcessor {
    public:
        PreweightProcessor() = default;
        ~PreweightProcessor() override { release(); }

        PreweightProcessor(const PreweightProcessor&) = delete;
        PreweightProcessor& operator=(const PreweightProcessor&) = delete;

        // ----------------------------------------------------------------
        // IProcessor::setInitContext
        // ----------------------------------------------------------------
        void setInitContext(const void* ctx) override
        {
            if (!ctx) {
                std::fprintf(stderr, "[YK][Preweight][E] setInitContext: null.\n"); return;
            }
            const auto* ic = static_cast<const PreweightInitContext*>(ctx);
            Nu_ = (int)ic->dims.iPU;
            Nv_ = (int)ic->dims.iPV;
            policy_ = normalizePreweightPolicy(ic->policy);
            cfg_ready_ = true;
        }

        // ----------------------------------------------------------------
        // IProcessor::init — 无 GPU 资源，只验参
        // ----------------------------------------------------------------
        bool init() override
        {
            if (!cfg_ready_) {
                std::fprintf(stderr, "[YK][Preweight][E] init: setInitContext() not called.\n");
                return false;
            }
            if (Nu_ <= 0 || Nv_ <= 0) {
                std::fprintf(stderr, "[YK][Preweight][E] init: invalid dims Nu=%d Nv=%d.\n",
                    Nu_, Nv_);
                return false;
            }
            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // IProcessor::setContext — 每 chunk 前注入 geo/gv/K
        // ----------------------------------------------------------------
        void setContext(const void* ctx) override
        {
            if (!is_initialized_) {
                std::fprintf(stderr, "[YK][Preweight][E] setContext: not initialized.\n"); return;
            }
            if (!ctx) {
                std::fprintf(stderr, "[YK][Preweight][E] setContext: null.\n"); return;
            }
            const auto* cc = static_cast<const PreweightChunkContext*>(ctx);
            if (!cc->d_geo || !cc->d_gv || cc->K <= 0) {
                std::fprintf(stderr, "[YK][Preweight][E] setContext: invalid chunk context.\n"); return;
            }
            chunk_ = *cc;
        }

        // ----------------------------------------------------------------
        // IProcessor::process
        //   d_input  : [K*Nv*Nu] device
        //   d_output : [K*Nv*Nu] device（in-place 亦可）
        // ----------------------------------------------------------------
        void process(const float* d_input,
            float* d_output,
            cudaStream_t stream = 0) override
        {
            if (!is_initialized_) {
                std::fprintf(stderr, "[YK][Preweight][E] process: not initialized.\n"); return;
            }
            if (!d_input || !d_output) {
                std::fprintf(stderr, "[YK][Preweight][E] process: null pointer.\n"); return;
            }
            if (!chunk_.d_geo || !chunk_.d_gv || chunk_.K <= 0) {
                std::fprintf(stderr, "[YK][Preweight][E] process: setContext() not called.\n"); return;
            }

            const int blockThreads = policy_.block_threads;
            const int warps_per_blk = blockThreads / 32;
            const int total_warps = chunk_.K * Nv_;
            const int blocks = (total_warps + warps_per_blk - 1) / warps_per_blk;
            const size_t smem_bytes = (size_t)warps_per_blk * kPreweightSlot * sizeof(float);

            preweight_vec_chunk_rowwarp_kernel << <blocks, blockThreads, smem_bytes, stream >> > (
                d_input, d_output,
                chunk_.d_geo, chunk_.d_gv,
                Nu_, Nv_, chunk_.K,
                policy_.bounds_check ? 1 : 0);

            YK_CUDA_KERNEL_CHECK();
        }

        // ----------------------------------------------------------------
        // IProcessor::release — 无 GPU 资源需释放
        // ----------------------------------------------------------------
        void release() override
        {
            Nu_ = Nv_ = 0;
            chunk_ = {};
            is_initialized_ = false;
            cfg_ready_ = false;
        }

        bool        isInitialized() const override { return is_initialized_; }
        const char* name()          const override { return "PreweightProcessor"; }

    private:
        int                 Nu_ = 0;
        int                 Nv_ = 0;
        SKernelLaunchPolicy policy_ = {};
        PreweightChunkContext chunk_ = {};
        bool is_initialized_ = false;
        bool cfg_ready_ = false;
    };

} // namespace YK