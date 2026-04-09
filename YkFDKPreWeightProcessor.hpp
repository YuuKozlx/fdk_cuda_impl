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
        const SConeProjGeomVec* __restrict__ geo,
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

        const float fv = (float)v;
        const float qs0_x = (dS_x + dV_x * fv) - src_x;
        const float qs0_y = (dS_y + dV_y * fv) - src_y;
        const float qs0_z = (dS_z + dV_z * fv) - src_z;

        const size_t base = ((size_t)i * Nv + (size_t)v) * Nu;
        const float* src_row = src + base;
        float* dst_row = dst + base;

        for (int u = lane; u < Nu; u += 32) {
            const float fu = (float)u;
            const float qsx = qs0_x + dU_x * fu;
            const float qsy = qs0_y + dU_y * fu;
            const float qsz = qs0_z + dU_z * fu;
            const float r2 = qsx * qsx + qsy * qsy + qsz * qsz;
            const float w = DSD * rsqrtf(fmaxf(r2, 1e-20f));
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
        void process(const void* d_input,
            void* d_output,
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

            const float* d_in = static_cast<const float*>(d_input);
            float* d_out = static_cast<float*>(d_output);

            const int blockThreads = policy_.block_threads;
            const int warps_per_blk = blockThreads / 32;
            const int total_warps = chunk_.K * Nv_;
            const int blocks = (total_warps + warps_per_blk - 1) / warps_per_blk;
            const size_t smem_bytes = (size_t)warps_per_blk * kPreweightSlot * sizeof(float);

            preweight_vec_chunk_rowwarp_kernel << <blocks, blockThreads, smem_bytes, stream >> > (
                d_in, d_out,
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




};
namespace YK {

    __global__ void print_d_out_kernel(const float* data, int Nu, int Nv, int K) {
        const int u = blockIdx.x * blockDim.x + threadIdx.x;
        const int angle = blockIdx.y * blockDim.y + threadIdx.y;
        if (u >= Nu || angle >= K) return;
        for (int v = 0; v < Nv; ++v) {
            const int idx = (angle * Nv + v) * Nu + u;
            printf("data[%d] = %f\n", idx, data[idx]);
        }
    }

    // ============================================================
    // Constant memory
    // ============================================================
    __constant__ float gC_parker_angle[kMaxChunkAng];

    // ============================================================
    // Parker weighting kernel
    // 数据布局：[K, Nv, Nu]，紧密排列，无 padding
    // beta  = gC_parker_angle[angle]，已归一化到 [0, 2π)
    // gamma = atan(u_mm / SDD)，u_mm 为探测器列的物理坐标
    // ============================================================
    __global__ void parker_weight_kernel(
        float* __restrict__ data,
        int   Nu,
        int   Nv,
        int   K,
        float fSDD,
        float fDetUSize,
        float fCentralFanAngle,
        float fScale)
    {
        const int u = blockIdx.x * blockDim.x + threadIdx.x;
        const int angle = blockIdx.y * blockDim.y + threadIdx.y;

        /*      printf("threadIdx=(%d,%d) blockIdx=(%d,%d) u=%d angle=%d, data[%d] = %f\n",
                  threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, u, angle, angle * Nv * Nu + 0 * Nv + u, data[angle * Nv * Nu + u]);*/

        if (u >= Nu || angle >= K) return;

        // 探测器列物理坐标（以探测器中心为原点）
        const float u_mm = (u - 0.5f * Nu + 0.5f) * fDetUSize;

        // 扇形角 γ
        const float gamma = atanf(u_mm / fSDD);

        // 当前视角扫描角 β
        const float beta = gC_parker_angle[angle];

        // Parker 权重分段函数
        const float t1 = 2.0f * (fCentralFanAngle + gamma);
        const float t2 = CUDA_PI + 2.0f * gamma;
        const float t3 = CUDA_PI + 2.0f * fCentralFanAngle;

        float w;
        if (beta <= 0.0f) {
            w = 0.0f;
        }
        else if (beta < t1) {
            const float arg = (CUDA_PI * 0.25f) * beta / (fCentralFanAngle + gamma);
            const float s = sinf(arg);
            w = s * s;
        }
        else if (beta <= t2) {
            w = 1.0f;
        }
        else if (beta < t3) {
            const float arg = (CUDA_PI * 0.25f) * (CUDA_PI + 2.0f * fCentralFanAngle - beta)
                / (fCentralFanAngle - gamma);
            const float s = sinf(arg);
            w = s * s;
        }
        else {
            w = 0.0f;
        }

        w *= fScale;
        //printf("w = %.4f \n", w);

        //printf("angle=%d u=%d beta=%.4f gamma=%.4f w=%.4f\n", angle, u, beta, gamma, w);


        // 沿 v 方向写回，同一 (angle, u) 所有 v 权重相同
        for (int v = 0; v < Nv; ++v) {
            const int idx = (angle * Nv + v) * Nu + u;
            data[idx] *= w;
            // printf("data = %.4f\n", data[idx]);
            //if (v == 0)
            //    printf("idx=%d w=%.4f data=%.4f\n", idx, w, data[idx]);
        }
    }

    // ============================================================
    // ParkerWeightProcessor : IProcessor
    // ============================================================
    class ParkerWeightProcessor : public IProcessor {
    public:
        ParkerWeightProcessor() = default;
        ~ParkerWeightProcessor() override { release(); }

        ParkerWeightProcessor(const ParkerWeightProcessor&) = delete;
        ParkerWeightProcessor& operator=(const ParkerWeightProcessor&) = delete;

        // ----------------------------------------------------------------
        // IProcessor::setInitContext
        // ----------------------------------------------------------------
        void setInitContext(const void* ctx) override
        {
            if (!ctx) {
                std::fprintf(stderr, "[YK][Parker][E] setInitContext: null.\n");
                return;
            }
            const auto* ic = static_cast<const ParkerWeightInitContext*>(ctx);

            Nu_ = ic->dims.iPU;
            Nv_ = ic->dims.iPV;
            fDetUSize_ = ic->fDetUSize;
            fSrcOrigin_ = ic->fSrcOrigin;
            fDetOrigin_ = ic->fDetOrigin;

            // fScale = π / range，使冗余区域积分为 1
            fScale_ = ic->fScanRangeRad / (float)CUDA_PI;
            fAngleBase_ = ic->fStartAngleRad;

            cfg_ready_ = true;
        }

        // ----------------------------------------------------------------
        // IProcessor::init
        // ----------------------------------------------------------------
        bool init() override
        {
            if (!cfg_ready_) {
                std::fprintf(stderr, "[YK][Parker][E] init: setInitContext() not called.\n");
                return false;
            }
            if (Nu_ <= 0 || Nv_ <= 0) {
                std::fprintf(stderr, "[YK][Parker][E] init: invalid dims Nu=%d Nv=%d.\n",
                    Nu_, Nv_);
                return false;
            }


            const float fSDD = fSrcOrigin_ + fDetOrigin_;
            fCentralFanAngle_ = std::fabs(
                std::atanf(fDetUSize_ * (Nu_ * 0.5f) / fSDD));

            // 检查扫描范围是否足够
            const float fRange = fScale_ * (float)CUDA_PI;
            if (fRange + 1e-3f < (float)CUDA_PI + 2.0f * fCentralFanAngle_) {
                std::fprintf(stderr,
                    "[YK][Parker][W] init: angular range (%.4f rad) smaller than "
                    "Parker weighting range (%.4f rad).\n",
                    fRange, (float)CUDA_PI + 2.0f * fCentralFanAngle_);
            }

            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // IProcessor::setContext
        // ----------------------------------------------------------------
        void setContext(const void* ctx) override
        {
            if (!is_initialized_) {
                std::fprintf(stderr, "[YK][Parker][E] setContext: not initialized.\n");
                return;
            }
            if (!ctx) {
                std::fprintf(stderr, "[YK][Parker][E] setContext: null.\n");
                return;
            }
            const auto* cc = static_cast<const ParkerWeightChunkContext*>(ctx);
            if (cc->K <= 0) {
                std::fprintf(stderr, "[YK][Parker][E] setContext: invalid K=%d.\n", cc->K);
                return;
            }
            chunk_ = *cc;
        }

        // ----------------------------------------------------------------
        // IProcessor::process
        //   d_input  : [K*Nv*Nu] device
        //   d_output : [K*Nv*Nu] device（in-place: d_output == d_input 亦可）
        // ----------------------------------------------------------------
        void process(const void* d_input,
            void* d_output,
            cudaStream_t stream = 0) override
        {

            if (!is_initialized_) {
                std::fprintf(stderr, "[YK][Parker][E] process: not initialized.\n");
                return;
            }
            if (!d_input || !d_output) {
                std::fprintf(stderr, "[YK][Parker][E] process: null pointer.\n");
                return;
            }

            if (d_input != d_output) {
                std::fprintf(stderr, "[YK][Parker][W] process: in-place only, d_input ignored.\n");
                return;
            }
            if (chunk_.K <= 0) {
                std::fprintf(stderr, "[YK][Parker][E] process: setContext() not called.\n");
                return;
            }





            auto d_out = static_cast<float*>(d_output);
            auto d_in = static_cast<const float*>(d_input);



            std::vector<float> rel(chunk_.K);
            for (int i = 0; i < chunk_.K; ++i) {
                float f = chunk_.h_angles[i] - fAngleBase_;
                while (f < 0.f)           f += 2.f * CUDA_PI;
                while (f >= 2.f * CUDA_PI) f -= 2.f * CUDA_PI;
                rel[i] = f;
            }

            const int K = chunk_.K;

            // 上传本 chunk 的相对角度到 constant memory

            YK_CUDA_CHECK(cudaMemcpyToSymbol(
                gC_parker_angle,
                rel.data(),
                K * sizeof(float),
                0,
                cudaMemcpyHostToDevice));


            dim3 dimBlock(32, 8);
            dim3 dimGrid(
                (Nu_ + 31) / 32,
                (K + 7) / 8);



            parker_weight_kernel << <dimGrid, dimBlock, 0, stream >> > (
                d_out,
                Nu_, Nv_, K,
                fSrcOrigin_ + fDetOrigin_,
                fDetUSize_,
                fCentralFanAngle_,
                fScale_);

            YK_CUDA_KERNEL_CHECK();
        }

        // ----------------------------------------------------------------
        // IProcessor::release
        // ----------------------------------------------------------------
        void release() override
        {
            Nu_ = 0;
            Nv_ = 0;
            fDetUSize_ = 1.f;
            fSrcOrigin_ = 0.f;
            fDetOrigin_ = 0.f;
            fCentralFanAngle_ = 0.f;
            fScale_ = 1.f;
            chunk_ = {};
            is_initialized_ = false;
            cfg_ready_ = false;
        }

        bool        isInitialized() const override { return is_initialized_; }
        const char* name()          const override { return "ParkerWeightProcessor"; }

    private:
        int   Nu_ = 0;
        int   Nv_ = 0;
        float fDetUSize_ = 1.f;
        float fSrcOrigin_ = 0.f;
        float fDetOrigin_ = 0.f;
        float fCentralFanAngle_ = 0.f;
        float fScale_ = 1.f;
        float fAngleBase_ = 0.f;  // 可选：基准角度，默认为 0（即第一个视角的绝对角度）；如果设置为其他值，则相对角度 = 绝对角度 - 基准角度

        ParkerWeightChunkContext chunk_ = {};
        bool                    is_initialized_ = false;
        bool                    cfg_ready_ = false;
    };

} // namespace YK