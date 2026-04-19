#pragma once
#include <cmath>
#include <cstdio>
#include <vector>

#include "FDK/YkFdkPipelineContext.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "Heli/YkHelicalParkerLaunch.hpp"

namespace YK {
    namespace Helical {

        // ----------------------------------------------------------------
        // HelicalParkerInitContext
        // ----------------------------------------------------------------
        struct HelicalParkerInitContext {
            SProjDims dims;
            float     fDetUSize;      // du_mm
            float     fDetVSize;      // dv_mm
            float     fSrcOrigin;     // SID
            float     fDetOrigin;     // ODD = SDD - SID
            float     pitch_mm;
            float     z0;             // 该段重建中心 Z
            float     margin_mm;      // 锥角 margin
        };

        // ----------------------------------------------------------------
        // HelicalParkerChunkContext
        // ----------------------------------------------------------------
        struct HelicalParkerChunkContext {
            const float* h_angles;    // 该 chunk 的角度列表
            const float* h_z_src;     // 该 chunk 每个投影的源 Z
            int          K;
            float        angle_base;
        };

        // ----------------------------------------------------------------
        // HelicalParkerProcessor
        // ----------------------------------------------------------------
        class HelicalParkerProcessor {
        public:

            void setInitContext(const HelicalParkerInitContext* ctx)
            {
                Nu_ = ctx->dims.iPU;
                Nv_ = ctx->dims.iPV;
                K_ = ctx->dims.iPAng;
                fDetUSize_ = ctx->fDetUSize;
                fDetVSize_ = ctx->fDetVSize;
                fSDD_ = ctx->fSrcOrigin + ctx->fDetOrigin;
                fSID_ = ctx->fSrcOrigin;
                pitch_mm_ = ctx->pitch_mm;
                z0_ = ctx->z0;
                z_half_range_ = ctx->pitch_mm * 0.5f + ctx->margin_mm;
                cfg_ready_ = true;
            }

            void setContext(const HelicalParkerChunkContext* ctx)
            {
                chunk_angles_ = ctx->h_angles;
                chunk_z_src_ = ctx->h_z_src;
                K_chunk_ = ctx->K;
                angle_base_ = ctx->angle_base;
            }

            bool init()
            {
                if (!cfg_ready_) {
                    YK_LOGE("[HelicalParkerProcessor] setInitContext not called");
                    return false;
                }
                if (Nu_ <= 0 || Nv_ <= 0) {
                    YK_LOGE("[HelicalParkerProcessor] invalid dims {}x{}",
                        Nu_, Nv_);
                    return false;
                }

                fCentralFanAngle_ = std::fabs(
                    std::atan(fDetUSize_ * (Nu_ * 0.5f) / fSDD_));

                // fScale：Parker 有效范围归一化
                // 有效范围 = π + 2γ_max
                const float parker_range =
                    CUDA_PI + 2.f * fCentralFanAngle_;
                fScale_ = parker_range / CUDA_PI;

                is_initialized_ = true;
                YK_LOGI("[HelicalParkerProcessor] init OK: "
                    "Nu={} Nv={} SDD={:.1f} SID={:.1f} fan={:.2f}deg "
                    "z0={:.2f}mm z_half_range={:.2f}mm fScale={:.4f}",
                    Nu_, Nv_, fSDD_, fSID_,
                    fCentralFanAngle_ * 180.f / CUDA_PI,
                    z0_, z_half_range_, fScale_);
                return true;
            }

            void release() { is_initialized_ = false; }

            // process：in-place，直接修改 d_inout
            //void process(float* d_inout, cudaStream_t stream)
            //{
            //    if (!is_initialized_ || !chunk_angles_ || !chunk_z_src_) {
            //        YK_LOGE("[HelicalParkerProcessor] not ready");
            //        return;
            //    }

            //    // 上传角度和源 Z 到 constant memory
            //    helical_parker_upload(
            //        chunk_angles_, chunk_z_src_,
            //        K_chunk_,
            //        chunk_angles_[0]);  // angle_base = 第一个角度

            //    helical_parker_launch(
            //        d_inout,
            //        Nu_, Nv_, K_chunk_,
            //        fSDD_,
            //        fDetUSize_, fDetVSize_,
            //        fCentralFanAngle_,
            //        fScale_,
            //        z0_,
            //        z_half_range_,
            //        stream);
            //}

            void process(float* d_inout, cudaStream_t stream)
            {
                // debug：模拟中心投影的权重
                {
                    const float angle_base = chunk_angles_[0];
                    const int   mid = K_chunk_ / 2;
                    float beta = chunk_angles_[mid] - angle_base;
                    while (beta < 0.f)             beta += 2.f * CUDA_PI;
                    while (beta >= 2.f * CUDA_PI)  beta -= 2.f * CUDA_PI;

                    // CPU debug 里
                    const int   mid_v = Nv_ / 2;
                    const float v_center = mid_v - 0.5f * (Nv_ - 1);
                    const float dz_det = v_center * fDetVSize_ * fSID_ / fSDD_;
                    const float t = dz_det / z_half_range_;
                    const float w_cone = (fabsf(t) >= 1.f) ? 0.f
                        : cosf(CUDA_PI * 0.5f * t) * cosf(CUDA_PI * 0.5f * t);


                    // 中心 u 的 parker 权重
                    const float u_mm = 0.f;  // 中心列
                    const float gamma = atanf(u_mm / fSDD_);
                    const float t1 = 2.f * (fCentralFanAngle_ + gamma);
                    const float t2 = CUDA_PI + 2.f * gamma;
                    const float t3 = CUDA_PI + 2.f * fCentralFanAngle_;

                    float w_parker = 0.f;
                    if (beta <= 0.f)  w_parker = 0.f;
                    else if (beta < t1) {
                        const float s = sinf(CUDA_PI * 0.25f * beta
                            / (fCentralFanAngle_ + gamma));
                        w_parker = s * s;
                    }
                    else if (beta <= t2)   w_parker = 1.f;
                    else if (beta < t3) {
                        const float s = sinf(CUDA_PI * 0.25f
                            * (CUDA_PI + 2.f * fCentralFanAngle_ - beta)
                            / (fCentralFanAngle_ - gamma));
                        w_parker = s * s;
                    }

                    YK_LOGI("[parker_debug] chunk_mid={} beta={:.2f}deg "
                        "v_center={:.1f} dz_det={:.4f}mm w_cone={:.4f} "
                        "w_parker={:.4f} fScale={:.4f} total={:.4f}",
                        mid,
                        beta * 180.f / CUDA_PI,
                        v_center, dz_det, w_cone, w_parker, fScale_,
                        w_parker * w_cone * fScale_);
                }
                if (!is_initialized_ || !chunk_angles_ || !chunk_z_src_) {
                    YK_LOGE("[HelicalParkerProcessor] not ready");
                    return;
                }

                // 上传角度和源 Z 到 constant memory
                helical_parker_upload(
                    chunk_angles_, chunk_z_src_,
                    K_chunk_,
                    angle_base_);  // angle_base = 第一个角度

                helical_parker_launch(
                    d_inout,
                    Nu_, Nv_, K_chunk_,
                    fSDD_,
                    fSID_,
                    fDetUSize_, fDetVSize_,
                    fCentralFanAngle_,
                    fScale_,
                    z0_,
                    z_half_range_,
                    stream);
            }

        private:
            bool  cfg_ready_ = false;
            bool  is_initialized_ = false;

            int   Nu_ = 0, Nv_ = 0, K_ = 0;
            float fDetUSize_ = 0.f;
            float fDetVSize_ = 0.f;
            float fSDD_ = 0.f;
            float fSID_ = 0.f;
            float pitch_mm_ = 0.f;
            float z0_ = 0.f;
            float z_half_range_ = 0.f;
            float fCentralFanAngle_ = 0.f;
            float fScale_ = 1.f;
            float angle_base_ = 0.f;
            int          K_chunk_ = 0;
            const float* chunk_angles_ = nullptr;
            const float* chunk_z_src_ = nullptr;
        };

    } // namespace Helical
} // namespace YK