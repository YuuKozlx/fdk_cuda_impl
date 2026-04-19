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
            float     fDetUSize;   // du_mm
            float     fSrcOrigin;  // SID
            float     fDetOrigin;  // ODD = SDD - SID
        };

        // ----------------------------------------------------------------
        // HelicalParkerChunkContext
        // ----------------------------------------------------------------
        struct HelicalParkerChunkContext {
            const float* h_angles;   // 该 chunk 的角度列表
            int          K;
            float        angle_base; // 该段第一个投影角度
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
                fSDD_ = ctx->fSrcOrigin + ctx->fDetOrigin;
                cfg_ready_ = true;
            }

            void setContext(const HelicalParkerChunkContext* ctx)
            {
                chunk_angles_ = ctx->h_angles;
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

                const float parker_range =
                    CUDA_PI + 2.f * fCentralFanAngle_;
                fScale_ = parker_range / CUDA_PI;

                is_initialized_ = true;
                YK_LOGI("[HelicalParkerProcessor] init OK: "
                    "Nu={} Nv={} SDD={:.1f} fan={:.2f}deg fScale={:.4f}",
                    Nu_, Nv_, fSDD_,
                    fCentralFanAngle_ * 180.f / CUDA_PI,
                    fScale_);
                return true;
            }

            void release() { is_initialized_ = false; }

            void process(float* d_inout, cudaStream_t stream)
            {
                if (!is_initialized_ || !chunk_angles_) {
                    YK_LOGE("[HelicalParkerProcessor] not ready");
                    return;
                }

                helical_parker_upload(
                    chunk_angles_,
                    K_chunk_,
                    angle_base_);

                helical_parker_launch(
                    d_inout,
                    Nu_, Nv_, K_chunk_,
                    fSDD_,
                    fDetUSize_,
                    fCentralFanAngle_,
                    fScale_,
                    stream);
            }

        private:
            bool  cfg_ready_ = false;
            bool  is_initialized_ = false;

            int   Nu_ = 0, Nv_ = 0, K_ = 0;
            float fDetUSize_ = 0.f;
            float fSDD_ = 0.f;
            float fCentralFanAngle_ = 0.f;
            float fScale_ = 1.f;

            int          K_chunk_ = 0;
            float        angle_base_ = 0.f;
            const float* chunk_angles_ = nullptr;
        };

    } // namespace Helical
} // namespace YK