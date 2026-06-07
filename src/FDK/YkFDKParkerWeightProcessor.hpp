#pragma once
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#include "global/IProcessor.hpp"
#include "FDK/YkFdkPipelineContext.hpp"
#include "global/YkGlobals.h"              // CUDA_PI
#include "global/YkMacro.hpp"

#include "FDK/cuh/YkFDKParkerLaunch.cuh"

namespace YK {
    namespace Fdk {

        // ============================================================
        // ParkerWeightProcessor
        //
        //   生命周期：
        //     setInitContext(&ParkerWeightInitContext{...})
        //     init()
        //     loop:
        //       setContext(&ParkerWeightChunkContext{ h_angles, K })
        //       process(d_inout, d_inout, stream)   ← 仅支持原地
        //     release()
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
                    std::fprintf(stderr, "[YK][Parker][E] setInitContext: null.\n"); return;
                }
                const auto* ic = static_cast<const ParkerWeightInitContext*>(ctx);

                Nu_ = ic->dims.iPU;
                Nv_ = ic->dims.iPV;
                fDetUSize_ = ic->fDetUSize;
                fSrcOrigin_ = ic->fSrcOrigin;
                fDetOrigin_ = ic->fDetOrigin;
                // fScale = scanRange / π，使冗余区域积分归一
                fScale_ = ic->fScanRangeRad / static_cast<float>(CUDA_PI);
                fAngleBase_ = ic->fStartAngleRad;
                cfg_ready_ = true;
                nDirSign_ = ic->nDirSign;
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
#ifdef _WIN32
                fCentralFanAngle_ = std::fabs(std::atanf(fDetUSize_ * (Nu_ * 0.5f) / fSDD));
#elif defined(__unix__)
                fCentralFanAngle_ = std::fabs(std::atan(fDetUSize_ * (Nu_ * 0.5f) / fSDD));
#endif
                // 检查扫描范围是否足够覆盖 Parker 权重范围
                const float fRange = fScale_ * static_cast<float>(CUDA_PI);
                if (fRange + 1e-3f < static_cast<float>(CUDA_PI) + 2.0f * fCentralFanAngle_) {
                    std::fprintf(stderr,
                        "[YK][Parker][W] init: angular range (%.4f rad) smaller than "
                        "Parker weighting range (%.4f rad).\n",
                        fRange, static_cast<float>(CUDA_PI) + 2.0f * fCentralFanAngle_);
                }

                is_initialized_ = true;
                return true;
            }

            // ----------------------------------------------------------------
            // IProcessor::setContext — 每 chunk 前上传角度
            // ----------------------------------------------------------------
            void setContext(const void* ctx) override
            {
                if (!is_initialized_) {
                    std::fprintf(stderr, "[YK][Parker][E] setContext: not initialized.\n"); return;
                }
                if (!ctx) {
                    std::fprintf(stderr, "[YK][Parker][E] setContext: null.\n"); return;
                }
                const auto* cc = static_cast<const ParkerWeightChunkContext*>(ctx);
                if (cc->K <= 0) {
                    std::fprintf(stderr, "[YK][Parker][E] setContext: invalid K=%d.\n", cc->K); return;
                }

                // 角度归一化 + 上传至 constant memory
                detail::pk_uploadAngles(cc->h_angles, cc->K, fAngleBase_, nDirSign_);
                chunk_ = *cc;
            }

            // ----------------------------------------------------------------
            // IProcessor::process  — 仅支持原地，d_input == d_output
            // ----------------------------------------------------------------
            void process(const void* d_input, void* d_output,
                cudaStream_t stream = 0) override
            {
                if (!is_initialized_) {
                    std::fprintf(stderr, "[YK][Parker][E] process: not initialized.\n"); return;
                }
                if (!d_input || !d_output) {
                    std::fprintf(stderr, "[YK][Parker][E] process: null pointer.\n"); return;
                }
                if (d_input != d_output) {
                    std::fprintf(stderr, "[YK][Parker][W] process: in-place only, d_input ignored.\n");
                    return;
                }
                if (chunk_.K <= 0) {
                    std::fprintf(stderr, "[YK][Parker][E] process: setContext() not called.\n"); return;
                }

                detail::pk_launchParker(
                    static_cast<float*>(d_output),
                    Nu_, Nv_, chunk_.K,
                    fSrcOrigin_ + fDetOrigin_,
                    fDetUSize_,
                    fCentralFanAngle_,
                    fScale_,
                    nDirSign_,
                    stream);
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
                fAngleBase_ = 0.f;
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
            int nDirSign_ = +1; // 方向符号，+1 或 -1，影响权重函数的正负号
            /// 基准角度：相对角 = 绝对角 - fAngleBase_
            float fAngleBase_ = 0.f;

            ParkerWeightChunkContext chunk_ = {};
            bool                     is_initialized_ = false;
            bool                     cfg_ready_ = false;
        };

    }
} // namespace YK::Fdk
