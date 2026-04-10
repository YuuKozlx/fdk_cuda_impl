#pragma once
#include <cstdio>

#include <cuda_runtime.h>

#include "../global/IProcessor.hpp"
#include "../FDK/YkVecGeo.hpp"
#include "../FDK/YkFdkPipelineContext.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"

#include "cuh/YkFDKPreWeightHelpers.cuh"
#include "cuh/YkFDKPreWeightLaunch.cuh"

namespace YK {
    namespace Fdk {

        // ============================================================
        // PreweightProcessor
        //
        //   生命周期：
        //     setInitContext(&PreweightInitContext{...})
        //     init()
        //     loop:
        //       setContext(&PreweightChunkContext{ d_geo, d_gv, K })
        //       process(d_in, d_out, stream)
        //     release()
        //
        //   支持原地：d_output == d_input 合法。
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
                Nu_ = static_cast<int>(ic->dims.iPU);
                Nv_ = static_cast<int>(ic->dims.iPV);
                policy_ = detail::normalizePreweightPolicy(ic->policy);
                cfg_ready_ = true;
            }

            // ----------------------------------------------------------------
            // IProcessor::init — 无 GPU 资源，仅验参
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
            //   d_output : [K*Nv*Nu] device（in-place 合法）
            // ----------------------------------------------------------------
            void process(const void* d_input, void* d_output,
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

                detail::pw_launchPreweight(
                    static_cast<const float*>(d_input),
                    static_cast<float*>(d_output),
                    chunk_.d_geo, chunk_.d_gv,
                    Nu_, Nv_, chunk_.K,
                    policy_, stream);
            }

            // ----------------------------------------------------------------
            // IProcessor::release — 无 GPU 资源
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
            int                    Nu_ = 0;
            int                    Nv_ = 0;
            SKernelLaunchPolicy    policy_ = {};
            PreweightChunkContext  chunk_ = {};
            bool                   is_initialized_ = false;
            bool                   cfg_ready_ = false;
        };

    }
} // namespace YK::Fdk
