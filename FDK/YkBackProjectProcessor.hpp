#pragma once
#include <cstdio>

#include <cuda_runtime.h>

#include "../global/IProcessor.hpp"
#include "YkVecGeo.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFdkPipelineContext.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"

#include "cuh/YkFDKBpLaunch.cuh"

namespace YK {
    namespace Fdk {

        // ============================================================
        // BpProcessor
        //
        //   生命周期：
        //     setInitContext(&BpInitContext{...})
        //     init()
        //     loop:
        //       setContext(&BpChunkContext{ d_texObjs, K, [d_geo, d_gv] })
        //       process(d_texObjs, d_vol, stream)
        //     release()
        //
        //   两种模式由 BpInitContext::use_precomputed 选择：
        //     true  — gC_coeffs constant memory 已在外部写入
        //     false — kernel 内直接计算，需要 d_geo / d_gv
        // ============================================================
        class BpProcessor : public IProcessor {
        public:
            BpProcessor() = default;
            ~BpProcessor() override { release(); }

            BpProcessor(const BpProcessor&) = delete;
            BpProcessor& operator=(const BpProcessor&) = delete;

            // ----------------------------------------------------------------
            // IProcessor::setInitContext
            // ----------------------------------------------------------------
            void setInitContext(const void* ctx) override
            {
                if (!ctx) {
                    std::fprintf(stderr, "[YK][Bp][E] setInitContext: null.\n"); return;
                }
                const auto* ic = static_cast<const BpInitContext*>(ctx);
                vol_geom_ = ic->vol_geom;
                use_precomputed_ = ic->use_precomputed;
                cfg_ready_ = true;
            }

            // ----------------------------------------------------------------
            // IProcessor::init
            // ----------------------------------------------------------------
            bool init() override
            {
                if (!cfg_ready_) {
                    std::fprintf(stderr, "[YK][Bp][E] init: setInitContext() not called.\n");
                    return false;
                }
                if (vol_geom_.Nx <= 0 || vol_geom_.Ny <= 0 || vol_geom_.Nz <= 0
                    || vol_geom_.vox_x <= 0.f
                    || vol_geom_.vox_y <= 0.f
                    || vol_geom_.vox_z <= 0.f) {
                    std::fprintf(stderr, "[YK][Bp][E] init: invalid vol_geom.\n");
                    return false;
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
                    std::fprintf(stderr, "[YK][Bp][E] setContext: not initialized.\n"); return;
                }
                if (!ctx) {
                    std::fprintf(stderr, "[YK][Bp][E] setContext: null.\n"); return;
                }
                const auto* cc = static_cast<const BpChunkContext*>(ctx);
                if (cc->K <= 0) {
                    std::fprintf(stderr, "[YK][Bp][E] setContext: invalid K=%d.\n", cc->K); return;
                }
                if (!use_precomputed_ && (!cc->d_geo || !cc->d_gv)) {
                    std::fprintf(stderr,
                        "[YK][Bp][E] setContext: non-precomputed mode requires d_geo and d_gv.\n");
                    return;
                }
                chunk_ = *cc;
            }

            // ----------------------------------------------------------------
            // IProcessor::process
            //   d_input  : const cudaTextureObject_t*  — device 纹理对象数组
            //   d_output : float*                      — [Nz*Ny*Nx] 体素累加缓冲
            // ----------------------------------------------------------------
            void process(const void* d_input, void* d_output,
                cudaStream_t stream = 0) override
            {
                if (!is_initialized_) {
                    std::fprintf(stderr, "[YK][Bp][E] process: not initialized.\n"); return;
                }
                if (!d_input || !d_output || chunk_.K <= 0) {
                    std::fprintf(stderr, "[YK][Bp][E] process: setContext() not called.\n"); return;
                }

                const cudaTextureObject_t* d_texObjs =
                    static_cast<const cudaTextureObject_t*>(d_input);
                float* d_vol = static_cast<float*>(d_output);

                if (use_precomputed_) {
                    detail::bp_launchBpPrecomputed(
                        d_texObjs, d_vol, vol_geom_,
                        chunk_.K, stream);
                }
                else {
                    detail::bp_launchBpDirect(
                        d_texObjs, chunk_.d_geo, chunk_.d_gv,
                        d_vol, vol_geom_,
                        chunk_.K, stream);
                }
            }

            // ----------------------------------------------------------------
            // IProcessor::release
            // ----------------------------------------------------------------
            void release() override
            {
                chunk_ = {};
                vol_geom_ = {};
                use_precomputed_ = true;
                is_initialized_ = false;
                cfg_ready_ = false;
            }

            bool        isInitialized() const override { return is_initialized_; }
            const char* name()          const override { return "BpProcessor"; }

        private:
            SVolGeom        vol_geom_ = {};
            bool            use_precomputed_ = true;
            BpChunkContext  chunk_ = {};
            bool            is_initialized_ = false;
            bool            cfg_ready_ = false;
        };

    }
} // namespace YK::Fdk
