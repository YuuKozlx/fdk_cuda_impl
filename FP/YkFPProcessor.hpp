#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#include "../global/IProcessor.hpp"
#include "YkFPPipelineContext.hpp"
#include "kernels/YkFPLaunch.cuh"
#include "kernels/YkFPHelpers.cuh"   // MainAxis, getMainAxis

namespace YK {
    namespace Fp {

        // ============================================================
        // FpProcessor
        //
        //   生命周期：
        //     setInitContext(&FpInitContext{ vol_geom, Nu, Nv, accumulate })
        //     init()
        //     loop:
        //       setContext(&FpChunkContext{ d_views, h_views, K, angleOffset })
        //       process(d_volTexArray, d_sino, stream)
        //     release()
        //
        //   d_input  : const cudaTextureObject_t* — 取 [0] 作为体积 texture
        //   d_output : float*                     — sinogram [Na_total][Nv][Nu]
        //
        //   主轴分组在 process() 内部完成，对调用方透明
        // ============================================================
        class FpProcessor : public IProcessor {
        public:
            FpProcessor() = default;
            ~FpProcessor() override { release(); }
            FpProcessor(const FpProcessor&) = delete;
            FpProcessor& operator=(const FpProcessor&) = delete;

            // ----------------------------------------------------------------
            // IProcessor::setInitContext
            // ----------------------------------------------------------------
            void setInitContext(const void* ctx) override
            {
                if (!ctx) {
                    std::fprintf(stderr, "[YK][Fp][E] setInitContext: null.\n"); return;
                }
                const auto* ic = static_cast<const FpInitContext*>(ctx);
                vol_geom_ = ic->vol_geom;
                Nu_ = ic->Nu;
                Nv_ = ic->Nv;
                accumulate_ = ic->accumulate;
                cfg_ready_ = true;
            }

            // ----------------------------------------------------------------
            // IProcessor::init
            // ----------------------------------------------------------------
            bool init() override
            {
                if (!cfg_ready_) {
                    std::fprintf(stderr, "[YK][Fp][E] init: setInitContext not called.\n");
                    return false;
                }
                if (vol_geom_.Nx <= 0 || vol_geom_.Ny <= 0 || vol_geom_.Nz <= 0
                    || Nu_ <= 0 || Nv_ <= 0) {
                    std::fprintf(stderr, "[YK][Fp][E] init: invalid geometry.\n");
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
                    std::fprintf(stderr, "[YK][Fp][E] setContext: not initialized.\n"); return;
                }
                if (!ctx) {
                    std::fprintf(stderr, "[YK][Fp][E] setContext: null.\n"); return;
                }
                const auto* cc = static_cast<const FpChunkContext*>(ctx);
                if (cc->K <= 0 || !cc->d_views || !cc->h_views) {
                    std::fprintf(stderr,
                        "[YK][Fp][E] setContext: invalid context (K=%d, d_views=%p, h_views=%p).\n",
                        cc->K, (void*)cc->d_views, (void*)cc->h_views);
                    return;
                }
                chunk_ = *cc;
            }

            // ----------------------------------------------------------------
            // IProcessor::process
            //   d_input  : const cudaTextureObject_t* — [0] 是体积 texture
            //   d_output : float*                     — sinogram [Na_total][Nv][Nu]
            // ----------------------------------------------------------------
            void process(const void* d_input, void* d_output,
                cudaStream_t stream = 0) override
            {
                if (!is_initialized_ || chunk_.K <= 0) {
                    std::fprintf(stderr, "[YK][Fp][E] process: not ready.\n"); return;
                }
                if (!d_input || !d_output) {
                    std::fprintf(stderr, "[YK][Fp][E] process: null pointer.\n"); return;
                }

                const cudaTextureObject_t volTex =
                    static_cast<const cudaTextureObject_t*>(d_input)[0];
                float* d_sino = static_cast<float*>(d_output);

                // ---- 主轴分组：扫描 h_views，找连续同主轴的 run ----
                int i = 0;
                while (i < chunk_.K)
                {
                    const detail::MainAxis ax = detail::getMainAxis(chunk_.h_views[i]);
                    int j = i + 1;
                    while (j < chunk_.K && detail::getMainAxis(chunk_.h_views[j]) == ax)
                        ++j;
                    // [i, j) 是同一主轴的连续 run，dispatch 到对应 launch 函数
                    dispatchGroup(ax, volTex, d_sino, i, j, stream);
                    i = j;
                }
            }

            // ----------------------------------------------------------------
            // IProcessor::release
            // ----------------------------------------------------------------
            void release() override
            {
                chunk_ = {};
                vol_geom_ = {};
                Nu_ = Nv_ = 0;
                accumulate_ = false;
                is_initialized_ = false;
                cfg_ready_ = false;
            }

            bool        isInitialized() const override { return is_initialized_; }
            const char* name()          const override { return "FpProcessor"; }

        private:
            void dispatchGroup(detail::MainAxis ax,
                cudaTextureObject_t volTex,
                float* d_sino,
                int startAngle, int endAngle,
                cudaStream_t stream)
            {
                const int K = endAngle - startAngle;

                // Joseph 用归一化几何，指针偏移后 startAngle=0
                const SConeProjGeomVec* d_v = chunk_.d_views_vox + startAngle;
                float* d_s = d_sino + (size_t)startAngle * Nu_ * Nv_;

                switch (ax) {
                case detail::MainAxis::X:
                    fp_launchGroupX(volTex, d_v, d_s,
                        vol_geom_, Nu_, Nv_, 0, K, accumulate_, stream);
                    break;
                case detail::MainAxis::Y:
                    fp_launchGroupY(volTex, d_v, d_s,
                        vol_geom_, Nu_, Nv_, 0, K, accumulate_, stream);
                    break;
                case detail::MainAxis::Z:
                    fp_launchGroupZ(volTex, d_v, d_s,
                        vol_geom_, Nu_, Nv_, 0, K, accumulate_, stream);
                    break;
                }
            }

            SVolGeom        vol_geom_ = {};
            int             Nu_ = 0;
            int             Nv_ = 0;
            bool            accumulate_ = false;
            FpChunkContext  chunk_ = {};
            bool            is_initialized_ = false;
            bool            cfg_ready_ = false;
        };

    } // namespace Fp
} // namespace YK