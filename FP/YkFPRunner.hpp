#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include <cstdio>
#include <algorithm>

#include "../global/YkMacro.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkCBCTParams.h"

#include "YkFPProcessor.hpp"
#include "YkFPPipelineContext.hpp"
#include "../FDK/YkVecGeo.hpp"
#include "../global/YkCudaTextureController.hpp"
#include "YkFPGpuContext.hpp"

namespace YK {
    namespace Fp {

        // ================================================================
        // FpReconstructor
        //
        //   生命周期A（固定体积）：
        //     init(params, Kchunk, stream)
        //     bindVolume(d_vol)
        //     run(d_sino_out, stream)
        //     release()
        //
        //   生命周期B（迭代更新体积）：
        //     init(params, Kchunk, stream)
        //     loop:
        //       update_volume(d_vol)
        //       bindVolume(d_vol)
        //       run(d_sino_out, stream)
        //     release()
        //
        //   d_sino_out 布局：[Na][Nv][Nu]，U 最快
        // ================================================================
        class FpReconstructor {
        public:
            FpReconstructor() = default;

            bool isInitialized() const { return is_initialized_; }
            bool isVolumeBound() const { return vol_tex_.valid(); }

            void release()
            {
                fp_.release();
                vol_tex_.destroy();
                geo_ = {};
                sino_ = {};
                Kchunk_ = 0;
                Na_ = Nu_ = Nv_ = 0;
                is_initialized_ = false;
            }

            // ----------------------------------------------------------------
            // init：从 SCBCTParams 初始化，内部构建几何
            // 对称于 FdkReconstructor::init(params, Kchunk, stream)
            // ----------------------------------------------------------------
            bool init(const SCBCTParams& params, int Kchunk,
                cudaStream_t stream, int deviceId = 0)
            {
                if (params.iPAng <= 0 || params.iPU <= 0 || params.iPV <= 0
                    || Kchunk <= 0) {
                    fprintf(stderr, "[FpReconstructor] invalid init params\n");
                    return false;
                }
                if ((int)params.angle_list.size() != params.iPAng) {
                    fprintf(stderr, "[FpReconstructor] angle_list size mismatch\n");
                    return false;
                }

                Kchunk_ = Kchunk;
                Na_ = params.iPAng;
                Nu_ = params.iPU;
                Nv_ = params.iPV;

                // ---- 构建体积几何（与 FdkReconstructor 完全一致）----
                vol_geom_ = SVolGeom::make_centered(
                    params.iVX, params.iVY, params.iVZ,
                    params.vox_xy_mm, params.vox_z_mm);
                vol_geom_.center = make_float3(
                    params.vol_offset_x_mm,
                    params.vol_offset_y_mm,
                    params.vol_offset_z_mm);

                // ---- 构建投影几何（与 FdkReconstructor::feed_impl 一致）----
                auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

                std::vector<SConeProjGeomVec>    h_geo(Na_);
                std::vector<SFDKGeoParamPerView> h_gv(Na_);

                build_circular_vec_geometry_from_theta(
                    h_geo, params.angle_list, Na_,
                    params.iPU, params.iPV,
                    params.du_mm, params.dv_mm,
                    params.SID, params.SDD - params.SID,
                    f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                    f3(rad2deg(params.tiltu_angle_rad),
                        rad2deg(params.tiltn_angle_rad),
                        rad2deg(params.tiltv_angle_rad)));



                // ---- 初始化 GPU 资源 ----
                geo_.init(h_geo, vol_geom_, deviceId);
                sino_.init(Na_, Nv_, Nu_, deviceId);

                // ---- 初始化 FpProcessor ----
                FpInitContext ictx;
                ictx.vol_geom = vol_geom_;
                ictx.Nu = Nu_;
                ictx.Nv = Nv_;
                ictx.accumulate = false;
                fp_.setInitContext(&ictx);
                if (!fp_.init()) {
                    fprintf(stderr, "[FpReconstructor] FpProcessor init failed\n");
                    return false;
                }

                is_initialized_ = true;
                return true;
            }

            // ----------------------------------------------------------------
            // bindVolume：绑定/更新体积 texture
            // 固定体积调一次；迭代场景每轮体积更新后重新调
            // ----------------------------------------------------------------
            bool bindVolume(const float* d_vol)
            {
                if (!is_initialized_) {
                    fprintf(stderr, "[FpReconstructor] bindVolume: call init() first\n");
                    return false;
                }
                if (!d_vol) {
                    fprintf(stderr, "[FpReconstructor] bindVolume: d_vol is null\n");
                    return false;
                }
                vol_tex_.destroy();
                Mem::TextureController tc;
                vol_tex_ = tc.createTex3DFromDevice(d_vol, vol_geom_);
                return vol_tex_.valid();
            }

            bool bindVolumeFromHost(const float* h_vol)
            {
                if (!is_initialized_) {
                    fprintf(stderr, "[FpReconstructor] bindVolumeFromHost: call init() first\n");
                    return false;
                }
                if (!h_vol) {
                    fprintf(stderr, "[FpReconstructor] bindVolumeFromHost: h_vol is null\n");
                    return false;
                }
                vol_tex_.destroy();
                Mem::TextureController tc;
                vol_tex_ = tc.createTex3DFromHost(h_vol, vol_geom_);
                return vol_tex_.valid();
            }

            //// ----------------------------------------------------------------
            //// run：使用当前绑定的体积执行正投，结果写入内部 sino 缓冲
            //// ----------------------------------------------------------------
            bool run(cudaStream_t stream)
            {
                return run_impl(sino_.data(), stream, nullptr);
            }

            // ----------------------------------------------------------------
            // run：结果写入外部 d_sino_out
            // ----------------------------------------------------------------
            bool run(float* d_sino_out, cudaStream_t stream)
            {
                return run_impl(d_sino_out, stream, nullptr);
            }

            // ----------------------------------------------------------------
            // run：带 dump 回调
            //   onDump(angleIndex, d_sino_slice_ptr, Nu*Nv)
            // ----------------------------------------------------------------
            bool run(
                float* d_sino_out,
                cudaStream_t stream,
                std::function<void(int, float*, size_t)> onDump)
            {
                return run_impl(d_sino_out, stream, onDump);
            }

            // ----------------------------------------------------------------
            // 访问内部 sinogram 缓冲（run() 后有效）
            // ----------------------------------------------------------------
            const FpSinoData& sinoData() const { return sino_; }

        private:
            SVolGeom     vol_geom_ = {};
            int          Kchunk_ = 0;
            int          Na_ = 0;
            int          Nu_ = 0;
            int          Nv_ = 0;
            bool         is_initialized_ = false;

            FpProcessor  fp_;
            FpGeoData    geo_;
            FpSinoData   sino_;
            Mem::TextureController::Tex3DHandle vol_tex_;

            // ----------------------------------------------------------------
            // run_impl
            // ----------------------------------------------------------------
            bool run_impl(
                float* d_sino_out,
                cudaStream_t stream,
                std::function<void(int, float*, size_t)> onDump)
            {
                if (!is_initialized_) {
                    fprintf(stderr, "[FpReconstructor] run: call init() first\n");
                    return false;
                }
                if (!vol_tex_.valid()) {
                    fprintf(stderr, "[FpReconstructor] run: call bindVolume() first\n");
                    return false;
                }
                if (!d_sino_out) {
                    fprintf(stderr, "[FpReconstructor] run: d_sino_out is null\n");
                    return false;
                }

                const size_t view_elems = (size_t)Nu_ * Nv_;
                bool reinit_needed = false;

                for (int base = 0; base < Na_; base += Kchunk_)
                {
                    const int K = std::min(Kchunk_, Na_ - base);

                    // 尾包：K 变化时重新初始化 processor
                    if (K != Kchunk_ && !reinit_needed) {
                        reinit_needed = true;
                        if (!reinitProcessor(K)) return false;
                    }

                    FpChunkContext chunkCtx = buildChunkContext(base, K);
                    fp_.setContext(&chunkCtx);
                    fp_.process(vol_tex_.texPtr(), d_sino_out, stream);

                    if (onDump) {
                        for (int i = 0; i < K; ++i)
                            onDump(base + i,
                                d_sino_out + (size_t)(base + i) * view_elems,
                                view_elems);
                    }
                }

                // 尾包后恢复标准 Kchunk，供下次 run 使用
                if (reinit_needed) {
                    if (!reinitProcessor(Kchunk_)) return false;
                }

                return true;
            }

            bool reinitProcessor(int K)
            {
                FpInitContext ictx;
                ictx.vol_geom = vol_geom_;
                ictx.Nu = Nu_;
                ictx.Nv = Nv_;
                ictx.accumulate = false;
                fp_.setInitContext(&ictx);
                if (!fp_.init()) {
                    fprintf(stderr, "[FpReconstructor] reinitProcessor K=%d failed\n", K);
                    return false;
                }
                return true;
            }

            FpChunkContext buildChunkContext(int offset, int K) const
            {
                FpChunkContext ctx;
                ctx.d_views = geo_.d_views(offset);
                ctx.h_views = geo_.h_views(offset);
                ctx.K = K;
                ctx.angleOffset = offset;
                return ctx;
            }
        };

        // ================================================================
        // 全局便捷函数（对称于 fdk_recon）
        // ================================================================
        YK_INLINE bool fp_project(
            const float* d_vol,
            float* d_sino_out,
            const SCBCTParams& params,
            int                Kchunk,
            cudaStream_t       stream,
            std::function<void(int, float*, size_t)> onDump = nullptr)
        {
            FpReconstructor fp;
            if (!fp.init(params, Kchunk, stream))
                return false;
            if (!fp.bindVolume(d_vol))
                return false;
            return fp.run(d_sino_out, stream, onDump);
        }

    } // namespace Fp
} // namespace YK