#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <vector>
#include <vector_functions.hpp>
#include <vector_types.h>
#include "YkBackProjectProcessor.hpp"
#include "YkFDKFilterProcessor.hpp"
#include "YkFDKGpuContext.hpp"
#include "YkFDKParkerWeight.cuh"
#include "YkFDKPreWeight.cuh"
#include "YkFDKVecGeoDerived.hpp"
#include "../global/IProcessor.hpp"

#include "../global/YkMacro.hpp"
#include "YkVecGeo.hpp"
#include "YkFDKParkerWeightProcessor.hpp"
#include "YkFdkPipelineContext.hpp"
#include "../global/YkCBCTParams.h"
#include "../global/YkGlobals.h"
#include "../util/YkIoDump.hpp"
#include "../util/YkUtil.hpp"
#include "../util/YkVecOperation.hpp"

namespace YK {

    // ================================================================
    // FdkReconstructor
    // ================================================================
    class FdkReconstructor {
    public:
        FdkReconstructor() = default;

        // reset：开始新一轮扫描时清空累积状态
        void reset() {
            angle_accum_.clear();
            total_received_ = 0;
        }

        int  totalReceived()  const { return total_received_; }
        bool isInitialized()  const { return is_initialized_; }
        void release()
        {
            pw_.release();
            pkw_.release();
            flt_.release();
            bp_.release();
            reset();              // 清 angle_accum_ 和 total_received_
            Kchunk_ = 0;
            bParker_ = false;
            is_initialized_ = false;
        }

        // ----------------------------------------------------------------
        // init：固定参数确定后调一次，processor 和 GPU 资源在此分配
        // ----------------------------------------------------------------
        bool init(const SCBCTParams& params, int Kchunk, cudaStream_t stream)
        {
            Kchunk_ = Kchunk;
            bParker_ = params.bShortScan;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;

            // PreweightProcessor
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] PreweightProcessor init failed\n");
                    return false;
                }
            }

            // ParkerWeightProcessor
            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] ParkerWeightProcessor init failed\n");
                    return false;
                }
            }

            // FilterProcessor（FFT plan 在此创建，只做一次）
            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    fprintf(stderr, "[FdkReconstructor] FilterProcessor init failed\n");
                    return false;
                }
            }

            // BpProcessor
            {
                SVolGeom vol_geom = SVolGeom::make_centered(
                    iVX, iVY, iVZ, params.vox_xy_mm, params.vox_z_mm);
                vol_geom.center = make_float3(
                    params.vol_offset_x_mm,
                    params.vol_offset_y_mm,
                    params.vol_offset_z_mm);

                BpInitContext ictx{};
                ictx.vol_geom = vol_geom;
                ictx.use_precomputed = false;
                bp_.setInitContext(&ictx);
                if (!bp_.init()) {
                    fprintf(stderr, "[FdkReconstructor] BpProcessor init failed\n");
                    return false;
                }
            }

            is_initialized_ = true;
            return true;
        }

        bool reinitProcessors(const SCBCTParams& params, int K, cudaStream_t stream)
        {
            // PreweightProcessor
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] PreweightProcessor reinit failed\n");
                    return false;
                }
            }

            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] ParkerWeightProcessor reinit failed\n");
                    return false;
                }
            }

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    fprintf(stderr, "[FdkReconstructor] FilterProcessor reinit failed\n");
                    return false;
                }
            }

            return true;
        }

        // ----------------------------------------------------------------
        // feed（无 dump）
        // ----------------------------------------------------------------
        bool feed(
            const float* h_proj_batch,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol = false)
        {
            return feed_impl(h_proj_batch, params, Kchunk_, stream,
                d_vol_out, clear_vol, {});
        }

        // ----------------------------------------------------------------
        // feed（有 dump）
        // ----------------------------------------------------------------
        bool feed(
            const float* h_proj_batch,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol,
            std::function<void(int, const char*, float*, size_t)> onDump)
        {
            return feed_impl(h_proj_batch, params, Kchunk_, stream,
                d_vol_out, clear_vol, onDump);
        }

    private:
        int  Kchunk_ = 0;
        bool bParker_ = false;
        bool is_initialized_ = false;

        int                total_received_ = 0;
        std::vector<float> angle_accum_;

        // processor 持久化，init() 后常驻
        Fdk::PreweightProcessor    pw_;
        Fdk::ParkerWeightProcessor pkw_;
        Fdk::FilterProcessor       flt_;
        Fdk::BpProcessor           bp_;

        // ----------------------------------------------------------------
        // feed_impl
        // ----------------------------------------------------------------
        bool feed_impl(
            const float* h_proj_batch,
            const SCBCTParams& params,
            int                Kchunk,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol,
            std::function<void(int, const char*, float*, size_t)> onDump)
        {
            if (!is_initialized_) {
                fprintf(stderr, "[FdkReconstructor] not initialized, call init() first\n");
                return false;
            }
            if (Kchunk > kMaxChunkAng) {
                fprintf(stderr, "[FdkReconstructor] Kchunk=%d exceeds kMaxChunkAng=%d\n",
                    Kchunk, kMaxChunkAng);
                return false;
            }
            if (!h_proj_batch || params.iPAng <= 0) {
                fprintf(stderr, "[FdkReconstructor] invalid input\n");
                return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                fprintf(stderr, "[FdkReconstructor] angle_list size mismatch\n");
                return false;
            }

            const int batch_count = params.iPAng;
            const int prev_total = total_received_;

            angle_accum_.insert(angle_accum_.end(),
                params.angle_list.begin(), params.angle_list.end());
            total_received_ += batch_count;
            const int new_total = total_received_;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            std::vector<SConeProjGeomVec>    h_geo_full(new_total);
            std::vector<SFDKGeoParamPerView> h_gv_full(new_total);

            build_circular_vec_geometry_from_theta(
                h_geo_full, angle_accum_, new_total, iPU, iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));

            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, params.scan_range_rad, h_geo_full, h_gv_full);

            std::vector<SConeProjGeomVec>    h_geo(
                h_geo_full.begin() + prev_total, h_geo_full.end());
            std::vector<SFDKGeoParamPerView> h_gv(
                h_gv_full.begin() + prev_total, h_gv_full.end());

            if (clear_vol) {
                const size_t n = (size_t)iVX * iVY * iVZ;
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
            }

            SProjDims dims{ iPU, iPV, batch_count };
            FdkGpuContext ctx;
            ctx.init(dims, h_geo, h_gv, Kchunk, stream);

            const size_t view_elems = (size_t)iPU * iPV;
            float* d_chunk_in = ctx.proj.chunk_in.data();
            float* d_chunk_pw = ctx.proj.chunk_pw.data();
            float* d_chunk_flt = ctx.proj.chunk_flt.data();

            bool filter_dirty = false;

            for (int base = 0; base < batch_count; base += Kchunk) {
                const int K = std::min(Kchunk, batch_count - base);

                // 尾包时重建所有 processor
                if (K != Kchunk_) {
                    if (!reinitProcessors(params, K, stream)) return false;
                    filter_dirty = true;
                }

                ctx.geo.uploadCoeffsChunk(ctx.geo.d_coeffs() + base, K, stream);
                ctx.proj.uploadProjChunk(
                    h_proj_batch + (size_t)base * view_elems, K, stream);

                PreweightChunkContext pctx{};
                pctx.d_geo = ctx.geo.d_geo() + base;
                pctx.d_gv = ctx.geo.d_gv() + base;
                pctx.K = K;
                pw_.setContext(&pctx);
                pw_.process(d_chunk_in, d_chunk_pw, stream);

                if (onDump)
                    for (int i = 0; i < K; ++i)
                        onDump(base + i, "pw",
                            d_chunk_pw + i * view_elems, view_elems);

                if (bParker_) {
                    ParkerWeightChunkContext pkctx{};
                    pkctx.h_angles = params.angle_list.data() + base;
                    pkctx.K = K;
                    pkw_.setContext(&pkctx);
                    pkw_.process(d_chunk_pw, d_chunk_pw, stream);

                    if (onDump)
                        for (int i = 0; i < K; ++i)
                            onDump(base + i, "parker",
                                d_chunk_pw + i * view_elems, view_elems);
                }

                FdkFilterContext fctx{ h_gv.data() + base, K };
                flt_.setContext(&fctx);
                flt_.process(d_chunk_pw, d_chunk_flt, stream);

                if (onDump)
                    for (int i = 0; i < K; ++i)
                        onDump(base + i, "flt",
                            d_chunk_flt + i * view_elems, view_elems);

                BpChunkContext bctx{};
                bctx.d_geo = ctx.geo.d_geo() + base;
                bctx.d_gv = ctx.geo.d_gv() + base;
                bctx.K = K;
                bp_.setContext(&bctx);
                bp_.process(ctx.proj.d_texObjs(), d_vol_out, stream);
            }


            // 恢复标准尺寸，供下次 feed 使用
            if (filter_dirty) {
                if (!reinitProcessors(params, Kchunk_, stream)) return false;
            }

            return true;
        }
    };

    // ================================================================
    // 全局便捷函数（单次全量重建）
    // ================================================================
    YK_INLINE bool fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        bool               clear_vol = true,
        std::function<void(int, const char*, float*, size_t)> onDump = nullptr)
    {
        FdkReconstructor recon;
        if (!recon.init(params, Kchunk, stream))
            return false;
        return recon.feed(h_proj, params, stream,
            d_vol_out, clear_vol, onDump);
    }

} // namespace YK