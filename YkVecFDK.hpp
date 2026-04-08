#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>
#include <functional>
#include <vector_functions.hpp>
#include <vector_types.h>
#include "IProcessor.hpp"
#include "YkBackProject.hpp"
#include "YkFDKFilterProcessor.hpp"
#include "YkFDKGpuContext.hpp"
#include "YkFDKPreWeightProcessor.hpp"
#include "YkFDKPrecompute.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFdkFilterContext.hpp"
#include "YkGlobals.h"
#include "YkIoDump.hpp"
#include "YkUtil.hpp"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"
#include "YkParams.h"

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

        int totalReceived() const { return total_received_; }

        // 无 dump
        bool feed(
            const float* h_proj_batch,
            const SCBCTParams& params,
            int                Kchunk,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol = false)
        {
            return feed_impl(h_proj_batch, params, Kchunk, stream,
                d_vol_out, clear_vol, {});
        }

        // 有 dump
        bool feed(
            const float* h_proj_batch,
            const SCBCTParams& params,
            int                Kchunk,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol,
            std::function<void(int, const char*, float*, size_t)> onDump)
        {
            return feed_impl(h_proj_batch, params, Kchunk, stream,
                d_vol_out, clear_vol, onDump);
        }

    private:
        int                total_received_ = 0;
        std::vector<float> angle_accum_;

        bool feed_impl(
            const float* h_proj_batch,
            const SCBCTParams& params,
            int                Kchunk,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol,
            std::function<void(int, const char*, float*, size_t)> onDump)
        {
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

            // 累积角度
            angle_accum_.insert(angle_accum_.end(),
                params.angle_list.begin(), params.angle_list.end());
            total_received_ += batch_count;
            const int new_total = total_received_;

            const int   iPU = params.iPU;
            const int   iPV = params.iPV;
            const int   iVX = params.iVX;
            const int   iVY = params.iVY;
            const int   iVZ = params.iVZ;

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            // 全量 geo/gv，保证 dtheta 全局一致
            std::vector<SConeProjGeomVec>    h_geo_full(new_total);
            std::vector<SFDKGeoParamPerView> h_gv_full(new_total);

            build_circular_vec_geometry_from_theta(
                h_geo_full, angle_accum_, new_total, iPU, iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.slant_angle_rad),
                    rad2deg(params.skew_angle_rad),
                    rad2deg(params.tilt_angle_rad)));

            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, params.scan_range_rad, h_geo_full, h_gv_full);

            // 截取本批切片
            std::vector<SConeProjGeomVec>    h_geo(
                h_geo_full.begin() + prev_total, h_geo_full.end());
            std::vector<SFDKGeoParamPerView> h_gv(
                h_gv_full.begin() + prev_total, h_gv_full.end());

            // 初始化处理器
            PreweightProcessor pw;
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.policy = {};
                IProcessor* proc = &pw;
                proc->setInitContext(&ictx);
                if (!pw.init()) {
                    fprintf(stderr, "[FdkReconstructor] PreweightProcessor init failed\n");
                    return false;
                }
            }

            ParkerWeightProcessor pkw;
            const bool bParker = params.bShortScan && batch_count > 1;
            if (bParker) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                IProcessor* proc = &pkw;
                proc->setInitContext(&ictx);
                if (!pkw.init()) {
                    fprintf(stderr, "[FdkReconstructor] ParkerWeightProcessor init failed\n");
                    return false;
                }
            }

            FilterProcessor flt;
            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                IProcessor* proc = &flt;
                proc->setInitContext(&ictx);
                if (!flt.init()) {
                    fprintf(stderr, "[FdkReconstructor] FilterProcessor init failed\n");
                    return false;
                }
            }

            SVolGeom vol_geom = SVolGeom::make_centered(
                iVX, iVY, iVZ, params.vox_xy_mm, params.vox_z_mm);
            vol_geom.center = make_float3(
                params.vol_offset_x_mm,
                params.vol_offset_y_mm,
                params.vol_offset_z_mm);

            BpProcessor bp;
            {
                BpInitContext ictx{};
                ictx.vol_geom = vol_geom;
                ictx.use_precomputed = true;
                IProcessor* proc = &bp;
                proc->setInitContext(&ictx);
                if (!bp.init()) {
                    fprintf(stderr, "[FdkReconstructor] BpProcessor init failed\n");
                    return false;
                }
            }

            // 清零 & GPU 资源
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

            IProcessor* pw_proc = &pw;
            IProcessor* pkw_proc = &pkw;
            IProcessor* fp_proc = &flt;
            IProcessor* bp_proc = &bp;

            for (int base = 0; base < batch_count; base += Kchunk) {
                const int K = std::min(Kchunk, batch_count - base);

                ctx.geo.uploadCoeffsChunk(ctx.geo.d_coeffs() + base, K, stream);
                ctx.proj.uploadProjChunk(
                    h_proj_batch + (size_t)base * view_elems, K, stream);

                PreweightChunkContext pctx{};
                pctx.d_geo = ctx.geo.d_geo() + base;
                pctx.d_gv = ctx.geo.d_gv() + base;
                pctx.K = K;
                pw_proc->setContext(&pctx);
                pw_proc->process(d_chunk_in, d_chunk_pw, stream);

                if (onDump)
                    for (int i = 0; i < K; ++i)
                        onDump(base + i, "pw",
                            d_chunk_pw + i * view_elems, view_elems);

                if (bParker) {
                    ParkerWeightChunkContext pkctx{};
                    pkctx.h_angles = params.angle_list.data() + base;  // 本批内偏移
                    pkctx.K = K;
                    pkw_proc->setContext(&pkctx);

                    pkw_proc->process(d_chunk_pw, d_chunk_pw, stream);

                    if (onDump)
                        for (int i = 0; i < K; ++i)
                            onDump(base + i, "parker",
                                d_chunk_pw + i * view_elems, view_elems);
                }

                FdkFilterContext fctx{ h_gv.data() + base, K };
                fp_proc->setContext(&fctx);
                fp_proc->process(d_chunk_pw, d_chunk_flt, stream);

                if (onDump)
                    for (int i = 0; i < K; ++i)
                        onDump(base + i, "flt",
                            d_chunk_flt + i * view_elems, view_elems);

                BpChunkContext bctx{};
                bctx.d_geo = ctx.geo.d_geo() + base;
                bctx.d_gv = ctx.geo.d_gv() + base;
                bctx.K = K;
                bp_proc->setContext(&bctx);
                bp_proc->process(ctx.proj.d_texObjs(), d_vol_out, stream);
            }

            return true;
        }
    };

    // ================================================================
    // 全局便捷函数
    // ================================================================

    // 无 dump
    inline bool fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        bool               clear_vol = true)
    {
        FdkReconstructor recon;
        return recon.feed(h_proj, params, Kchunk, stream, d_vol_out, clear_vol);
    }

    // 有 dump
    inline bool fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        bool               clear_vol,
        std::function<void(int, const char*, float*, size_t)> onDump)
    {
        FdkReconstructor recon;
        return recon.feed(h_proj, params, Kchunk, stream,
            d_vol_out, clear_vol, onDump);
    }

} // namespace YK