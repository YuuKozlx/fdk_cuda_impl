#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cstdio>
#include <cuda_runtime_api.h>
#include <sstream>

#include <driver_types.h>
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
#include "YkSampling2D.hpp"
#include "YkUtil.hpp"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"
#include "YkParams.h"

namespace YK {
    static void fdk_recon_impl(
        const float* h_proj,
        float* d_vol_out,
        SCBCTParams params,
        int Kchunk, cudaStream_t stream,
        bool clear_vol = true,
        std::function<void(int a, const char* tag, float* d_buf, size_t n)> onDump = {})
    {
        if (Kchunk > kMaxChunkAng) {
            fprintf(stderr, "K=%d exceeds kMaxChunkAng=%d\n", Kchunk, kMaxChunkAng);
            return;
        }

        const int   iPA = params.iPAng;
        const int   iPU = params.iPU;
        const int   iPV = params.iPV;
        const int   iVX = params.iVX;
        const int   iVY = params.iVY;
        const int   iVZ = params.iVZ;
        const float du_mm = params.du_mm;
        const float dv_mm = params.dv_mm;
        const float vox_xy = params.vox_xy_mm;
        const float vox_z = params.vox_z_mm;

        auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };
        const float skew_deg = rad2deg(params.skew_angle_rad);
        const float slant_deg = rad2deg(params.slant_angle_rad);
        const float tilt_deg = rad2deg(params.tilt_angle_rad);

        // ---- 1. 构建投影几何参数 ----
        std::vector<SConeProjGeomVec> h_geo(iPA);
        build_circular_vec_geometry_from_theta(
            h_geo, params.angle_list, iPA, iPU, iPV,
            du_mm, dv_mm, params.SID, params.SDD - params.SID,
            f3(params.offsetU_mm, params.offsetV_mm, 0.f),
            f3(slant_deg, skew_deg, tilt_deg));

        std::vector<SFDKGeoParamPerView> h_gv(iPA);
        GeoDerivedManagerVec{}.build_geo_params(iPU, iPV, params.ScanRange_rad, h_geo, h_gv);

        // ---- 2. 初始化处理器 ----
        PreweightProcessor pw;
        {
            PreweightInitContext ictx{};
            ictx.dims = SProjDims{ iPU, iPV, Kchunk };
            ictx.policy = {};
            IProcessor* proc = &pw;
            proc->setInitContext(&ictx);
            if (!pw.init()) {
                std::fprintf(stderr, "[fdk_recon_impl] PreweightProcessor init failed.\n");
                return;
            }
        }

        // Parker weighting（可选，短扫描时启用）
        ParkerWeightProcessor pkw;
        const bool bParker = params.bShortScan && iPA > 1;
        if (bParker) {
            ParkerWeightInitContext ictx{};
            ictx.dims = SProjDims{ iPU, iPV, Kchunk };
            ictx.fDetUSize = params.du_mm;
            ictx.fSrcOrigin = params.SID;
            ictx.fDetOrigin = params.SDD - params.SID;
            ictx.h_angles = params.angle_list.data();
            ictx.iPA = iPA;
            IProcessor* proc = &pkw;
            proc->setInitContext(&ictx);
            if (!pkw.init()) {
                std::fprintf(stderr, "[fdk_recon_impl] ParkerWeightProcessor init failed.\n");
                return;
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
                std::fprintf(stderr, "[fdk_recon_impl] FilterProcessor init failed.\n");
                return;
            }
        }

        // ---- 体积几何 ----
        SVolGeom vol_geom = SVolGeom::make_centered(iVX, iVY, iVZ, vox_xy, vox_z);
        vol_geom.center = make_float3(
            params.vol_offset_x_mm, params.vol_offset_y_mm, params.vol_offset_z_mm);

        BpProcessor bp;
        {
            BpInitContext ictx{};
            ictx.vol_geom = vol_geom;
            ictx.use_precomputed = true;
            IProcessor* proc = &bp;
            proc->setInitContext(&ictx);
            if (!bp.init()) {
                std::fprintf(stderr, "[fdk_recon_impl] BpProcessor init failed.\n");
                return;
            }
        }

        // ---- 3. 清零体积 & 初始化 GPU 资源 ----
        if (clear_vol) {
            const size_t n = (size_t)iVX * iVY * iVZ;
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
        }

        SProjDims dims{ iPU, iPV, iPA };
        FdkGpuContext ctx;
        ctx.init(dims, h_geo, h_gv, Kchunk, stream);

        // ---- 4. 分块主循环 ----
        const size_t view_elems = (size_t)iPU * iPV;

        IProcessor* pw_proc = &pw;
        IProcessor* pkw_proc = &pkw;
        IProcessor* fp_proc = &flt;
        IProcessor* bp_proc = &bp;

        float* d_chunk_in = ctx.proj.chunk_in.data();
        float* d_chunk_pw = ctx.proj.chunk_pw.data();
        float* d_chunk_flt = ctx.proj.chunk_flt.data();

        for (int base = 0; base < iPA; base += Kchunk) {
            const int K = std::min(Kchunk, iPA - base);

            ctx.geo.uploadCoeffsChunk(ctx.geo.d_coeffs() + base, K, stream);
            ctx.proj.uploadProjChunk(h_proj + (size_t)base * view_elems, K, stream);

            // Preweight（cos 加权）
            PreweightChunkContext pctx{};
            pctx.d_geo = ctx.geo.d_geo() + base;
            pctx.d_gv = ctx.geo.d_gv() + base;
            pctx.K = K;
            pw_proc->setContext(&pctx);
            pw_proc->process(d_chunk_in, d_chunk_pw, stream);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(base + i, "pw", d_chunk_pw + i * view_elems, view_elems);

            // Parker weighting（in-place，短扫描时才执行）
            if (bParker) {
                ParkerWeightChunkContext pkctx{};
                pkctx.baseAngle = base;
                pkctx.K = K;
                pkw_proc->setContext(&pkctx);
                pkw_proc->process(d_chunk_pw, d_chunk_pw, stream);

                if (onDump)
                    for (int i = 0; i < K; ++i)
                        onDump(base + i, "parker", d_chunk_pw + i * view_elems, view_elems);
            }

            // Filter
            FdkFilterContext fctx{ h_gv.data() + base, K };
            fp_proc->setContext(&fctx);
            fp_proc->process(d_chunk_pw, d_chunk_flt, stream);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(base + i, "flt", d_chunk_flt + i * view_elems, view_elems);

            // Backproject
            BpChunkContext bctx{};
            bctx.d_geo = ctx.geo.d_geo() + base;
            bctx.d_gv = ctx.geo.d_gv() + base;
            bctx.K = K;
            bp_proc->setContext(&bctx);
            bp_proc->process(ctx.proj.d_texObjs(), d_vol_out, stream);
        }
    }


    // 无 dump 版本
    inline void fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        SCBCTParams params,
        int Kchunk, cudaStream_t stream,
        bool clear_vol = true) // ← 新增)
    {
        {
            YK::Util::CudaTimer t("fdk_recon_impl", stream);
            fdk_recon_impl(h_proj, d_vol_out, params, Kchunk, stream, clear_vol);
        }
    }

    // 有 dump 版本
    using FdkDumpFn = void(*)(int a, const char* tag, float* d_buf, size_t n);

    inline void fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        SCBCTParams params,
        int Kchunk, cudaStream_t stream,
        bool clear_vol,
        FdkDumpFn onDump)
    {
        fdk_recon_impl(h_proj, d_vol_out, params, Kchunk, stream, clear_vol, onDump);
    }

} // namespace YK
