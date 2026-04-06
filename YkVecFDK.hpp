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

        int iPA = params.iPAng;
        int iPU = params.iPU;
        int iPV = params.iPV;
        int iVX = params.iVX;
        int iVY = params.iVY;
        int iVZ = params.iVZ;
        float du_mm = params.du_mm;
        float dv_mm = params.dv_mm;
        float SID = params.SID;
        float SDD = params.SDD;
        float vox_xy = params.vox_xy_mm;
        float vox_z = params.vox_z_mm;

        float offsetU_mm = params.offsetU_mm;
        float offsetV_mm = params.offsetV_mm;

        auto rad2deg = [](float rad) { return rad * 180.f / CUDA_PI; };

        float skew_deg = rad2deg(params.skew_angle_rad);
        float slant_deg = rad2deg(params.slant_angle_rad);
        float tilt_deg = rad2deg(params.tilt_angle_rad);


        SVolumeGeometry vol_geom = SVolumeGeometry::make_centered(
            iVX, iVY, iVZ,
            vox_xy, vox_z);
        vol_geom.center = make_float3(params.vol_offset_x_mm, params.vol_offset_y_mm, params.vol_offset_z_mm); // 可选：调整体积中心位置


        std::vector<float> angles = params.angle_list;
        float3 offset = f3(offsetU_mm, offsetV_mm, 0.0f);
        // ---- 0. 构建几何参数 ----
        std::vector<SConeProjectionVec> h_geo(iPA);
        build_circular_vec_geometry_from_theta(h_geo, angles, iPA, iPU, iPV, du_mm, dv_mm, SID, SDD - SID, f3(offsetU_mm, offsetV_mm, 0.0f), f3(slant_deg, skew_deg, tilt_deg));


        // ---- 1. 推导 per-view 参数 ----
        std::vector<SFDKGeoParamPerView> h_gv(iPA);
        GeoDerivedManagerVec{}.build_geo_params(iPU, iPV, h_geo, h_gv);

        // ---- 2. 管线工具 ----
        PreweightProcessor pw;
        {
            PreweightInitContext ictx{};
            SProjDims dims;
            dims.iPU = iPU;
            dims.iPV = iPV;
            dims.iPAng = Kchunk;
            ictx.dims = dims;
            ictx.policy = {};
            IProcessor* proc = &pw;
            proc->setInitContext(&ictx);
            if (!proc->init()) {
                std::fprintf(stderr, "[fdk_recon_impl] PreweightProcessor init failed.\n");
                return;
            }
        }

        FilterProcessor fp;
        {
            FdkFilterInitContext ictx{};
            SProjDims dims;
            dims.iPU = iPU;
            dims.iPV = iPV;
            dims.iPAng = Kchunk;
            ictx.dims = dims;
            ictx.desc = params.desc;
            ictx.policy = {};
            ictx.stream = stream;

            IProcessor* proc = &fp;
            proc->setInitContext(&ictx);
            if (!proc->init()) {
                std::fprintf(stderr, "[fdk_recon_impl] FilterProcessor init failed.\n");
                return;
            }
        }

        BpProcessor bp;
        {
            BpInitContext ictx{};
            ictx.vol_geom = vol_geom;
            ictx.use_precomputed = true;   // 或 false
            IProcessor* proc = &bp;
            proc->setInitContext(&ictx);
            if (!proc->init()) {
                std::fprintf(stderr, "[fdk_recon_impl] BpProcessor init failed.\n");
                return;
            }
        }



        // ---- 3. GPU 资源 ----
        if (clear_vol) {
            const size_t n = (size_t)iVX * iVY * iVZ;
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
        }

        FdkGpuContext ctx;
        SProjDims dims;
        dims.iPU = iPU;
        dims.iPV = iPV;
        dims.iPAng = iPA;
        ctx.init(dims, h_geo, h_gv, Kchunk, stream);

        std::vector<FdkAffineCoeff> h_coeffs(iPA);
        YK_CUDA_CHECK(cudaMemcpyAsync(h_coeffs.data(), ctx.d_coeffs,
            iPA * sizeof(FdkAffineCoeff),
            cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // ---- 4. 主循环 ----
        const size_t view_elems = (size_t)iPU * iPV;

        IProcessor* pw_proc = &pw;
        IProcessor* fp_proc = &fp;
        IProcessor* bp_proc = &bp;

        for (int base = 0; base < iPA; base += Kchunk) {
            const int K = std::min(Kchunk, (int)iPA - base);

            YK_CUDA_CHECK(cudaMemcpyToSymbol(gC_coeffs,
                h_coeffs.data() + base,
                K * sizeof(FdkAffineCoeff)));

            YK_CUDA_CHECK(cudaMemcpyAsync(
                ctx.d_chunk_in,
                h_proj + (size_t)base * view_elems,
                (size_t)K * view_elems * sizeof(float),
                cudaMemcpyHostToDevice, stream));

            PreweightChunkContext pctx{};
            pctx.d_geo = ctx.d_geo + base;
            pctx.d_gv = ctx.d_gv + base;
            pctx.K = K;
            pw_proc->setContext(&pctx);
            pw_proc->process(ctx.d_chunk_in, ctx.d_chunk_pw, stream);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(base + i, "pw", ctx.d_chunk_pw + i * view_elems, view_elems);

            FdkFilterContext fctx{ h_gv.data() + base, K };
            fp_proc->setContext(&fctx);
            fp_proc->process(ctx.d_chunk_pw, ctx.d_chunk_flt, stream);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(base + i, "flt", ctx.d_chunk_flt + i * view_elems, view_elems);

            for (int i = 0; i < K; ++i)
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    ctx.chunk.slotPtr(i),
                    ctx.d_chunk_flt + i * view_elems,
                    view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));

            ctx.chunk.uploadTexObjs(K, stream);


            // 循环内 setContext 时，非预计算版本需要额外传 d_geo / d_gv
            BpChunkContext bctx{};
            bctx.d_texObjs = ctx.chunk.d_texObjs;
            bctx.d_geo = ctx.d_geo + base;  // 非预计算版本用，预计算版本传 nullptr 也可
            bctx.d_gv = ctx.d_gv + base;
            bctx.d_vol = d_vol_out;
            bctx.K = K;
            bp_proc->setContext(&bctx);
            bp_proc->process(nullptr, nullptr, stream);


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
