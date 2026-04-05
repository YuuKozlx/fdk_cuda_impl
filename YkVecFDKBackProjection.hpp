#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cstdio>
#include <cuda_runtime_api.h>
#include <sstream>

#include <functional>
#include "IProcessor.hpp"
#include "YkFDKBackproject.cuh"
#include "YkFDKFilter.hpp"
#include "YkFDKFilter2.hpp"
#include "YkFDKGpuContext.hpp"
#include "YkFDKPrecompute.hpp"
#include "YkFDKVecAlignPadCrop.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFDKVecPreWeight.hpp"
#include "YkFDKVecPreWeight2.hpp"
#include "YkFdkFilterContext.hpp"
#include "YkGlobals.h"
#include "YkIoDump.hpp"
#include "YkSampling2D.hpp"
#include "YkUtil.hpp"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"

namespace YK {
    static void fdk_recon_impl(
        const float* h_proj,
        float* d_vol_out,
        const std::vector<SConeProjectionVec>& h_geo,
        SDimensions3D dims, float vox,
        int Kchunk, cudaStream_t stream,
        std::function<void(int a, const char* tag, float* d_buf, size_t n)> onDump = {})
    {
        // ---- 1. 推导 per-view 参数 ----
        std::vector<SFDKGeoParamPerView> h_gv(dims.iPAng);
        GeoDerivedManagerVec{}.build_geo_params(dims.iPU, dims.iPV, h_geo, h_gv);

        // ---- 2. 管线工具 ----
        PreweightProcessor pw;
        {
            PreweightInitContext ictx{};
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
            SDimensions3D chunk_dims = dims;
            chunk_dims.iPAng = Kchunk;

            FdkFilterInitContext ictx{};
            ictx.dims = chunk_dims;
            ictx.desc = {};
            ictx.policy = {};
            ictx.stream = stream;

            IProcessor* proc = &fp;
            proc->setInitContext(&ictx);
            if (!proc->init()) {
                std::fprintf(stderr, "[fdk_recon_impl] FilterProcessor init failed.\n");
                return;
            }
        }

        // ---- 3. GPU 资源 ----
        // ctx.init 内部已完成 geo/gv 上传、d_vol 清零、launchPrecomputeCoeffs
        FdkGpuContext ctx;
        ctx.init(dims, h_geo, h_gv, Kchunk, stream);

        // h_coeffs 用于每 chunk 上传到 constant 内存
        std::vector<FdkAffineCoeff> h_coeffs(dims.iPAng);
        YK_CUDA_CHECK(cudaMemcpyAsync(h_coeffs.data(), ctx.d_coeffs,
            dims.iPAng * sizeof(FdkAffineCoeff),
            cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // ---- 4. 主循环 ----
        const size_t view_elems = (size_t)dims.iPU * dims.iPV;

        IProcessor* pw_proc = &pw;
        IProcessor* fp_proc = &fp;

        for (int base = 0; base < dims.iPAng; base += Kchunk) {
            const int K = std::min(Kchunk, (int)dims.iPAng - base);

            // constant 内存：当前 chunk 系数
            YK_CUDA_CHECK(cudaMemcpyToSymbol(gC_coeffs,
                h_coeffs.data() + base,
                K * sizeof(FdkAffineCoeff)));

            // H2D：整个 chunk 一次上传
            YK_CUDA_CHECK(cudaMemcpyAsync(
                ctx.d_chunk_in,
                h_proj + (size_t)base * view_elems,
                (size_t)K * view_elems * sizeof(float),
                cudaMemcpyHostToDevice, stream));

            // preweight
            PreweightChunkContext pctx{};
            pctx.d_geo = ctx.d_geo + base;
            pctx.d_gv = ctx.d_gv + base;
            pctx.K = K;
            pw_proc->setContext(&pctx);
            pw_proc->process(ctx.d_chunk_in, ctx.d_chunk_pw, stream);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(base + i, "pw",
                        ctx.d_chunk_pw + i * view_elems, view_elems);

            // 滤波：全程虚接口
            FdkFilterContext fctx{ h_gv.data() + base, K };
            fp_proc->setContext(&fctx);
            fp_proc->process(ctx.d_chunk_pw, ctx.d_chunk_flt, stream);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(base + i, "flt",
                        ctx.d_chunk_flt + i * view_elems, view_elems);

            // chunk flt → texture slots
            for (int i = 0; i < K; ++i)
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    ctx.chunk.slotPtr(i),
                    ctx.d_chunk_flt + i * view_elems,
                    view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));

            ctx.chunk.uploadTexObjs(K, stream);

            launchBpKernel(
                ctx.chunk.d_texObjs,
                ctx.d_geo + base,
                ctx.d_gv + base,
                ctx.d_vol,
                dims.iVX, dims.iVY, dims.iVZ, vox,
                K, stream);
        }

        // ---- 5. 输出 ----
        if (d_vol_out && d_vol_out != ctx.d_vol) {
            const size_t n = (size_t)dims.iVX * dims.iVY * dims.iVZ;
            YK_CUDA_CHECK(cudaMemcpyAsync(d_vol_out, ctx.d_vol,
                n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        // ctx / fp / pw 析构时自动释放所有资源
    }
    // ============================================================
    // Streaming recon (minimal fixes + use derived offsets correctly)
    // ============================================================
    //static void fdk_recon_impl(
    //    const float* h_proj,
    //    float* d_vol_out,
    //    const std::vector<SConeProjectionVec>& h_geo,
    //    SDimensions3D dims, float vox,
    //    int Kchunk, cudaStream_t stream,
    //    std::function<void(int a, const char* tag, float* d_buf, size_t n)> onDump = {})
    //{
    //    // ---- 1. 推导 per-view 参数 ----
    //    std::vector<SFDKGeoParamPerView> h_gv(dims.iPAng);
    //    GeoDerivedManagerVec{}.build_geo_params(dims.iPU, dims.iPV, h_geo, h_gv);

    //    // ---- 2. 管线工具 ----
    //    AlignPadCropManagerVec align;
    //    align.init(dims.iPU, dims.iPV, 1, stream);
    //    const int paddedN = align.paddedN();

    //    FilterManager fm;
    //    fm.init(dims.iPU, paddedN, h_gv[0].du_mm, dims.iPV, stream);

    //    PreweightManagerVec pw;

    //    // ---- 3. GPU 资源 ----
    //    FdkGpuContext ctx;
    //    ctx.init(dims, h_geo, h_gv, Kchunk, paddedN, stream);

    //    // 系数缓冲
    //    FdkAffineCoeff* d_coeffs = nullptr;
    //    YK_CUDA_CHECK(cudaMalloc(&d_coeffs, dims.iPAng * sizeof(FdkAffineCoeff)));

    //    // 一次性预计算所有角度的系数
    //    launchPrecomputeCoeffs(ctx.d_geo, ctx.d_gv, d_coeffs, dims.iPAng, stream);

    //    // host 侧系数，用于上传到 constant
    //    std::vector<FdkAffineCoeff> h_coeffs(dims.iPAng);
    //    YK_CUDA_CHECK(cudaMemcpyAsync(h_coeffs.data(), d_coeffs,
    //        dims.iPAng * sizeof(FdkAffineCoeff),
    //        cudaMemcpyDeviceToHost, stream));
    //    cudaStreamSynchronize(stream); // 等系数 D2H 完成

    //    // ---- 4. 主循环 ----
    //    const size_t view_elems = (size_t)dims.iPU * dims.iPV;

    //    for (int base = 0; base < dims.iPAng; base += Kchunk) {
    //        const int K = std::min(Kchunk, (int)dims.iPAng - base);

    //        // 上传当前 chunk 系数到 constant 内存
    //        YK_CUDA_CHECK(cudaMemcpyToSymbol(gC_coeffs,
    //            h_coeffs.data() + base,
    //            K * sizeof(FdkAffineCoeff)));

    //        // view 处理
    //        for (int i = 0; i < K; ++i) {
    //            const int a = base + i;

    //            YK_CUDA_CHECK(cudaMemcpyAsync(ctx.d_view_in,
    //                h_proj + a * view_elems,
    //                view_elems * sizeof(float),
    //                cudaMemcpyHostToDevice, stream));

    //            pw.applyChunk(dims, ctx.d_view_in, ctx.d_view_pw,
    //                ctx.d_geo + a, ctx.d_gv + a, 1, stream);

    //            if (onDump) onDump(a, "pw", ctx.d_view_pw, view_elems);

    //            align.padChunk(ctx.d_view_pw, ctx.d_padded,
    //                std::vector<float>{h_gv[a].offsetU_pix});
    //            fm.setStream(stream);
    //            fm.apply(ctx.d_padded);
    //            align.cropChunk(ctx.d_padded, ctx.d_view_flt);

    //            if (onDump) onDump(a, "flt", ctx.d_view_flt, view_elems);

    //            YK_CUDA_CHECK(cudaMemcpyAsync(
    //                ctx.chunk.slotPtr(i), ctx.d_view_flt,
    //                view_elems * sizeof(float),
    //                cudaMemcpyDeviceToDevice, stream));
    //        }

    //        ctx.chunk.uploadTexObjs(K, stream);

    //        // BP，gC_coeffs 已是当前 chunk 偏移，base_a 不需要传
    //        //launchBpKernel(
    //        //    ctx.chunk.d_texObjs, ctx.d_vol,
    //        //    dims.iVX, dims.iVY, dims.iVZ, vox,
    //        //    K, stream);

    //        // 非预计算版本 
    //        launchBpKernel(
    //            ctx.chunk.d_texObjs,
    //            ctx.d_geo + base,   // 偏移到当前 chunk 起始
    //            ctx.d_gv + base,
    //            ctx.d_vol,
    //            dims.iVX, dims.iVY, dims.iVZ, vox,
    //            K, stream);


    //    }

    //    // ---- 5. 输出 ----
    //    if (d_vol_out && d_vol_out != ctx.d_vol) {
    //        const size_t n = (size_t)dims.iVX * dims.iVY * dims.iVZ;
    //        YK_CUDA_CHECK(cudaMemcpyAsync(d_vol_out, ctx.d_vol,
    //            n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    //    }

    //    YK_CUDA_CHECK(cudaFree(d_coeffs));
    //    // ctx 析构时自动 cudaFree 所有资源
    //}


    // 无 dump 版本
    inline void fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const std::vector<SConeProjectionVec>& h_geo,
        SDimensions3D dims, float vox,
        int Kchunk, cudaStream_t stream)
    {
        fdk_recon_impl(h_proj, d_vol_out, h_geo, dims, vox, Kchunk, stream, nullptr);
    }

    // 有 dump 版本
    using FdkDumpFn = void(*)(int a, const char* tag, float* d_buf, size_t n);

    inline void fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const std::vector<SConeProjectionVec>& h_geo,
        SDimensions3D dims, float vox,
        int Kchunk, cudaStream_t stream,
        FdkDumpFn onDump)
    {
        fdk_recon_impl(h_proj, d_vol_out, h_geo, dims, vox, Kchunk, stream, onDump);
    }

} // namespace YK
