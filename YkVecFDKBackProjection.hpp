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
#include "YkFDKFilterProcessor.hpp"
#include "YkFDKGpuContext.hpp"
#include "YkFDKPrecompute.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFdkFilterContext.hpp"
#include "YkGlobals.h"
#include "YkIoDump.hpp"
#include "YkSampling2D.hpp"
#include "YkUtil.hpp"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"
#include "YkFDKPreWeightProcessor.hpp"

namespace YK {
    static void fdk_recon_impl(
        const float* h_proj,
        float* d_vol_out,
        const std::vector<SConeProjectionVec>& h_geo,
        SDimensions3D dims, float vox,
        int Kchunk, cudaStream_t stream,
        bool clear_vol = true,
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
        if (clear_vol) {
            const size_t n = (size_t)dims.iVX * dims.iVY * dims.iVZ;
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
        }

        FdkGpuContext ctx;
        ctx.init(dims, h_geo, h_gv, Kchunk, stream);  // ctx.d_vol 不再使用，但 init 内部仍会分配

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

            // ↓ 改动1：直接写 d_vol_out，不再经过 ctx.d_vol 中转
            launchBpKernel(
                ctx.chunk.d_texObjs,
                ctx.d_geo + base,
                ctx.d_gv + base,
                d_vol_out,             // ← 原来是 ctx.d_vol
                dims.iVX, dims.iVY, dims.iVZ, vox,
                K, stream);
        }

        // ↓ 改动2：删除末尾的 d_vol_out != ctx.d_vol 拷贝，不再需要
    }


    // 无 dump 版本
    inline void fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const std::vector<SConeProjectionVec>& h_geo,
        SDimensions3D dims, float vox,
        int Kchunk, cudaStream_t stream,
        bool clear_vol = true) // ← 新增)
    {
        fdk_recon_impl(h_proj, d_vol_out, h_geo, dims, vox, Kchunk, stream, clear_vol);
    }

    // 有 dump 版本
    using FdkDumpFn = void(*)(int a, const char* tag, float* d_buf, size_t n);

    inline void fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const std::vector<SConeProjectionVec>& h_geo,
        SDimensions3D dims, float vox,
        int Kchunk, cudaStream_t stream, bool clear_vol,
        FdkDumpFn onDump)
    {
        fdk_recon_impl(h_proj, d_vol_out, h_geo, dims, vox, Kchunk, stream, clear_vol, onDump);
    }

} // namespace YK
