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
#include "YkBackProject.hpp"
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

        BpProcessor bp;
        {
            BpInitContext ictx{};
            ictx.dims = dims;
            ictx.vox = vox;
            IProcessor* proc = &bp;
            proc->setInitContext(&ictx);
            if (!proc->init()) {
                std::fprintf(stderr, "[fdk_recon_impl] BpProcessor init failed.\n");
                return;
            }
        }

        // ---- 3. GPU 资源 ----
        if (clear_vol) {
            const size_t n = (size_t)dims.iVX * dims.iVY * dims.iVZ;
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
        }

        FdkGpuContext ctx;
        ctx.init(dims, h_geo, h_gv, Kchunk, stream);

        std::vector<FdkAffineCoeff> h_coeffs(dims.iPAng);
        YK_CUDA_CHECK(cudaMemcpyAsync(h_coeffs.data(), ctx.d_coeffs,
            dims.iPAng * sizeof(FdkAffineCoeff),
            cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // ---- 4. 主循环 ----
        const size_t view_elems = (size_t)dims.iPU * dims.iPV;

        IProcessor* pw_proc = &pw;
        IProcessor* fp_proc = &fp;
        IProcessor* bp_proc = &bp;

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

            // 预计算版本，gC_Coeffs 通过 cudaMemcpyToSymbol 上传，kernel 内直接读取；launchBpKernel 里不再传入仿射系数数组；

            BpChunkContext bctx{};
            bctx.d_texObjs = ctx.chunk.d_texObjs;
            bctx.d_vol = d_vol_out;
            bctx.K = K;
            bp_proc->setContext(&bctx);
            bp_proc->process(nullptr, nullptr, stream);


            // 非预计算版本，kernel 内直接计算仿射系数，launchBpKernel 里不再传入预计算的仿射系数数组；
            // 循环内替换 launchBpKernel 直接调用
            //BpChunkContext bctx{};
            //bctx.d_texObjs = ctx.chunk.d_texObjs;
            //bctx.d_geo = ctx.d_geo + base;
            //bctx.d_gv = ctx.d_gv + base;
            //bctx.d_vol = d_vol_out;
            //bctx.K = K;
            //IProcessor* bp_proc = &bp;
            //bp_proc->setContext(&bctx);
            //bp_proc->process(nullptr, nullptr, stream);


        }
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
