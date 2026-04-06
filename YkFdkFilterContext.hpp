#pragma once
#include "YkVecGeo.hpp"
#include "YkFDKCreateFilterKernel.hpp"

namespace YK {

    // 初始化阶段配置
    struct FdkFilterInitContext {
        SDimensions3D       dims;           // dims.iPAng = Kchunk
        FilterKernelDesc    desc;
        SKernelLaunchPolicy policy = {};
        cudaStream_t        stream = 0;
    };


    struct FdkFilterContext {
        const SFDKGeoParamPerView* h_gv = nullptr;  // host pointer，当前 chunk 起始
        int                        K = 0;         // 当前 chunk 实际视角数
    };



    // ============================================================
    // Context 结构体
    // ============================================================

    // setInitContext 注入
    struct PreweightInitContext {
        SDimensions3D       dims = {};
        SKernelLaunchPolicy policy = {};
    };

    // setContext 注入（每 chunk 前调用）
    struct PreweightChunkContext {
        const SConeProjectionVec* d_geo = nullptr;  // device，已偏移到 chunk 起始
        const SFDKGeoParamPerView* d_gv = nullptr;  // device，已偏移到 chunk 起始
        int                        K = 0;         // 当前 chunk 实际视角数
    };



    //    // ----------------------------------------------------------------
    //// Init context（对应 PreweightInitContext 的角色）
    //// ----------------------------------------------------------------
    //    struct BpInitContext {
    //        SDimensions3D dims;
    //        float         vox;
    //    };
    //
    //    // ----------------------------------------------------------------
    //    // Chunk context（对应 PreweightChunkContext 的角色）
    //    // ----------------------------------------------------------------
    //    struct BpChunkContext {
    //        const cudaTextureObject_t* d_texObjs;  // chunk 内各帧 texture
    //        const SConeProjectionVec* d_geo;      // ctx.d_geo + base
    //        const SFDKGeoParamPerView* d_gv;       // ctx.d_gv  + base
    //        float* d_vol;      // 外部 volume buffer
    //        int                        K;
    //    };


        // ----------------------------------------------------------------
    // Init context（对应 PreweightInitContext 的角色）
    // ----------------------------------------------------------------
    struct BpInitContext {
        SDimensions3D dims;
        float         vox;
    };

    // ----------------------------------------------------------------
    // Chunk context（对应 PreweightChunkContext 的角色）
    // ----------------------------------------------------------------
    struct BpChunkContext {
        const cudaTextureObject_t* d_texObjs;  // chunk 内各帧 texture
        const SConeProjectionVec* d_geo;      // ctx.d_geo + base
        const SFDKGeoParamPerView* d_gv;       // ctx.d_gv  + base
        float* d_vol;      // 外部 volume buffer
        int                        K;
    };

} // namespace YK