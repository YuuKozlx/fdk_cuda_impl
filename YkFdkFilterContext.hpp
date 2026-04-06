#pragma once
#include "YkFDKCreateFilterKernel.hpp"
#include "YkGlobals.h"
#include "YkVecGeo.hpp"

namespace YK {

    // 初始化阶段配置
    struct FdkFilterInitContext {
        SProjDims       dims;           // dims.iPAng = Kchunk
        SFilterKernelDesc    desc;
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
        SProjDims      dims = {};
        SKernelLaunchPolicy policy = {};
    };

    // setContext 注入（每 chunk 前调用）
    struct PreweightChunkContext {
        const SConeProjectionVec* d_geo = nullptr;  // device，已偏移到 chunk 起始
        const SFDKGeoParamPerView* d_gv = nullptr;  // device，已偏移到 chunk 起始
        int                        K = 0;         // 当前 chunk 实际视角数
    };



    // ----------------------------------------------------------------
    // Init / Chunk context
    // ----------------------------------------------------------------
    struct BpInitContext {
        SVolumeGeometry vol_geom;
        bool use_precomputed = true;  // true: 预计算版本，false: 非预计算版本
    };

    struct BpChunkContext {
        const cudaTextureObject_t* d_texObjs = nullptr;
        const SConeProjectionVec* d_geo = nullptr;  // 非预计算版本需要
        const SFDKGeoParamPerView* d_gv = nullptr;  // 非预计算版本需要
        float* d_vol = nullptr;
        int                        K = 0;
    };



} // namespace YK