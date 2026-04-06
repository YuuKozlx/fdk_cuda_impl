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
        const SConeProjGeomVec* d_geo = nullptr;  // device，已偏移到 chunk 起始
        const SFDKGeoParamPerView* d_gv = nullptr;  // device，已偏移到 chunk 起始
        int                        K = 0;         // 当前 chunk 实际视角数
    };

    // ============================================================
// ParkerWeightInitContext / ParkerWeightChunkContext
// ============================================================
    struct ParkerWeightInitContext {
        SProjDims    dims = {};
        float        fDetUSize = 1.f;
        float        fSrcOrigin = 0.f;
        float        fDetOrigin = 0.f;
        const float* h_angles = nullptr;
        int          iPA = 0;
    };

    struct ParkerWeightChunkContext {
        int baseAngle = 0;
        int K = 0;
    };

    // ----------------------------------------------------------------
    // Init / Chunk context
    // ----------------------------------------------------------------
    struct BpInitContext {
        SVolGeom vol_geom;
        bool use_precomputed = true;  // true: 预计算版本，false: 非预计算版本
    };

    struct BpChunkContext {
        const SConeProjGeomVec* d_geo = nullptr;
        const SFDKGeoParamPerView* d_gv = nullptr;
        int                        K = 0;
    };



} // namespace YK