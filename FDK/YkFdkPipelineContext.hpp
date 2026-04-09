#pragma once
#include "Filter/YkFilterKernel.cuh"
#include "global/YkGlobals.h"
#include "FDK/YkVecGeo.hpp"

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
        int          iPAnglesTotal = 0; // 整个扫描的总视角数（非 chunk 内），用于计算相对角度
        float        fScanRangeRad = 2.f * CUDA_PI; // 扫描范围（弧度），短扫描时 < 2Pi
        float        fStartAngleRad;     // 全局起始角度
    };

    struct ParkerWeightChunkContext {
        const float* h_angles = nullptr;  // 本 chunk 的角度列表，大小 = K
        int          K = 0;
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