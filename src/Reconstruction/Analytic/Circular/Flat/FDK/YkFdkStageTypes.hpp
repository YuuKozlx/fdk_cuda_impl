#pragma once
#include "Filter/YkCreateFilterKernel.cuh"
#include "global/YkGlobals.h"
#include "global/YkFdkKernelTypes.hpp"
#include "global/YkFilterTypes.hpp"
#include "global/YkKernelLaunchPolicy.hpp"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"

namespace YK {

    // 初始化阶段配置
    struct FdkFilterConfig {
        SProjDims       dims;           // dims.iPAng = Kchunk
        SFilterKernelDesc    desc;
        SKernelLaunchPolicy policy = {};
        cudaStream_t        stream = 0;
    };


    struct FdkFilterChunk {
        const SFDKGeoParamPerView* h_gv = nullptr;  // host pointer，当前 chunk 起始
        int                        K = 0;         // 当前 chunk 实际视角数
    };

    // ============================================================
    // Context 结构体
    // ============================================================

    struct PreweightConfig {
        SProjDims      dims = {};
        SKernelLaunchPolicy policy = {};
    };

    struct PreweightChunk {
        const SConeProjGeomVec* d_geo = nullptr;  // device，已偏移到 chunk 起始
        const SFDKGeoParamPerView* d_gv = nullptr;  // device，已偏移到 chunk 起始
        int                        K = 0;         // 当前 chunk 实际视角数
    };

    // ============================================================
// ParkerWeightConfig / ParkerWeightChunk
// ============================================================
    struct ParkerWeightConfig {
        SProjDims    dims = {};
        int          iPAnglesTotal = 0; // 整个扫描的总视角数（非 chunk 内），用于计算相对角度
        float        fScanRangeRad = 2.f * CUDA_PI; // 扫描范围（弧度），短扫描时 < 2Pi
        float        fStartAngleRad;     // 全局起始角度
        int          nDirSign = 1;          // 方向符号，1 或 -1，取决于角度增减方向，影响冗余区域定义
    };

    struct ParkerWeightChunk {
        const SConeProjGeomVec* h_geometry = nullptr; // 当前 chunk 的唯一角度来源
        const SConeProjGeomVec* d_geometry = nullptr; // 每像素真实扇角的几何来源
        const SFDKGeoParamPerView* d_gv = nullptr; // 每视图真实距离、主点和像素尺寸
        int          K = 0;
    };

    // ----------------------------------------------------------------
    // 静态配置 / 运行期 chunk
    // ----------------------------------------------------------------
    struct BpConfig {
        SVolGeom vol_geom;
        bool use_precomputed = true;  // true: 预计算版本，false: 非预计算版本
        int max_chunk_views = 0;      // Pipeline prepare 后固定的工作区上界
    };

    struct BpChunk {
        const SConeProjGeomVec* d_geo = nullptr;
        const SFDKGeoParamPerView* d_gv = nullptr;
        int                        K = 0;
    };



} // namespace YK
