#pragma once
#include "Filter/YkCreateFilterKernel.cuh"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"

namespace YK {

    // ��ʼ���׶�����
    struct FdkFilterInitContext {
        SProjDims       dims;           // dims.iPAng = Kchunk
        SFilterKernelDesc    desc;
        SKernelLaunchPolicy policy = {};
        cudaStream_t        stream = 0;
    };


    struct FdkFilterContext {
        const SFDKGeoParamPerView* h_gv = nullptr;  // host pointer����ǰ chunk ��ʼ
        int                        K = 0;         // ��ǰ chunk ʵ���ӽ���
    };

    // ============================================================
    // Context �ṹ��
    // ============================================================

    // setInitContext ע��
    struct PreweightInitContext {
        SProjDims      dims = {};
        SKernelLaunchPolicy policy = {};
    };

    // setContext ע�루ÿ chunk ǰ���ã�
    struct PreweightChunkContext {
        const SConeProjGeomVec* d_geo = nullptr;  // device����ƫ�Ƶ� chunk ��ʼ
        const SFDKGeoParamPerView* d_gv = nullptr;  // device����ƫ�Ƶ� chunk ��ʼ
        int                        K = 0;         // ��ǰ chunk ʵ���ӽ���
    };

    // ============================================================
// ParkerWeightInitContext / ParkerWeightChunkContext
// ============================================================
    struct ParkerWeightInitContext {
        SProjDims    dims = {};
        float        fDetUSize = 1.f;
        float        fSrcOrigin = 0.f;
        float        fDetOrigin = 0.f;
        int          iPAnglesTotal = 0; // ����ɨ������ӽ������� chunk �ڣ������ڼ�����ԽǶ�
        float        fScanRangeRad = 2.f * CUDA_PI; // ɨ�跶Χ�����ȣ�����ɨ��ʱ < 2Pi
        float        fStartAngleRad;     // ȫ����ʼ�Ƕ�
    };

    struct ParkerWeightChunkContext {
        const float* h_angles = nullptr;  // �� chunk �ĽǶ��б�����С = K
        int          K = 0;
    };

    // ----------------------------------------------------------------
    // Init / Chunk context
    // ----------------------------------------------------------------
    struct BpInitContext {
        SVolGeom vol_geom;
        bool use_precomputed = true;  // true: Ԥ����汾��false: ��Ԥ����汾
    };

    struct BpChunkContext {
        const SConeProjGeomVec* d_geo = nullptr;
        const SFDKGeoParamPerView* d_gv = nullptr;
        int                        K = 0;
    };



} // namespace YK