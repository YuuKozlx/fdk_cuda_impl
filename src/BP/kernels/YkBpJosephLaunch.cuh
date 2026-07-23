// YkBPJosephLaunch.cuh
#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "../../common/YkVecGeo.hpp"
#include "../../global/YkGlobals.h"
#include "../YkBpCommon.cuh"

namespace YK {
    namespace Bp {

        void joseph_bp_launch(
            const float* d_sino,
            const std::vector<SConeProjGeomVec>& h_views,
            const SConeProjGeomVec* d_views_vox,
            float* d_vol,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            BpStepSuperSample ss = BpStepSuperSample::x1);  // ← 默认 x1，与 FP 一致

        void joseph_bp_launch(
            cudaTextureObject_t                  sinoTex,
            const std::vector<SConeProjGeomVec>& h_views,
            const SConeProjGeomVec* d_views_vox,
            float* d_vol,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            BpStepSuperSample ss = BpStepSuperSample::x1);
        void joseph_bp_v2_launch(
            cudaTextureObject_t                  sinoTex,
            const SConeProjGeomVec* d_views_world,
            float* d_vol,
            const SVolGeom& vg,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream
        );
        void joseph_bp_v3_launch(
            cudaTextureObject_t      sinoTex,
            const SConeProjGeomVec* d_views,
            const FdkAffineCoeff* d_coeffs,
            float* d_vol,
            const SVolGeom& vg,
            int Na,
            bool accumulate,
            cudaStream_t stream);





    }; // namespace Bp
}; // namespace YK