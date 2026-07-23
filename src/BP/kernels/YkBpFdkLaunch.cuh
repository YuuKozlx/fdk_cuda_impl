#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "../../common/YkVecGeo.hpp"
#include "../../global/YkGlobals.h"
#include "../YkBpCommon.cuh"


namespace YK {
    namespace Bp {

        void fdk_bp_launch(
            cudaTextureObject_t     sinoTex,
            const SConeProjGeomVec* d_views_world,
            const FdkAffineCoeff* d_coeffs,
            float* d_vol,
            const SVolGeom& vg,
            int Na,
            bool accumulate,
            cudaStream_t stream);

        void fdk_matched_bp_launch(
            cudaTextureObject_t     sinoTex,
            const SConeProjGeomVec* d_views_world,
            const FdkAffineCoeff* d_coeffs,
            float* d_vol,
            const SVolGeom& vg,
            int Na,
            bool accumulate,
            cudaStream_t stream);




    }; // namespace Bp
}; // namespace YK