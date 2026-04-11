#pragma once
#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"
#include "../../global/YkMacro.hpp"              // YK_CUDA_KERNEL_CHECK
#include "../YkFDKVecGeoDerived.hpp"   // FdkAffineCoeff, SVolGeom
#include "../YkVecGeo.hpp"



namespace YK {
    namespace Fdk {


        void bp_launchBpPrecomputed(
            const cudaTextureObject_t* d_texObjs,
            float* d_vol,
            const SVolGeom& vol_geom,
            int K,
            cudaStream_t stream);


        // ----------------------------------------------------------------
        // bp_launchBpDirect
        //   反投影（非预计算版本）：kernel 内实时计算投影坐标。
        // ----------------------------------------------------------------
        void bp_launchBpDirect(
            const cudaTextureObject_t* d_texObjs,
            const SConeProjGeomVec* d_geo,
            const SFDKGeoParamPerView* d_gv,
            float* d_vol,
            const SVolGeom& vol_geom,
            int K,
            cudaStream_t stream);

    }
} // namespace YK::Fdk::detail
