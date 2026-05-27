#pragma once
#include <cuda_runtime.h>
#include "../../global/YkGlobals.h"

namespace YK {
    namespace Bp
    {
        // YkBPSiddonLaunch.cuh
        void bp_siddon_launch(
            const float* d_sino,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            cudaStream_t            stream);


        void bp_siddon_voxel_launch(
            const float* d_sino,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            cudaStream_t            stream);

        void bp_siddon_voxel_v2_launch(
            cudaTextureObject_t              sinoTex,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            bool accumulate,
            cudaStream_t stream);
    };


}