#pragma once 
#include <driver_types.h>
#include <vector>
#include "../../global/YkGlobals.h"

namespace YK {

    namespace Fp {
        void fp_cvp_launch(
            const float* d_vol,
            float* d_sino,
            const SConeProjGeomVec* h_views,
            const YK::SVolGeom& g,
            int Na, int Nu, int Nv,
            cudaStream_t stream);

        void fp_cvp_launch(
            cudaTextureObject_t tex_vol,    // ¡û ÎÆÀí°æ±¾
            float* d_sino,
            const SConeProjGeomVec* h_views,
            const YK::SVolGeom& g,
            int Na, int Nu, int Nv,
            cudaStream_t stream);
    }
}