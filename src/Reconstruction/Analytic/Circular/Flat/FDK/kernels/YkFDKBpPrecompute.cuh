#pragma once
#include <cuda_runtime.h>

#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKVecGeoDerived.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFdkStageTypes.hpp"   // FdkAffineCoeff
#include "common/YkVecGeo.hpp"

namespace YK {
    namespace Fdk {

        // ----------------------------------------------------------------
        // bp_launchPrecomputeCoeffs
        // ----------------------------------------------------------------
        void bp_launchPrecomputeCoeffs(
            const SConeProjGeomVec* d_geo,
            const SFDKGeoParamPerView* d_gv,
            FdkAffineCoeff* d_coeffs,
            int Ang, cudaStream_t stream);

    }
} // namespace YK::Fdk::detail
