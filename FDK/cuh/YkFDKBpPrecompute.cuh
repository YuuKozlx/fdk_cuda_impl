#pragma once
#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"
#include "../../global/YkMacro.hpp"
#include "../YkFDKVecGeoDerived.hpp"
#include "../YkFdkPipelineContext.hpp"   // FdkAffineCoeff
#include "../YkVecGeo.hpp"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // bp_launchPrecomputeCoeffs
            // ----------------------------------------------------------------
            void bp_launchPrecomputeCoeffs(
                const SConeProjGeomVec* d_geo,
                const SFDKGeoParamPerView* d_gv,
                FdkAffineCoeff* d_coeffs,
                int Ang, cudaStream_t stream);


        }
    }
} // namespace YK::Fdk::detail
