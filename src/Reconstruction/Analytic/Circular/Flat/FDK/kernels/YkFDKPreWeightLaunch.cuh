#pragma once
#include <cuda_runtime.h>

#include "common/YkVecGeo.hpp"
#include "global/YkKernelLaunchPolicy.hpp"


#include "Reconstruction/Analytic/Circular/Flat/FDK/kernels/YkFDKPreWeightHelpers.cuh"
#include "global/YkMacro.hpp"
namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // pw_launchPreweight
            //   启动锥束余弦预加权 kernel。
            //   d_src / d_dst 均为 [K, Nv, Nu] device buffer（支持原地）。
            // ----------------------------------------------------------------
            void pw_launchPreweight(
                const float* d_src,
                float* d_dst,
                const SConeProjGeomVec* d_geo,
                const SFDKGeoParamPerView* d_gv,
                int Nu, int Nv, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream);


        }
    }
} // namespace YK::Fdk::detail
