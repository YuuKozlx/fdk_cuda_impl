#pragma once
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"              // CUDA_PI, kMaxChunkAng
#include "../../global/YkFdkKernelTypes.hpp"
#include "../../global/YkMacro.hpp"              // YK_CUDA_CHECK, YK_CUDA_KERNEL_CHECK
#include "YkFDKParkerHelpers.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // pk_uploadAngles
            //   将本 chunk 的绝对角度转换为相对角度（归一化到 [0, 2π)）
            //   并上传到 constant memory gC_parker_angle。
            //
            //   注意：cudaMemcpyToSymbol 是同步调用，无需显式 stream 参数。
            // ----------------------------------------------------------------
            void pk_uploadAngles(
                const float* h_angles,
                int          K,
                float        fAngleBase,
                int          nDirSign);


            // ----------------------------------------------------------------
            // pk_launchParker
            //   原地 Parker 加权：d_data [K, Nv, Nu] device buffer。
            // ----------------------------------------------------------------
            void pk_launchParker(
                float* d_data,
                int Nu, int Nv, int K,
                const SConeProjGeomVec* d_geometry,
                const SFDKGeoParamPerView* d_gv,
                float fScale,
                int nDirSign,
                cudaStream_t stream);


        }
    }
} // namespace YK::Fdk::detail
