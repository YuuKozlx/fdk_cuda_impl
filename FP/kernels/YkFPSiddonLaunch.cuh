#pragma once
#include <cuda_runtime.h>
#include "../../global/YkGlobals.h"

namespace YK {
    namespace Fp {
        namespace detail {

            // 无分组，无 slice 循环
            // d_views / d_sino 已偏移到本 chunk 起始
            void siddon_launchGroup(
                const float* d_vol,    // 线性内存 [Nz][Ny][Nx]
                float* d_sino,   // [K][Nv][Nu]
                const SConeProjGeomVec* d_views,  // K 个
                const SVolGeom& g,
                int Nu, int Nv, int K,
                bool accumulate,
                cudaStream_t stream);

        } // namespace detail
    } // namespace Fp
} // namespace YK