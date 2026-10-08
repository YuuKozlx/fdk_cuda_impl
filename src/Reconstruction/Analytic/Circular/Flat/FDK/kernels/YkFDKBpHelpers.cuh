#pragma once
#include <cuda_runtime.h>

#include "common/YkVecGeo.hpp"             // SConeProjGeomVec, SFDKGeoParamPerView
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKVecGeoDerived.hpp"   // FdkAffineCoeff（类型定义）
#include "util/YkVecOperation.hpp"       // f3_sub, f3_dot

namespace YK {
    namespace Fdk {



        void bp_uploadCoeffsChunk(
            const FdkAffineCoeff* d_src, int K, cudaStream_t stream);

        void bp_readbackCoeffs(           // ← 新增
            FdkAffineCoeff* h_dst, int K);
    }
} // namespace YK::Fdk::detail
