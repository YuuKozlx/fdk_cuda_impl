#pragma once

#include <cuda_runtime.h>

namespace YK {

// Vector cone-beam geometry for one view. Positions use millimetres; detU
// and detV are one-pixel steps. angle.x is the single angle source in radians.
struct SConeProjGeomVec {
    float4 src;
    float4 srcCR;
    float4 detS;
    float4 detU;
    float4 detV;
    float4 angle;
};

} // namespace YK
