#pragma once

#include <cuda_runtime.h>

#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"

namespace YK::DltFpBp {

void dltSiddonForwardLaunch(const float* volume, float* projection,
    const SDltRayGeometry* geometry, const SVolGeom& volume_geometry,
    int channels, int rows, int views, bool accumulate, cudaStream_t stream);

void dltSiddonBackLaunch(const float* projection, float* volume,
    const SDltRayGeometry* geometry, const SVolGeom& volume_geometry,
    int channels, int rows, int views, cudaStream_t stream);

} // namespace YK::DltFpBp
