#pragma once

#include <cuda_runtime.h>

#include "CylFpBp/YkCylFpBpTypes.hpp"

namespace YK::CylFpBp::detail {

void launch_forward(const float* volume, float* projection,
    const SCylConeProjGeomVec* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate);

void launch_backproject(const float* projection, float* volume,
    const SCylConeProjGeomVec* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate);

} // namespace YK::CylFpBp::detail
