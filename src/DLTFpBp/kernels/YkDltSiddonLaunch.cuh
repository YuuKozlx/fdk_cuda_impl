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

// Voxel-driven, non-adjoint backprojection. Each volume voxel is projected
// into every view, samples the detector image, and is weighted by the chord
// length through that voxel. This is intentionally separate from the matched
// Siddon backprojector because it is an approximation, not A^T.
void dltVoxelBackLaunch(const float* projection, float* volume,
    const SDltVoxelBackGeometry* geometry, const SVolGeom& volume_geometry,
    int channels, int rows, int views, bool bilinear, bool accumulate,
    cudaStream_t stream);

} // namespace YK::DltFpBp
