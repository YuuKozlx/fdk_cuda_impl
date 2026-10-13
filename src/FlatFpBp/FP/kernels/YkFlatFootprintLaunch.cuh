#pragma once

#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"
#include "common/cuda/operators/YkOperatorKernelTypes.cuh"

namespace YK::Footprint {

void forward_launch(const float* volume, float* projection,
    const CudaOp::SFootprintView* views, const SVolGeom& volume_geometry,
    int view_count, int channels, int rows, bool accumulate,
    cudaStream_t stream);

void back_launch(const float* projection, float* volume,
    const CudaOp::SFootprintView* views, const SVolGeom& volume_geometry,
    int view_count, int channels, int rows, bool accumulate,
    cudaStream_t stream);

} // namespace YK::Footprint
