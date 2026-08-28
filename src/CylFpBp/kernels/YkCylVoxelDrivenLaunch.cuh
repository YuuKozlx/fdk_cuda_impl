#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"
namespace YK::CylFpBp::detail {
void launch_voxel_driven_v3(cudaTextureObject_t, const SVoxelDrivenView*, float*,
    int, const SVolGeom&, cudaStream_t, bool);
} // namespace YK::CylFpBp::detail
