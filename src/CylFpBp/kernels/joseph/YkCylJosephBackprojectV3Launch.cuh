#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"
namespace YK::CylFpBp::detail {
void launch_joseph_backproject_v3(cudaTextureObject_t, const SVoxelDrivenView*, float*,
    int, const SVolGeom&, cudaStream_t, bool);
void launch_joseph_layered_forward(cudaTextureObject_t, float*,
    const SVoxelDrivenView*, const SSiddonChannelRay*, int, int, int,
    const SVolGeom&, cudaStream_t, bool);
void launch_joseph_exact_gather(cudaTextureObject_t, float*,
    const SVoxelDrivenView*, const SSiddonChannelRay*, int, int, int,
    const SVolGeom&, cudaStream_t, bool);
} // namespace YK::CylFpBp::detail
