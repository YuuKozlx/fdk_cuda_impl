#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"
namespace YK::CylFpBp::detail {
void launch_cyl_siddon_forward(cudaTextureObject_t, float*, const SKernelView*,
    const SSiddonChannelRay*, int, int, int, const SVolGeom&, bool,
    cudaStream_t);
void launch_cyl_siddon_backproject(cudaTextureObject_t, float*,
    const SKernelView*, const SSiddonChannelRay*, int, int, int,
    const SVolGeom&, bool, cudaStream_t);
void launch_cyl_siddon_backproject_v2(cudaTextureObject_t, float*,
    const SVoxelDrivenView*, int,
    const SVolGeom&, bool, cudaStream_t);
void launch_cyl_siddon_backproject_v3(cudaTextureObject_t, float*,
    const SVoxelDrivenView*, const SCylVoxelChannelRay*, int, int,
    const SVolGeom&, bool, cudaStream_t);
void launch_cyl_siddon_backproject_v3_standard(cudaTextureObject_t, float*,
    const SVoxelDrivenView*, const SCylVoxelChannelRay*, int, int,
    const SVolGeom&, bool, cudaStream_t);
} // namespace YK::CylFpBp::detail
