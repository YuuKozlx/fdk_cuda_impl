#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"
namespace YK::CylFpBp::detail {
void launch_cyl_fdk_bp(cudaTextureObject_t, const SCylFdkView*, float*, int,
    const SVolGeom&, cudaStream_t, bool);
void launch_cyl_fdk_matched_bp(cudaTextureObject_t, const SCylFdkView*, float*,
    int, const SVolGeom&, cudaStream_t, bool);
} // namespace YK::CylFpBp::detail
