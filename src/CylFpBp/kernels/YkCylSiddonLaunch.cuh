#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"
namespace YK::CylFpBp::detail {
void launch_cyl_siddon_forward(cudaTextureObject_t, float*, const SKernelView*,
    int, int, int, const SVolGeom&, bool, cudaStream_t);
} // namespace YK::CylFpBp::detail
