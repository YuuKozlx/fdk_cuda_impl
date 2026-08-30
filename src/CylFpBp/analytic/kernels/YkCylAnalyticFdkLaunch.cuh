#pragma once

#include <cuda_runtime.h>

#include "CylFpBp/kernels/YkCylKernelTypes.cuh"

namespace YK::CylFpBp::detail {
void launch_cyl_analytic_fdk_bp(cudaTextureObject_t, const SCylFdkView*, float*,
    int, const SVolGeom&, cudaStream_t, bool);
}
