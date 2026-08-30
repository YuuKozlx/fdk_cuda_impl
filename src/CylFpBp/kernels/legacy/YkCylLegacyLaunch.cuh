#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"
namespace YK::CylFpBp::detail {
void launch_legacy_forward(const float*, float*, const SCylConeProjGeomVec*, int,
    int, int, const SVolGeom&, const Config&, cudaStream_t, bool);
void launch_legacy_backproject(const float*, float*, const SCylConeProjGeomVec*,
    int, int, int, const SVolGeom&, const Config&, cudaStream_t, bool);
void launch_precomputed_forward(const float*, float*, const SKernelView*, int,
    int, int, const SVolGeom&, const Config&, cudaStream_t, bool);
void launch_precomputed_backproject(const float*, float*, const SKernelView*, int,
    int, int, const SVolGeom&, const Config&, cudaStream_t, bool);
} // namespace YK::CylFpBp::detail
