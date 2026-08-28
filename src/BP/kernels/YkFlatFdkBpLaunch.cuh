#pragma once
#include <cuda_runtime.h>
#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "global/YkFdkKernelTypes.hpp"

namespace YK::Bp {
void fdk_bp_launch(cudaTextureObject_t projection_texture,
    const SConeProjGeomVec* views_world, const FdkAffineCoeff* coefficients,
    float* volume, const SVolGeom& volume_geometry, int views,
    bool accumulate, cudaStream_t stream);
void fdk_matched_bp_launch(cudaTextureObject_t projection_texture,
    const SConeProjGeomVec* views_world, const FdkAffineCoeff* coefficients,
    float* volume, const SVolGeom& volume_geometry, int views,
    bool accumulate, cudaStream_t stream);
} // namespace YK::Bp
