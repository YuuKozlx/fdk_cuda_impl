#pragma once
#include <cuda_runtime.h>
#include "global/YkGlobals.h"

namespace YK::Fp {
void fp_siddon_launch(const float* volume, float* projection,
    const SConeProjGeomVec* views, const SVolGeom& volume_geometry,
    int channels, int rows, int views_count, bool accumulate,
    cudaStream_t stream);
void fp_siddon_launch(cudaTextureObject_t volume_texture, float* projection,
    const SConeProjGeomVec* views, const SVolGeom& volume_geometry,
    int channels, int rows, int views_count, bool accumulate,
    cudaStream_t stream);
} // namespace YK::Fp
