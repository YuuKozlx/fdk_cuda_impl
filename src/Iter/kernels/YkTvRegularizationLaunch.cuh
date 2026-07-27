#pragma once

#include <cuda_runtime.h>

namespace YK::Iter {

void tv_gradient_launch(const float* volume, float* gradient,
    int nx, int ny, int nz,
    float spacing_x, float spacing_y, float spacing_z,
    float epsilon, int dimensionality, cudaStream_t stream);

} // namespace YK::Iter
