#pragma once

#include <cuda_runtime.h>

namespace YK::Iter {

void tv_gradient_launch(const float* volume, float* gradient,
    int nx, int ny, int nz,
    float spacing_x, float spacing_y, float spacing_z,
    float epsilon, int dimensionality, cudaStream_t stream);

// TIGRE POCS 系列使用的 TV/AwTV 下降步。每次内迭代都会把梯度归一化为
// L2=1，再以绝对图像距离 step 更新，因此 step 对应 TIGRE 的 dtvg。
void tigre_tv_descent_launch(float* volume, float* gradient,
    int nx, int ny, int nz, float step, int iterations,
    float epsilon, bool adaptive_weighted, float delta,
    cudaStream_t stream);

} // namespace YK::Iter
