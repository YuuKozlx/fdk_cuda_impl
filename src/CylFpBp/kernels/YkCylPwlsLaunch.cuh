#pragma once

#include <cuda_runtime.h>

namespace YK::CylFpBp::detail {

// residual = weights * (measured - forward). weights 可以为空，此时退化为
// 未加权 PLS；数据布局为连续的 [view][row][channel]。
void launch_cyl_pwls_residual(const float* measured, const float* forward,
    const float* weights, float weight_scale, float* residual, size_t count,
    cudaStream_t stream);

void launch_cyl_pwls_apply_weights(const float* input, const float* weights,
    float weight_scale, float* output, size_t count, cudaStream_t stream);

} // namespace YK::CylFpBp::detail
