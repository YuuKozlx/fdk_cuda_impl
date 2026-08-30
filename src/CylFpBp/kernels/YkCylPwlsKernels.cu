#include "CylFpBp/kernels/YkCylPwlsLaunch.cuh"

#include <algorithm>

#include "global/YkMacro.hpp"

namespace YK::CylFpBp::detail {
namespace {

__global__ void residual_kernel(const float* measured, const float* forward,
    const float* weights, float weight_scale, float* residual, size_t count)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float value = measured[i] - forward[i];
        residual[i] = weight_scale * (weights ? weights[i] : 1.f) * value;
    }
}

__global__ void apply_weights_kernel(const float* input, const float* weights,
    float weight_scale, float* output, size_t count)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        output[i] = weight_scale * (weights ? weights[i] : 1.f) * input[i];
}

} // namespace

void launch_cyl_pwls_residual(const float* measured, const float* forward,
    const float* weights, float weight_scale, float* residual, size_t count,
    cudaStream_t stream)
{
    if (count == 0) return;
    const int blocks = static_cast<int>(std::min<size_t>((count + 255) / 256,
        65535));
    residual_kernel<<<blocks, 256, 0, stream>>>(measured, forward, weights,
        weight_scale, residual, count);
    YK_CUDA_KERNEL_CHECK();
}

void launch_cyl_pwls_apply_weights(const float* input, const float* weights,
    float weight_scale, float* output, size_t count, cudaStream_t stream)
{
    if (count == 0) return;
    const int blocks = static_cast<int>(std::min<size_t>((count + 255) / 256,
        65535));
    apply_weights_kernel<<<blocks, 256, 0, stream>>>(input, weights,
        weight_scale, output, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
