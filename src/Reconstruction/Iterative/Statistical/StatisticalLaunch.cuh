#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
namespace YK::Iter::Statistical {
// Reused reduction scratch: 1024 double block sums plus a host mirror.
constexpr int reduction_blocks = 1024;
void weighted_residual_launch(float*, const float*, const float*, size_t, double*, cudaStream_t);
void squared_change_launch(const float*, const float*, size_t, double*, cudaStream_t);
void max_absolute_launch(const float*, size_t, double*, cudaStream_t);
void gradient_update_launch(float*, const float*, const float*, size_t, float, float, float, cudaStream_t);
}
