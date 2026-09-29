#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace YK::DltFpBp {

void dltFillOnesLaunch(float* data, size_t count, cudaStream_t stream);
void dltResidualLaunch(const float* measured, const float* forward,
    float* residual, size_t count, cudaStream_t stream);
void dltDivideLaunch(float* values, const float* denominator, float epsilon,
    size_t count, cudaStream_t stream);
void dltUpdateLaunch(float* volume, const float* backprojection,
    const float* column_weight, float relaxation, float epsilon,
    size_t count, cudaStream_t stream);
void dltClampMinLaunch(float* values, float minimum, size_t count,
    cudaStream_t stream);
void dltMeanZLaunch(const float* volume, float* image,
    int nx, int ny, int nz, cudaStream_t stream);
void dltThresholdInfLaunch(float* values, float threshold, size_t count,
    cudaStream_t stream);
void dltUpdate2dLaunch(float* volume, const float* backprojection,
    const float* column_weight_2d, float relaxation, float epsilon,
    int nx, int ny, int nz, cudaStream_t stream);

} // namespace YK::DltFpBp
