#pragma once

#include <cuda_runtime_api.h>

struct CUstream_st;

namespace YK::Iter {

void parallel_pwls_update_launch(float* volume, const float* residual_bp,
    const float* data_curvature, int nx, int ny, int nz,
    float relaxation, float data_curvature_scale,
    int regularizer, float regularization,
    float huber_delta, float epsilon,
    float lower_bound, float upper_bound, ::CUstream_st* stream);

} // namespace YK::Iter
