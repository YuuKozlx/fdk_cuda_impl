#include "Iter/kernels/YkParallelPwlsLaunch.cuh"

#include <cuda_runtime.h>

#include "global/YkGlobals.h"

namespace YK::Iter {
namespace {

__global__ void parallel_pwls_update_kernel(float* volume,
    const float* residual_bp, const float* data_curvature,
    int nx, int ny, int nz, float relaxation, float data_curvature_scale,
    float regularization,
    int regularizer, float huber_delta, float epsilon,
    float lower_bound, float upper_bound, size_t count)
{
    const size_t xy = static_cast<size_t>(nx) * ny;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(i % nx);
        const int y = static_cast<int>((i / nx) % ny);
        const int z = static_cast<int>(i / xy);
        const float current = volume[i];
        float prior_gradient = 0.f;
        float prior_curvature = 0.f;
        if (regularizer != 0 && regularization > 0.f) {
            const auto accumulate_neighbor = [&](float neighbor) {
                const float difference = current - neighbor;
                if (regularizer == 2) {
                    const float magnitude = fabsf(difference);
                    const float weight = magnitude > huber_delta
                        ? huber_delta / magnitude : 1.f;
                    prior_gradient += regularization * weight * difference;
                    prior_curvature += regularization * weight;
                }
                else {
                    prior_gradient += regularization * difference;
                    prior_curvature += regularization;
                }
            };
            if (x > 0) accumulate_neighbor(volume[i - 1]);
            if (x + 1 < nx) accumulate_neighbor(volume[i + 1]);
            if (y > 0) accumulate_neighbor(volume[i - nx]);
            if (y + 1 < ny) accumulate_neighbor(volume[i + nx]);
            if (z > 0) accumulate_neighbor(volume[i - xy]);
            if (z + 1 < nz) accumulate_neighbor(volume[i + xy]);
        }
        const float denominator = data_curvature_scale * data_curvature[i] +
            prior_curvature + epsilon;
        const float next = current + relaxation *
            (residual_bp[i] - prior_gradient) / denominator;
        volume[i] = fminf(fmaxf(next, lower_bound), upper_bound);
    }
}

} // namespace

void parallel_pwls_update_launch(float* volume, const float* residual_bp,
    const float* data_curvature, int nx, int ny, int nz,
    float relaxation, float data_curvature_scale,
    int regularizer, float regularization,
    float huber_delta, float epsilon,
    float lower_bound, float upper_bound, ::CUstream_st* stream)
{
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    if (!volume || !residual_bp || !data_curvature || count == 0) return;
    const auto launch = SKernelLaunchPolicy{}.make1D(count);
    parallel_pwls_update_kernel<<<launch.grid, launch.block, 0, stream>>>(
        volume, residual_bp, data_curvature, nx, ny, nz, relaxation,
        data_curvature_scale, regularization, regularizer, huber_delta, epsilon,
        lower_bound, upper_bound, count);
}

} // namespace YK::Iter
