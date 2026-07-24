#include "Heli/kernels/YkHelicalIcdLaunch.cuh"

#include <cuda_runtime.h>

#include "global/YkGlobals.h"

namespace YK {
namespace Helical {
namespace {

__global__ void helical_icd_update_kernel(float* volume,
    const float* residual_bp, const float* data_curvature,
    int nx, int ny, int nz, float relaxation, float regularization,
    float epsilon, float lower_bound, float upper_bound, size_t count)
{
    const size_t xy = static_cast<size_t>(nx) * ny;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(i % nx);
        const int y = static_cast<int>((i / nx) % ny);
        const int z = static_cast<int>(i / xy);

        float neighbor_sum = 0.f;
        int degree = 0;
        if (x > 0) { neighbor_sum += volume[i - 1]; ++degree; }
        if (x + 1 < nx) { neighbor_sum += volume[i + 1]; ++degree; }
        if (y > 0) { neighbor_sum += volume[i - nx]; ++degree; }
        if (y + 1 < ny) { neighbor_sum += volume[i + nx]; ++degree; }
        if (z > 0) { neighbor_sum += volume[i - xy]; ++degree; }
        if (z + 1 < nz) { neighbor_sum += volume[i + xy]; ++degree; }

        const float current = volume[i];
        const float prior_gradient = regularization *
            (static_cast<float>(degree) * current - neighbor_sum);
        const float denominator = data_curvature[i] +
            regularization * static_cast<float>(degree) + epsilon;
        const float next = current + relaxation *
            (residual_bp[i] - prior_gradient) / denominator;
        volume[i] = fminf(fmaxf(next, lower_bound), upper_bound);
    }
}

} // namespace

void helical_icd_update_launch(float* volume, const float* residual_bp,
    const float* data_curvature, int nx, int ny, int nz,
    float relaxation, float regularization, float epsilon,
    float lower_bound, float upper_bound, ::CUstream_st* stream)
{
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    if (!volume || !residual_bp || !data_curvature || count == 0) return;
    const auto launch = SKernelLaunchPolicy{}.make1D(count);
    helical_icd_update_kernel<<<launch.grid, launch.block, 0, stream>>>(
        volume, residual_bp, data_curvature, nx, ny, nz, relaxation,
        regularization, epsilon, lower_bound, upper_bound, count);
}

} // namespace Helical
} // namespace YK
