#include "YkTvRegularizationLaunch.cuh"

#include "global/YkMacro.hpp"

#include <cmath>

namespace YK::Iter {
namespace {

__device__ __forceinline__ float read_volume(
    const float* volume, int x, int y, int z, int nx, int ny)
{
    return volume[(static_cast<size_t>(z) * ny + y) * nx + x];
}

__device__ __forceinline__ void normalized_forward_gradient(
    const float* volume, int x, int y, int z,
    int nx, int ny, int nz,
    float inv_dx, float inv_dy, float inv_dz,
    float epsilon, bool use_z,
    float& px, float& py, float& pz)
{
    const float center = read_volume(volume, x, y, z, nx, ny);
    const float gx = x + 1 < nx
        ? (read_volume(volume, x + 1, y, z, nx, ny) - center) * inv_dx
        : 0.f;
    const float gy = y + 1 < ny
        ? (read_volume(volume, x, y + 1, z, nx, ny) - center) * inv_dy
        : 0.f;
    const float gz = use_z && z + 1 < nz
        ? (read_volume(volume, x, y, z + 1, nx, ny) - center) * inv_dz
        : 0.f;
    const float denominator = sqrtf(
        gx * gx + gy * gy + gz * gz + epsilon * epsilon);
    px = gx / denominator;
    py = gy / denominator;
    pz = gz / denominator;
}

__global__ void tv_gradient_kernel(const float* volume, float* gradient,
    int nx, int ny, int nz,
    float inv_dx, float inv_dy, float inv_dz,
    float epsilon, bool use_z)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z >= nz) return;

    float px = 0.f, py = 0.f, pz = 0.f;
    normalized_forward_gradient(volume, x, y, z, nx, ny, nz,
        inv_dx, inv_dy, inv_dz, epsilon, use_z, px, py, pz);

    float px_prev = 0.f, py_prev = 0.f, pz_prev = 0.f;
    float discard_a = 0.f, discard_b = 0.f;
    if (x > 0)
        normalized_forward_gradient(volume, x - 1, y, z, nx, ny, nz,
            inv_dx, inv_dy, inv_dz, epsilon, use_z,
            px_prev, discard_a, discard_b);
    if (y > 0)
        normalized_forward_gradient(volume, x, y - 1, z, nx, ny, nz,
            inv_dx, inv_dy, inv_dz, epsilon, use_z,
            discard_a, py_prev, discard_b);
    if (use_z && z > 0)
        normalized_forward_gradient(volume, x, y, z - 1, nx, ny, nz,
            inv_dx, inv_dy, inv_dz, epsilon, use_z,
            discard_a, discard_b, pz_prev);

    const float divergence = (px - px_prev) * inv_dx +
        (py - py_prev) * inv_dy +
        (use_z ? (pz - pz_prev) * inv_dz : 0.f);
    gradient[(static_cast<size_t>(z) * ny + y) * nx + x] = -divergence;
}

} // namespace

void tv_gradient_launch(const float* volume, float* gradient,
    int nx, int ny, int nz,
    float spacing_x, float spacing_y, float spacing_z,
    float epsilon, int dimensionality, cudaStream_t stream)
{
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);
    tv_gradient_kernel<<<grid, block, 0, stream>>>(volume, gradient,
        nx, ny, nz, 1.f / spacing_x, 1.f / spacing_y, 1.f / spacing_z,
        epsilon, dimensionality == 3);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::Iter
