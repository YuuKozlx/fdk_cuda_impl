#include "YkTvRegularizationLaunch.cuh"

#include "global/YkMacro.hpp"
#include "YkIterLaunch.cuh"

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

__device__ __forceinline__ float adaptive_weight(float difference, float delta)
{
    if (delta == 0.f) return 1.f;
    const float ratio = difference / delta;
    return __expf(-(ratio * ratio));
}

__global__ void tigre_tv_gradient_kernel(const float* volume, float* gradient,
    int nx, int ny, int nz, float epsilon, bool adaptive, float delta)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z >= nz) return;

    auto components = [=] __device__ (int px, int py, int pz,
        float& dz, float& dy, float& dx) {
        if (px < 0 || py < 0 || pz < 0 || px >= nx || py >= ny || pz >= nz) {
            dz = dy = dx = 0.f;
            return;
        }
        const float center = read_volume(volume, px, py, pz, nx, ny);
        dz = pz > 0 ? center - read_volume(volume, px, py, pz - 1, nx, ny) : 0.f;
        dy = py > 0 ? center - read_volume(volume, px, py - 1, pz, nx, ny) : 0.f;
        dx = px > 0 ? center - read_volume(volume, px - 1, py, pz, nx, ny) : 0.f;
    };

    float dz = 0.f, dy = 0.f, dx = 0.f;
    float dzi = 0.f, dyi = 0.f, dxi = 0.f;
    float dzj = 0.f, dyj = 0.f, dxj = 0.f;
    float dzk = 0.f, dyk = 0.f, dxk = 0.f;
    components(x, y, z, dz, dy, dx);
    components(x + 1, y, z, dzi, dyi, dxi);
    components(x, y + 1, z, dzj, dyj, dxj);
    components(x, y, z + 1, dzk, dyk, dxk);

    const auto weighted_norm = [=] __device__ (float gz, float gy, float gx) {
        const float wz = adaptive ? adaptive_weight(gz, delta) : 1.f;
        const float wy = adaptive ? adaptive_weight(gy, delta) : 1.f;
        const float wx = adaptive ? adaptive_weight(gx, delta) : 1.f;
        return sqrtf(wz * gz * gz + wy * gy * gy + wx * gx * gx) + epsilon;
    };
    const float wz = adaptive ? adaptive_weight(dz, delta) : 1.f;
    const float wy = adaptive ? adaptive_weight(dy, delta) : 1.f;
    const float wx = adaptive ? adaptive_weight(dx, delta) : 1.f;
    const float wzi = adaptive ? adaptive_weight(dzi, delta) : 1.f;
    const float wyj = adaptive ? adaptive_weight(dyj, delta) : 1.f;
    const float wxk = adaptive ? adaptive_weight(dxk, delta) : 1.f;

    const float value = (wz * dz + wy * dy + wx * dx) /
            weighted_norm(dz, dy, dx) -
        wzi * dzi / weighted_norm(dzi, dyi, dxi) -
        wyj * dyj / weighted_norm(dzj, dyj, dxj) -
        wxk * dxk / weighted_norm(dzk, dyk, dxk);
    gradient[(static_cast<size_t>(z) * ny + y) * nx + x] = value;
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

void tigre_tv_descent_launch(float* volume, float* gradient,
    int nx, int ny, int nz, float step, int iterations,
    float epsilon, bool adaptive_weighted, float delta,
    cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        tigre_tv_gradient_kernel<<<grid, block, 0, stream>>>(volume, gradient,
            nx, ny, nz, epsilon, adaptive_weighted, delta);
        YK_CUDA_KERNEL_CHECK();
        float norm2 = 0.f;
        dot_launch(gradient, gradient, count, &norm2, stream);
        const float norm = sqrtf(norm2);
        if (norm <= epsilon) break;
        axpy_launch(volume, gradient, -step / norm, count, stream);
    }
}

} // namespace YK::Iter
