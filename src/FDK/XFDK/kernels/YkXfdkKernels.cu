#include "FDK/XFDK/kernels/YkXfdkLaunch.cuh"

#include <cmath>

#include "global/YkMacro.hpp"

namespace YK::Fdk::detail {
namespace {

__device__ float logicalViewCoordinate(float alpha, const SXfdkGeometry& g)
{
    return (alpha - g.alpha0_rad) / g.signed_dtheta_rad;
}

__device__ float sampleWindowView(cudaTextureObject_t texture,
    float u, float v, float logical_view, const SXfdkChunk& chunk)
{
    const float local_view = logical_view - static_cast<float>(chunk.source_begin);
    const int i0 = static_cast<int>(floorf(local_view));
    const int i1 = i0 + 1;
    if (i0 < 0 || i1 >= chunk.source_count) return 0.f;
    const float f = local_view - i0;
    const float p0 = tex3D<float>(texture, u + 0.5f, v + 0.5f,
        static_cast<float>(i0) + 0.5f);
    const float p1 = tex3D<float>(texture, u + 0.5f, v + 0.5f,
        static_cast<float>(i1) + 0.5f);
    return fmaf(f, p1 - p0, p0);
}

// 将目标平行束坐标 (vartheta, xi, gamma) 映射回原始平板的
// (alpha, u, v)。xi=-R sin(beta)，u=SDD tan(beta)，
// v=SDD tan(gamma)/cos(beta)。重排同时乘 cos(epsilon)=cos(beta)cos(gamma)。
__global__ void xfdkRebinKernel(cudaTextureObject_t input, float* output,
    SXfdkGeometry g, SXfdkChunk chunk, size_t elements)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int ixi = static_cast<int>(index % g.output_xi);
        const size_t row = index / g.output_xi;
        const int igamma = static_cast<int>(row % g.output_gamma);
        const int local_theta = static_cast<int>(row / g.output_gamma);
        const int global_theta = chunk.theta_begin + local_theta;

        const float xi = fmaf(static_cast<float>(ixi), g.dxi_mm, g.xi_min_mm);
        const float gamma = fmaf(static_cast<float>(igamma), g.dgamma_rad,
            g.gamma_min_rad);
        const float beta = asinf(fminf(fmaxf(-xi / g.sid_mm, -1.f), 1.f));
        const float cos_beta = cosf(beta);
        const float cos_gamma = cosf(gamma);
        if (cos_beta <= 1e-6f || cos_gamma <= 1e-6f) {
            output[index] = 0.f;
            continue;
        }

        const float theta = fmaf(static_cast<float>(global_theta),
            g.signed_dtheta_rad, g.alpha0_rad);
        const float alpha = theta - beta;
        // 本项目 detU 与论文的 a 轴相反：正 u 对应 beta<0。
        const float u = -g.sdd_mm * tanf(beta) / g.input_du_mm +
            0.5f * static_cast<float>(g.input_u - 1);
        const float v = g.sdd_mm * tanf(gamma) /
            (cos_beta * g.input_dv_mm) +
            0.5f * static_cast<float>(g.input_v - 1);
        const float view = logicalViewCoordinate(alpha, g);
        const float correction = cos_beta * cos_gamma;
        output[index] = correction * sampleWindowView(input, u, v, view, chunk);
    }
}

__global__ void xfdkBackprojectionKernel(cudaTextureObject_t filtered,
    float* volume, SXfdkGeometry g, SVolGeom vol, SXfdkChunk chunk,
    bool accumulate, size_t voxels)
{
    const float3 origin = vol.origin();
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < voxels;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int ix = static_cast<int>(index % vol.Nx);
        const size_t yz = index / vol.Nx;
        const int iy = static_cast<int>(yz % vol.Ny);
        const int iz = static_cast<int>(yz / vol.Ny);
        const float x = fmaf(static_cast<float>(ix), vol.vox_x, origin.x);
        const float y = fmaf(static_cast<float>(iy), vol.vox_y, origin.y);
        const float z = fmaf(static_cast<float>(iz), vol.vox_z, origin.z);
        const float r = sqrtf(fmaf(x, x, y * y));
        const float phi = atan2f(-x, y);
        const float delta = xfdkCoverageHalfRange(g, r, z);
        float sum = 0.f;

        if (delta >= 0.f) {
            for (int local_theta = 0; local_theta < chunk.theta_count;
                 ++local_theta) {
                const int global_theta = chunk.theta_begin + local_theta;
                const float theta = fmaf(static_cast<float>(global_theta),
                    g.signed_dtheta_rad, g.alpha0_rad);
                const float sine = sinf(theta), cosine = cosf(theta);
                const float xi = fmaf(x, cosine, y * sine);
                const float tangential = fmaf(-x, sine, y * cosine);
                const float horizontal_depth = sqrtf(fmaxf(
                    g.sid_mm * g.sid_mm - xi * xi, 0.f));
                const float ray_depth = horizontal_depth + tangential;
                if (ray_depth <= 1e-6f) continue;
                const float gamma = atanf(z / ray_depth);
                if (fabsf(gamma) > g.gamma_max_rad) continue;

                const float wc = xfdkCompositeWeight(g, r, z, theta, phi);
                if (wc <= 0.f) continue;

                const float fxi = (xi - g.xi_min_mm) / g.dxi_mm;
                const float fgamma = (gamma - g.gamma_min_rad) / g.dgamma_rad;
                sum += wc * tex3D<float>(filtered, fxi + 0.5f,
                    fgamma + 0.5f, static_cast<float>(local_theta) + 0.5f);
            }
        }
        const float value = 0.5f * g.dtheta_rad * sum;
        volume[index] = accumulate ? volume[index] + value : value;
    }
}

} // namespace

void launchXfdkRebin(cudaTextureObject_t input, float* output,
    const SXfdkGeometry& geometry, const SXfdkChunk& chunk,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t elements = static_cast<size_t>(geometry.output_xi) *
        geometry.output_gamma * chunk.theta_count;
    const auto launch = policy.make1D(elements);
    xfdkRebinKernel<<<launch.grid, launch.block, 0, stream>>>(
        input, output, geometry, chunk, elements);
    YK_CUDA_KERNEL_CHECK();
}

void launchXfdkBackprojection(cudaTextureObject_t filtered, float* volume,
    const SXfdkGeometry& geometry, const SVolGeom& volume_geometry,
    const SXfdkChunk& chunk, bool accumulate,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t voxels = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    const auto launch = policy.make1D(voxels);
    xfdkBackprojectionKernel<<<launch.grid, launch.block, 0, stream>>>(
        filtered, volume, geometry, volume_geometry, chunk, accumulate, voxels);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::Fdk::detail
