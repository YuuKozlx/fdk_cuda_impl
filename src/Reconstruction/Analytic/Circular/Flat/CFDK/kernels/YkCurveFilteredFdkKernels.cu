#include "Reconstruction/Analytic/Circular/Flat/CFDK/kernels/YkCurveFilteredFdkLaunch.cuh"

#include <cmath>

#include "global/YkMacro.hpp"

namespace YK::Fdk::detail {
namespace {

__device__ float periodicViewCoordinate(float beta,
    const SCurveFilteredFdkGeometry& g)
{
    // signed_dtheta 同时编码采集方向。fmod 后的坐标严格落在 [0,N)，
    // 后续显式取相邻两帧，因此角度轴可以周期插值而 U/V 仍保持 border-zero。
    float coordinate = (beta - g.beta0_rad) / g.signed_dtheta_rad;
    coordinate = fmodf(coordinate, static_cast<float>(g.views));
    return coordinate < 0.f ? coordinate + g.views : coordinate;
}

__device__ float samplePeriodicViews(cudaTextureObject_t texture,
    float u, float v, float view_coordinate, int views)
{
    const int view0 = static_cast<int>(floorf(view_coordinate));
    const int view1 = view0 + 1 == views ? 0 : view0 + 1;
    const float fraction = view_coordinate - view0;
    const float p0 = tex3D<float>(texture, u + 0.5f, v + 0.5f,
        static_cast<float>(view0) + 0.5f);
    const float p1 = tex3D<float>(texture, u + 0.5f, v + 0.5f,
        static_cast<float>(view1) + 0.5f);
    return fmaf(fraction, p1 - p0, p0);
}

__global__ void curveFilteredFdkRebinPreweightKernel(
    cudaTextureObject_t input, float* output,
    SCurveFilteredFdkGeometry g, size_t elements)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int it = static_cast<int>(index % g.output_t);
        const size_t row = index / g.output_t;
        const int ic = static_cast<int>(row % g.output_c);
        const int iview = static_cast<int>(row / g.output_c);

        const float t = fmaf(static_cast<float>(it), g.dt_mm, g.t_min_mm);
        const float c = fmaf(static_cast<float>(ic), g.dc_mm, g.c_min_mm);
        const float t2 = t * t;
        const float r2 = g.sid_mm * g.sid_mm;
        const float q2 = r2 - t2;
        if (q2 <= 0.f) {
            output[index] = 0.f;
            continue;
        }
        const float q = sqrtf(q2);
        const float source_angle = fmaf(static_cast<float>(iview),
            g.signed_dtheta_rad, g.beta0_rad);
        // 论文的探测器 a 轴与本项目 detU 方向相反，因此式 (17)/(29)
        // 转到项目坐标后符号翻转。令 delta=asin(t/R)、beta=theta+delta、
        // a=tR/q，并记 e_theta=(-sin(theta),cos(theta))。可直接验证：
        //   dot(R*n_beta, e_theta) = t
        //   dot(a*e_beta, e_theta) = t
        // 所以源点和虚拟探测器点位于同一条式 (33) 射线上。
        const float beta = source_angle + asinf(t / g.sid_mm);
        const float a = t * g.sid_mm / q;
        const float abs_c = fabsf(c);

        float b = 0.f;
        float weight = 0.f;
        if (abs_c <= g.c0_mm) {
            const float sc = 2.f * q - g.sid_mm;
            if (sc <= 0.f) {
                output[index] = 0.f;
                continue;
            }
            b = c * r2 / (q * sc);
            weight = sc * rsqrtf(fmaxf(
                5.f * r2 - 4.f * t2 - 4.f * g.sid_mm * q + c * c,
                1e-20f));
        } else if (abs_c <= g.s0_mm) {
            const float blend = (abs_c - g.c0_mm) /
                fmaxf(g.s0_mm - g.c0_mm, 1e-20f);
            // 论文式 (25)-(26)：SC'=2q-R+lambda(R-q)。
            const float sc = 2.f * q - g.sid_mm +
                blend * (g.sid_mm - q);
            b = c * r2 / (q * sc);
            // 式 (31) 第二行：过渡区统一采用 R^2/sqrt(R^4+b^2(R^2-t^2))。
            weight = r2 * rsqrtf(fmaxf(r2 * r2 + b * b * q2, 1e-20f));
        } else if (abs_c <= g.bm_mm) {
            const float blend = (abs_c - g.s0_mm) /
                fmaxf(g.bm_mm - g.s0_mm, 1e-20f);
            b = c * r2 / (q2 + blend * t2);
            // 式 (31) 第二行同样适用于 s0<|c|<=bm。
            weight = r2 * rsqrtf(fmaxf(r2 * r2 + b * b * q2, 1e-20f));
        } else {
            output[index] = 0.f;
            continue;
        }

        // a/b 位于论文的中心虚拟平面，采样实际平板前按 SDD/SID 放大。
        const float detector_scale = g.sdd_mm / g.sid_mm;
        const float u = a * detector_scale / g.input_du_mm +
            0.5f * static_cast<float>(g.input_u - 1);
        const float v = b * detector_scale / g.input_dv_mm +
            0.5f * static_cast<float>(g.input_v - 1);
        const float view = periodicViewCoordinate(beta, g);
        output[index] = weight * samplePeriodicViews(input, u, v, view, g.views);
    }
}

__global__ void curveFilteredFdkBackprojectionKernel(
    cudaTextureObject_t filtered, float* volume,
    SCurveFilteredFdkGeometry g, SVolGeom vol, bool accumulate,
    size_t voxels)
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
        float sum = 0.f;

        for (int iview = 0; iview < g.views; ++iview) {
            const float source_angle = fmaf(static_cast<float>(iview),
                g.signed_dtheta_rad, g.beta0_rad);
            float sine = 0.f, cosine = 0.f;
            sincosf(source_angle, &sine, &cosine);
            // t 取项目 detU=e_theta 方向的投影。项目源位于 +R*n_theta，
            // 而论文式 (35) 的 q+v 表示“源到旋转中心”的径向深度，
            // 因此在项目世界坐标中 v 必须取径向投影的负值；否则 theta=0
            // 时分母会错误地变成 R+x，而物理源到体素距离应为 R-x。
            const float t = y * cosine - x * sine;
            const float v = -(x * cosine + y * sine);
            float c = 0.f;
            if (!mapCurveFilteredFdkBackprojectionC(g, t, v, z, c)) continue;

            const float ft = (t - g.t_min_mm) / g.dt_mm;
            const float fc = (c - g.c_min_mm) / g.dc_mm;
            sum += tex3D<float>(filtered, ft + 0.5f, fc + 0.5f,
                static_cast<float>(iview) + 0.5f);
        }
        const float value = 0.5f * g.dtheta_rad * sum;
        volume[index] = accumulate ? volume[index] + value : value;
    }
}

} // namespace

void launchCurveFilteredFdkRebinPreweight(
    cudaTextureObject_t input_projection, float* output,
    const SCurveFilteredFdkGeometry& geometry,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t elements = static_cast<size_t>(geometry.output_t) *
        geometry.output_c * geometry.views;
    const auto launch = policy.make1D(elements);
    curveFilteredFdkRebinPreweightKernel<<<launch.grid, launch.block, 0, stream>>>(
        input_projection, output, geometry, elements);
    YK_CUDA_KERNEL_CHECK();
}

void launchCurveFilteredFdkBackprojection(
    cudaTextureObject_t filtered_projection, float* volume,
    const SCurveFilteredFdkGeometry& geometry,
    const SVolGeom& volume_geometry, bool accumulate,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t voxels = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    const auto launch = policy.make1D(voxels);
    curveFilteredFdkBackprojectionKernel<<<launch.grid, launch.block, 0, stream>>>(
        filtered_projection, volume, geometry, volume_geometry, accumulate, voxels);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::Fdk::detail
