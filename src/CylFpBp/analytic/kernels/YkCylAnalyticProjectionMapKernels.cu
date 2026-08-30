#include "CylFpBp/analytic/YkCylAnalyticProjectionMapper.hpp"

#include "global/YkMacro.hpp"

namespace YK::CylFpBp::Analytic::detail {
namespace {

__device__ float sample_bilinear(const float* data, int rows, int channels,
    int view, float row, float channel)
{
    if (channel < 0.f || channel > channels - 1.f ||
        row < 0.f || row > rows - 1.f) return 0.f;
    const int u0 = static_cast<int>(floorf(channel));
    const int v0 = static_cast<int>(floorf(row));
    const int u1 = min(u0 + 1, channels - 1);
    const int v1 = min(v0 + 1, rows - 1);
    const float fu = channel - u0;
    const float fv = row - v0;
    const size_t base = static_cast<size_t>(view) * rows * channels;
    const float p00 = data[base + static_cast<size_t>(v0) * channels + u0];
    const float p01 = data[base + static_cast<size_t>(v0) * channels + u1];
    const float p10 = data[base + static_cast<size_t>(v1) * channels + u0];
    const float p11 = data[base + static_cast<size_t>(v1) * channels + u1];
    const float top = p00 + fu * (p01 - p00);
    const float bottom = p10 + fu * (p11 - p10);
    return top + fv * (bottom - top);
}

__global__ void cylindrical_to_equiangular_map_kernel(
    const float* physical, float* target, ProjectionMapConfig config,
    size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % config.channels);
        const size_t view_row = index / config.channels;
        const int row = static_cast<int>(view_row % config.rows);
        const int view = static_cast<int>(view_row / config.rows);

        // 目标射线位于以源点为轴心、半径 SDD 的虚拟圆柱。其横向射线
        // 与物理圆柱相交后得到源到交点距离 rho；选取 beta=0 时连续到
        // rho=SDD 的根，避免跳到圆柱背面。
        const float beta = (channel - config.target_principal_u) *
            config.target_angle_step_rad;
        const float center_distance = config.source_to_detector_mm -
            config.physical_radius_mm;
        float sine = 0.f, cosine = 0.f;
        sincosf(beta, &sine, &cosine);
        const float discriminant = config.physical_radius_mm *
            config.physical_radius_mm - center_distance * center_distance *
            sine * sine;
        if (discriminant < 0.f) {
            target[index] = 0.f;
            continue;
        }
        const float rho = center_distance * cosine + sqrtf(discriminant);
        if (!(rho > 1e-6f)) {
            target[index] = 0.f;
            continue;
        }

        const float surface_tangent = rho * sine;
        const float surface_radial = rho * cosine - center_distance;
        const float alpha = atan2f(surface_tangent, surface_radial);
        const float physical_channel = config.physical_principal_u +
            config.physical_radius_mm * alpha /
                config.physical_arc_step_mm;

        // 同一三维射线上轴向坐标与横向源距成比例。目标虚拟圆柱上的
        // q_target 位于 rho=SDD，映射到物理圆柱时乘 rho/SDD。
        const float q_target = (row - config.target_principal_v) *
            config.target_row_step_mm;
        const float physical_row = config.physical_principal_v +
            q_target * rho / config.source_to_detector_mm /
                config.physical_row_step_mm;
        target[index] = sample_bilinear(physical, config.rows,
            config.channels, view, physical_row, physical_channel);
    }
}

} // namespace

void launch_cylindrical_to_equiangular_map(const float* physical_projection,
    float* equiangular_projection, const ProjectionMapConfig& config,
    cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(config.channels) * config.rows *
        config.views;
    const auto launch = config.launch.make1D(count);
    cylindrical_to_equiangular_map_kernel<<<launch.grid, launch.block, 0,
        stream>>>(physical_projection, equiangular_projection, config, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::Analytic::detail
