#include "CylFpBp/Analytic/YkCylAnalyticProjectionMapper.hpp"

#include "global/YkMacro.hpp"
#include "global/YkCudaTextureController.hpp"

namespace YK::CylFpBp::Analytic::detail {
namespace {

__device__ float sample_linear(const float* data, int rows, int channels,
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

__device__ float cubic_weight(float x)
{
    x = fabsf(x);
    if (x <= 1.f) return 1.5f * x * x * x - 2.5f * x * x + 1.f;
    if (x < 2.f) return -0.5f * x * x * x + 2.5f * x * x - 4.f * x + 2.f;
    return 0.f;
}

__device__ float sample_cubic(const float* data, int rows, int channels,
    int view, float row, float channel)
{
    if (channel < 0.f || channel > channels - 1.f ||
        row < 0.f || row > rows - 1.f) return 0.f;
    const int cx = static_cast<int>(floorf(channel));
    const int cy = static_cast<int>(floorf(row));
    const float fu = channel - cx;
    const float fv = row - cy;
    const size_t base = static_cast<size_t>(view) * rows * channels;
    float sum = 0.f;
    float norm = 0.f;
    for (int j = -1; j <= 2; ++j) {
        const int y = min(max(cy + j, 0), rows - 1);
        const float wy = cubic_weight(static_cast<float>(j) - fv);
        for (int i = -1; i <= 2; ++i) {
            const int x = min(max(cx + i, 0), channels - 1);
            const float wx = cubic_weight(static_cast<float>(i) - fu);
            const float w = wx * wy;
            sum += w * data[base + static_cast<size_t>(y) * channels + x];
            norm += w;
        }
    }
    return norm > 1e-6f ? sum / norm : 0.f;
}

__device__ float sinc_pi(float x)
{
    const float ax = fabsf(x);
    if (ax < 1e-5f) return 1.f;
    const float p = 3.14159265358979323846f * x;
    return sinf(p) / p;
}

__device__ float lanczos3_weight(float x)
{
    const float ax = fabsf(x);
    return ax < 3.f ? sinc_pi(x) * sinc_pi(x / 3.f) : 0.f;
}

__device__ float sample_lanczos3(const float* data, int rows, int channels,
    int view, float row, float channel)
{
    if (channel < 0.f || channel > channels - 1.f ||
        row < 0.f || row > rows - 1.f) return 0.f;
    const int cx = static_cast<int>(floorf(channel));
    const int cy = static_cast<int>(floorf(row));
    const float base_x = channel - cx;
    const float base_y = row - cy;
    const size_t base = static_cast<size_t>(view) * rows * channels;
    float sum = 0.f;
    float norm = 0.f;
    for (int j = -2; j <= 3; ++j) {
        const int y = min(max(cy + j, 0), rows - 1);
        const float wy = lanczos3_weight(static_cast<float>(j) - base_y);
        for (int i = -2; i <= 3; ++i) {
            const int x = min(max(cx + i, 0), channels - 1);
            const float wx = lanczos3_weight(static_cast<float>(i) - base_x);
            const float w = wx * wy;
            sum += w * data[base + static_cast<size_t>(y) * channels + x];
            norm += w;
        }
    }
    return norm > 1e-6f ? sum / norm : 0.f;
}

__global__ void cylindrical_to_flat_map_kernel(
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

        // 目标是局部姿态为零的虚拟平板。对每个平板像素构造射线，
        // 再求该射线与真实物理柱面的交点并反向采样。
        const float u = (channel - config.target_principal_u) *
            config.target_du_mm;
        const float v = (row - config.target_principal_v) *
            config.target_row_step_mm;
        const float D = config.source_to_detector_mm;
        const float c = config.physical_source_axis_mm;
        // 目标平板射线 d=(D,u,v) 与真实柱面求交。由于柱面轴与
        // detector V 平行，v 不进入横向二次方程，但会通过 lambda
        // 进入最终轴向坐标 q=lambda*v。
        const float A = D * D + u * u;
        const float B = -c * D;
        const float C = c * c - config.physical_radius_mm *
            config.physical_radius_mm;
        const float discriminant = B * B - A * C;
        if (!(A > 1e-12f) || discriminant < 0.f) {
            target[index] = 0.f;
            continue;
        }
        const float lambda = (-B + sqrtf(discriminant)) / A;
        if (!(lambda > 1e-6f)) {
            target[index] = 0.f;
            continue;
        }

        const float surface_tangent = lambda * u;
        const float surface_radial = lambda * D - c;
        const float alpha = atan2f(surface_tangent, surface_radial);
        const float alpha_center =
            (config.physical_principal_u -
                0.5f * static_cast<float>(config.channels - 1)) *
            config.physical_arc_step_mm / config.physical_radius_mm;
        const float physical_channel =
            0.5f * static_cast<float>(config.channels - 1) +
            (alpha - alpha_center) * config.physical_radius_mm /
                config.physical_arc_step_mm;

        // 物理柱面交点相对柱面中心的轴向坐标为 lambda*v；
        // principal_v 已将 detector center 的轴向偏移编码进去。
        const float q = config.physical_source_axis_axial_mm + lambda * v;
        const float physical_row =
            0.5f * static_cast<float>(config.rows - 1) +
            (q - config.physical_center_axial_mm) /
                config.physical_row_step_mm;
        target[index] = config.interpolation_order >= ProjectionMapConfig::Lanczos3
            ? sample_lanczos3(physical, config.rows, config.channels, view,
                physical_row, physical_channel)
            : config.interpolation_order >= ProjectionMapConfig::Cubic
            ? sample_cubic(physical, config.rows, config.channels, view,
                physical_row, physical_channel)
            : sample_linear(physical, config.rows, config.channels, view,
                physical_row, physical_channel);
    }
}

} // namespace

void launch_cylindrical_to_flat_map(const float* physical_projection,
    float* flat_projection, const ProjectionMapConfig& config,
    cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(config.channels) * config.rows *
        config.views;
    const auto launch = config.launch.make1D(count);
    cylindrical_to_flat_map_kernel<<<launch.grid, launch.block, 0,
        stream>>>(physical_projection, flat_projection, config, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::Analytic::detail
