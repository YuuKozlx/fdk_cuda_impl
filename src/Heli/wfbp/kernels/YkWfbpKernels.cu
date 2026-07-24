/*
 * wFBP rebinning and backprojection are adapted from FreeCT_wFBP.
 * Copyright (C) 2015 John Hoffman. GPL-2.0-or-later.
 * Adaptation copyright (C) 2026 YKCBCT contributors.
 *
 * The original fixed grids, global textures and raw CUDA allocations are not
 * retained. The kernels use the project's launch policy and linear buffers.
 */
#include "Heli/wfbp/kernels/YkWfbpLaunch.cuh"

#include <cmath>

#include "global/YkMacro.hpp"

namespace YK { namespace Helical { namespace Wfbp { namespace detail {
namespace {

__device__ float sample_projection(const float* data, int views, int rows, int channels,
    float view, int row, float channel)
{
    if (view < 0.f || view > views - 1.f || channel < 0.f || channel > channels - 1.f)
        return 0.f;
    const int v0 = min(static_cast<int>(floorf(view)), views - 1);
    const int v1 = min(v0 + 1, views - 1);
    const int u0 = min(static_cast<int>(floorf(channel)), channels - 1);
    const int u1 = min(u0 + 1, channels - 1);
    const float tv = view - v0;
    const float tu = channel - u0;
    const size_t row_stride = static_cast<size_t>(channels);
    const size_t view_stride = static_cast<size_t>(rows) * channels;
    const float a = data[static_cast<size_t>(v0) * view_stride + row * row_stride + u0];
    const float b = data[static_cast<size_t>(v0) * view_stride + row * row_stride + u1];
    const float c = data[static_cast<size_t>(v1) * view_stride + row * row_stride + u0];
    const float d = data[static_cast<size_t>(v1) * view_stride + row * row_stride + u1];
    return (a + tu * (b - a)) + tv * ((c + tu * (d - c)) - (a + tu * (b - a)));
}

__global__ void flat_to_arc_kernel(const float* flat, float* arc, Geometry g,
    float flat_du_mm, size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % g.input_channels);
        const size_t t = index / g.input_channels;
        const int row = static_cast<int>(t % g.rows);
        const int view = static_cast<int>(t / g.rows);
        const float beta = (channel - g.central_channel) * g.fan_angle_step;
        const float flat_channel = g.central_channel + g.sdd * tanf(beta) / flat_du_mm;
        arc[index] = sample_projection(flat, g.views, g.rows, g.input_channels,
            static_cast<float>(view), row, flat_channel);
    }
}

__global__ void rebin_kernel(const float* input, float* output, Geometry g, size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int p = static_cast<int>(index % g.output_channels);
        const size_t t = index / g.output_channels;
        const int row = static_cast<int>(t % g.rows);
        const int view = static_cast<int>(t / g.rows);

        // FreeCT n1_rebin/n2_rebin: parallel angle alpha is displaced from the
        // measured source angle by fan angle beta.
        const float sine_beta = (p - g.parallel_center) *
            (g.fan_angle_step / static_cast<float>(g.output_channels / g.input_channels));
        if (fabsf(sine_beta) >= 1.f) {
            output[index] = 0.f;
            continue;
        }
        const float beta = asinf(sine_beta);
        const float source_view = view - beta / g.angle_step;
        const float source_channel = beta / g.fan_angle_step + g.central_channel;
        output[index] = sample_projection(input, g.views, g.rows, g.input_channels,
            source_view, row, source_channel);
    }
}

__global__ void pad_kernel(const float* input, float* padded, int channels,
    int padded_channels, int rows, int views, size_t count)
{
    const int start = (padded_channels - channels) / 2;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int u = static_cast<int>(index % padded_channels);
        const size_t row = index / padded_channels;
        const int source_u = u - start;
        padded[index] = source_u >= 0 && source_u < channels
            ? input[row * channels + source_u] : 0.f;
    }
}

__global__ void crop_kernel(const float* padded, float* output, int channels,
    int padded_channels, size_t count)
{
    const int start = (padded_channels - channels) / 2;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int u = static_cast<int>(index % channels);
        const size_t row = index / channels;
        output[index] = padded[row * padded_channels + start + u];
    }
}

__device__ float redundancy_weight(float q, float flat)
{
    q = fabsf(q);
    if (q < flat) return 1.f;
    if (q >= 1.f) return 0.f;
    const float c = cosf(0.5f * CUDA_PI * (q - flat) / (1.f - flat));
    return c * c;
}

__device__ float sample_rebinned(const float* data, const Geometry& g,
    int view, float row, float channel)
{
    if (view < 0 || view >= g.views || row < 0.f || row > g.rows - 1.f ||
        channel < 0.f || channel > g.output_channels - 1.f) return 0.f;
    const int r0 = min(static_cast<int>(floorf(row)), g.rows - 1);
    const int r1 = min(r0 + 1, g.rows - 1);
    const int p0 = min(static_cast<int>(floorf(channel)), g.output_channels - 1);
    const int p1 = min(p0 + 1, g.output_channels - 1);
    const float tr = row - r0;
    const float tp = channel - p0;
    const size_t base = static_cast<size_t>(view) * g.rows * g.output_channels;
    const float a = data[base + static_cast<size_t>(r0) * g.output_channels + p0];
    const float b = data[base + static_cast<size_t>(r0) * g.output_channels + p1];
    const float c = data[base + static_cast<size_t>(r1) * g.output_channels + p0];
    const float d = data[base + static_cast<size_t>(r1) * g.output_channels + p1];
    return (a + tp * (b - a)) + tr * ((c + tp * (d - c)) - (a + tp * (b - a)));
}

__global__ void backproject_kernel(const float* filtered, float* volume,
    int nx, int ny, int nz, float dx, float dy, float dz,
    float ox, float oy, float oz, Geometry g, float flat, size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x_index = static_cast<int>(index % nx);
        const size_t yz = index / nx;
        const int y_index = static_cast<int>(yz % ny);
        const int z_index = static_cast<int>(yz / ny);
        const float x = (x_index - 0.5f * (nx - 1)) * dx + ox;
        const float y = (y_index - 0.5f * (ny - 1)) * dy + oy;
        const float z = (z_index - 0.5f * (nz - 1)) * dz + oz;
        if (x * x + y * y >= g.sid * g.sid) {
            volume[index] = 0.f;
            continue;
        }

        float result = 0.f;
        const int half_turn = max(1, g.views_per_turn / 2);
        // As in FreeCT bp_a/bp_b, samples sharing a half-turn phase are first
        // normalized by W(q), then all phases are integrated.
        for (int phase = 0; phase < half_turn; ++phase) {
            float weighted = 0.f;
            float weights = 0.f;
            for (int view = phase; view < g.views; view += half_turn) {
                const float theta = g.first_angle + view * g.angle_step;
                const float p = x * sinf(theta) - y * cosf(theta);
                if (fabsf(p) >= g.sid) continue;
                const float p_index = p / g.parallel_spacing + g.parallel_center;
                const float ray_z = g.start_z + g.pitch *
                    (theta - asinf(p / g.sid)) / (2.f * CUDA_PI);
                const float l = sqrtf(g.sid * g.sid - p * p) -
                    x * cosf(theta) - y * sinf(theta);
                const float cone = l * tanf(g.cone_half_angle);
                if (cone <= 1e-6f) continue;
                const float q = (z - ray_z) / cone;
                const float w = redundancy_weight(q, flat);
                if (w <= 0.f) continue;
                const float row = 0.5f * (q + 1.f) * (g.rows - 1);
                weighted += w * sample_rebinned(filtered, g, view, row, p_index);
                weights += w;
            }
            if (weights > 1e-6f) result += weighted / weights;
        }
        volume[index] = result * g.angle_step;
    }
}

} // namespace

void launch_flat_to_equiangular_arc(const float* flat, float* arc,
    const Geometry& geometry, float flat_du_mm,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(geometry.views) * geometry.rows *
        geometry.input_channels;
    const auto launch = policy.make1D(count);
    flat_to_arc_kernel<<<launch.grid, launch.block, 0, stream>>>(
        flat, arc, geometry, flat_du_mm, count);
    YK_CUDA_KERNEL_CHECK();
}

void launch_rebin(const float* input, float* output, const Geometry& geometry,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(geometry.views) * geometry.rows *
        geometry.output_channels;
    const auto launch = policy.make1D(count);
    rebin_kernel<<<launch.grid, launch.block, 0, stream>>>(input, output, geometry, count);
    YK_CUDA_KERNEL_CHECK();
}

void launch_pad(const float* input, float* padded, int channels, int padded_channels,
    int rows, int views, const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(views) * rows * padded_channels;
    const auto launch = policy.make1D(count);
    pad_kernel<<<launch.grid, launch.block, 0, stream>>>(input, padded, channels,
        padded_channels, rows, views, count);
    YK_CUDA_KERNEL_CHECK();
}

void launch_crop(const float* padded, float* output, int channels, int padded_channels,
    int rows, int views, const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = policy.make1D(count);
    crop_kernel<<<launch.grid, launch.block, 0, stream>>>(padded, output, channels,
        padded_channels, count);
    YK_CUDA_KERNEL_CHECK();
}

void launch_backproject(const float* filtered, float* volume,
    int nx, int ny, int nz, float dx, float dy, float dz,
    float ox, float oy, float oz, const Geometry& geometry,
    float redundancy_flat, const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    const auto launch = policy.make1D(count);
    backproject_kernel<<<launch.grid, launch.block, 0, stream>>>(filtered, volume,
        nx, ny, nz, dx, dy, dz, ox, oy, oz, geometry, redundancy_flat, count);
    YK_CUDA_KERNEL_CHECK();
}

} } } } // namespace YK::Helical::Wfbp::detail
