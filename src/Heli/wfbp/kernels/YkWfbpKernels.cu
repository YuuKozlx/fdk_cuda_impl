/*
 * FreeCT_wFBP 算法适配。
 * Copyright (C) 2015 John Hoffman. GPL-2.0-or-later.
 * Adaptation copyright (C) 2026 YKCBCT contributors.
 *
 * 保留 FreeCT 的 fan-to-parallel、FFS 焦点校正、r(t) 滤波核以及 W(q)
 * 半圈归一化；仅将全局纹理、固定网格和裸显存改为本工程基础设施。
 */
#include "Heli/wfbp/kernels/YkWfbpLaunch.cuh"

#include <cmath>

#include "global/YkMacro.hpp"

namespace YK { namespace Helical { namespace Wfbp { namespace detail {
namespace {

__device__ float sample_projection(const float* data, int views, int rows, int channels,
    float view, float row, float channel)
{
    if (view < 0.f || view > views - 1.f || row < 0.f || row > rows - 1.f ||
        channel < 0.f || channel > channels - 1.f) return 0.f;
    const int v0 = min(static_cast<int>(floorf(view)), views - 1);
    const int v1 = min(v0 + 1, views - 1);
    const int r0 = min(static_cast<int>(floorf(row)), rows - 1);
    const int r1 = min(r0 + 1, rows - 1);
    const int u0 = min(static_cast<int>(floorf(channel)), channels - 1);
    const int u1 = min(u0 + 1, channels - 1);
    const float tv = view - v0;
    const float tr = row - r0;
    const float tu = channel - u0;
    const size_t view_stride = static_cast<size_t>(rows) * channels;
    const size_t row_stride = channels;
    const size_t v0r0 = static_cast<size_t>(v0) * view_stride +
        static_cast<size_t>(r0) * row_stride;
    const size_t v0r1 = static_cast<size_t>(v0) * view_stride +
        static_cast<size_t>(r1) * row_stride;
    const size_t v1r0 = static_cast<size_t>(v1) * view_stride +
        static_cast<size_t>(r0) * row_stride;
    const size_t v1r1 = static_cast<size_t>(v1) * view_stride +
        static_cast<size_t>(r1) * row_stride;
    const float a = data[v0r0 + u0] + tu * (data[v0r0 + u1] - data[v0r0 + u0]);
    const float b = data[v0r1 + u0] + tu * (data[v0r1 + u1] - data[v0r1 + u0]);
    const float c = data[v1r0 + u0] + tu * (data[v1r0 + u1] - data[v1r0 + u0]);
    const float d = data[v1r1 + u0] + tu * (data[v1r1 + u1] - data[v1r1 + u0]);
    return (a + tr * (b - a)) + tv * ((c + tr * (d - c)) - (a + tr * (b - a)));
}

__global__ void flat_to_arc_kernel(const float* flat, float* arc, Geometry g,
    float flat_du_mm, size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % g.input_channels);
        const size_t t = index / g.input_channels;
        const int row = static_cast<int>(t % g.input_rows);
        const int view = static_cast<int>(t / g.input_rows);
        const float beta = (channel - g.central_channel) * g.fan_angle_step;
        const float flat_channel = g.central_channel + g.sdd * tanf(beta) / flat_du_mm;
        arc[index] = sample_projection(flat, g.raw_views, g.input_rows,
            g.input_channels, static_cast<float>(view), static_cast<float>(row),
            flat_channel);
    }
}

__device__ float signed_angle(float x1, float x2, float y1, float y2)
{
    const float denominator = hypotf(x1, x2) * hypotf(y1, y2);
    return asinf(fmaxf(-1.f, fminf(1.f, (x1 * y2 - x2 * y1) / denominator)));
}

__device__ float focal_alpha_shift(float sid, float da, float dr)
{
    return signed_angle(sid, 0.f, sid + dr, da);
}

__device__ float focal_beta(Geometry g, float channel, float da, float dr)
{
    const float b0 = (channel - g.central_channel) * g.fan_angle_step;
    return signed_angle(-(g.sid + dr), -da,
        -(g.sdd * cosf(b0) + dr), -(g.sdd * sinf(b0) + da));
}

__device__ int focal_spot_index(const Geometry& g, int phi_spot, int z_spot)
{
    if (g.focal_spot_mode == EFocalSpotMode::PhiAndZ)
        return 2 * z_spot + phi_spot; // FreeCT a1,a2,a3,a4
    if (g.focal_spot_mode == EFocalSpotMode::Phi) return phi_spot;
    if (g.focal_spot_mode == EFocalSpotMode::Z) return z_spot;
    return 0;
}

__device__ void focal_offsets(const Geometry& g, int phi_spot, int z_spot,
    float& da, float& dr)
{
    da = g.phi_spot_count == 2 ? (phi_spot == 0 ? g.phi_ffs_shift : -g.phi_ffs_shift)
                                : 0.f;
    dr = g.z_spot_count == 2 ? (z_spot == 0 ? -g.z_ffs_shift : g.z_ffs_shift)
                              : 0.f;
}

__device__ int z_spot_for_output_row(const Geometry& g, int output_row)
{
    if (g.z_spot_count == 1) return 0;
    const int parity = output_row & 1;
    return g.reverse_row_interleave ? 1 - parity : parity;
}

__device__ float raw_row_for_output(const Geometry& g, int output_row)
{
    return g.z_spot_count == 1 ? static_cast<float>(output_row)
        : static_cast<float>(output_row / 2);
}

__device__ float intermediate_beta(const Geometry& g, float intermediate,
    int z_spot)
{
    const int i0 = max(0, min(static_cast<int>(floorf(intermediate)),
        g.input_channels * g.phi_spot_count - 1));
    const float channel = static_cast<float>(i0 / g.phi_spot_count);
    const int phi_spot = i0 % g.phi_spot_count;
    float da = 0.f, dr = 0.f;
    focal_offsets(g, phi_spot, z_spot, da, dr);
    return focal_beta(g, channel, da, dr);
}

__device__ float lookup_intermediate_index(const Geometry& g, float target_beta,
    int z_spot)
{
    // FreeCT p/a 路径先将两个 phi 焦点交错成 2*Nu 通道 beta lookup，随后
    // 再反查到统一平行通道。无 phi-FFS 时该表退化为普通 Nu 通道。
    int low = 0;
    int high = g.input_channels * g.phi_spot_count - 1;
    const float first = intermediate_beta(g, static_cast<float>(low), z_spot);
    const float last = intermediate_beta(g, static_cast<float>(high), z_spot);
    const bool increasing = last >= first;
    if ((increasing && (target_beta < first || target_beta > last)) ||
        (!increasing && (target_beta > first || target_beta < last))) return -1.f;
    while (high - low > 1) {
        const int mid = low + (high - low) / 2;
        const float value = intermediate_beta(g, static_cast<float>(mid), z_spot);
        if ((value < target_beta) == increasing) low = mid;
        else high = mid;
    }
    const float beta_low = intermediate_beta(g, static_cast<float>(low), z_spot);
    const float beta_high = intermediate_beta(g, static_cast<float>(high), z_spot);
    const float denominator = beta_high - beta_low;
    if (fabsf(denominator) < 1e-8f) return static_cast<float>(low);
    return static_cast<float>(low) + (target_beta - beta_low) / denominator;
}

__device__ float parallel_target_beta(const Geometry& g, int output_channel,
    int z_spot)
{
    float sine_beta = (output_channel - g.parallel_center) *
        (g.fan_angle_step * 0.5f);
    if (g.z_spot_count == 2) {
        // FreeCT z/a 路径第二阶段不是直接使用名义扇角：两个轴向焦点的
        // 有效旋转半径不同，目标平行坐标需乘 r_f / r_fr(0, +/-dr)。
        // 这里故意不带 phi 偏移；官方 a1_rebin_b/a2_rebin_b 同样只用 dr。
        float da = 0.f, dr = 0.f;
        focal_offsets(g, 0, z_spot, da, dr);
        const float effective_radius = fabsf(g.sid + dr);
        if (effective_radius <= 1e-6f) return nanf("");
        sine_beta *= g.sid / effective_radius;
    }
    if (fabsf(sine_beta) >= 1.f) return nanf("");
    return asinf(sine_beta);
}

__device__ float sample_intermediate(const float* input, const Geometry& g,
    int sequence_output_view, float raw_row, int z_spot, int intermediate)
{
    intermediate = max(0, min(intermediate,
        g.input_channels * g.phi_spot_count - 1));
    const int phi_spot = intermediate % g.phi_spot_count;
    const float channel = static_cast<float>(intermediate / g.phi_spot_count);
    float da = 0.f, dr = 0.f;
    focal_offsets(g, phi_spot, z_spot, da, dr);
    const float beta = focal_beta(g, channel, da, dr);
    const int spot = focal_spot_index(g, phi_spot, z_spot);
    // FreeCT 先按 spot 拆分序列，再在该序列内插值。sequence_view 的
    // -spot/n_ffs 项对应官方 p2/a2/a3/a4 中的 -1/-2/-3 相位项。
    const float sequence_view = sequence_output_view -
        (beta + focal_alpha_shift(g.sid, da, dr)) / g.angle_step -
        static_cast<float>(spot) / g.focal_spot_count;
    if (sequence_view < 0.f || sequence_view > g.sequence_views - 1.f) return 0.f;
    const int v0 = min(static_cast<int>(floorf(sequence_view)), g.sequence_views - 1);
    const int v1 = min(v0 + 1, g.sequence_views - 1);
    const float tv = sequence_view - v0;
    const float raw0 = static_cast<float>(v0 * g.focal_spot_count + spot);
    const float raw1 = static_cast<float>(v1 * g.focal_spot_count + spot);
    const float a = sample_projection(input, g.raw_views, g.input_rows,
        g.input_channels, raw0, raw_row, channel);
    const float b = sample_projection(input, g.raw_views, g.input_rows,
        g.input_channels, raw1, raw_row, channel);
    return a + tv * (b - a);
}

__global__ void rebin_kernel(const float* input, float* output, Geometry g,
    size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int p = static_cast<int>(index % g.output_channels);
        const size_t t = index / g.output_channels;
        const int row = static_cast<int>(t % g.rows);
        const int view = static_cast<int>(t / g.rows);
        const int sequence_output_view = view + g.add_projections;

        const int z_spot = z_spot_for_output_row(g, row);
        const float target_beta = parallel_target_beta(g, p, z_spot);
        if (!isfinite(target_beta)) {
            output[index] = 0.f;
            continue;
        }
        const float raw_row = raw_row_for_output(g, row);
        if (raw_row < 0.f || raw_row > g.input_rows - 1.f) {
            output[index] = 0.f;
            continue;
        }
        const float intermediate = lookup_intermediate_index(g, target_beta, z_spot);
        if (intermediate < 0.f) {
            output[index] = 0.f;
            continue;
        }
        const int i0 = static_cast<int>(floorf(intermediate));
        const int i1 = min(i0 + 1, g.input_channels * g.phi_spot_count - 1);
        const float t_beta = intermediate - i0;
        const float a = sample_intermediate(input, g, sequence_output_view,
            raw_row, z_spot, i0);
        const float b = sample_intermediate(input, g, sequence_output_view,
            raw_row, z_spot, i1);
        output[index] = a + t_beta * (b - a);
    }
}

__global__ void pad_kernel(const float* input, float* padded, int channels,
    int padded_channels, size_t count)
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

__device__ float freect_r(float t)
{
    if (fabsf(t) < 1e-6f) return 0.5f;
    return sinf(t) / t + (cosf(t) - 1.f) / (t * t);
}

__global__ void build_freect_filter_kernel(float* filter, int length,
    float spacing, FreeCtFilterConfig config)
{
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < length;
         u += gridDim.x * blockDim.x) {
        // 官方先在 [-N/2,N/2) 生成，再 fftshift；等价于以下循环索引。
        const int n = u <= length / 2 ? u : u - length;
        const float t = CUDA_PI * config.cutoff_c * static_cast<float>(n);
        const float center = config.apodization_a * freect_r(t);
        const float side = 0.5f * (1.f - config.apodization_a) *
            (freect_r(t + CUDA_PI) + freect_r(t - CUDA_PI));
        filter[u] = (config.cutoff_c * config.cutoff_c / (2.f * spacing)) *
            (center + side);
    }
}

__global__ void multiply_filter_kernel(cufftComplex* data,
    const cufftComplex* filter, int complex_channels, int batch, float scale,
    size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(index % complex_channels);
        const cufftComplex a = data[index];
        const cufftComplex b = filter[k];
        data[index] = make_cuFloatComplex(
            scale * (a.x * b.x - a.y * b.y),
            scale * (a.x * b.y + a.y * b.x));
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
        // FreeCT W(q)：同一半圈相位的多圈观测先按 W(q) 归一化，随后
        // 对半圈内各相位按 dtheta 积分。
        for (int phase = 0; phase < half_turn; ++phase) {
            float weighted = 0.f;
            float weights = 0.f;
            for (int view = phase; view < g.views; view += half_turn) {
                const float theta = g.first_angle + view * g.angle_step;
                const float p = x * sinf(theta) - y * cosf(theta);
                if (fabsf(p) >= g.sid) continue;
                const float p_index = p / g.parallel_spacing + g.parallel_center;
                const float l = sqrtf(g.sid * g.sid - p * p) -
                    x * cosf(theta) - y * sinf(theta);
                if (l <= 1e-6f || g.cone_half_angle <= 1e-6f) continue;
                const float table_z = g.start_z + g.pitch * theta / (2.f * CUDA_PI);
                const float q = (z - table_z +
                    g.pitch * asinf(p / g.sid) / (2.f * CUDA_PI)) /
                    (l * tanf(g.cone_half_angle));
                const float w = redundancy_weight(q, flat);
                if (w <= 0.f) continue;
                // principal row 显式进入行映射，因此非零 V offset 不再被丢弃。
                const float row = g.central_row + q *
                    fminf(g.central_row, static_cast<float>(g.rows - 1) - g.central_row);
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
    const size_t count = static_cast<size_t>(geometry.raw_views) *
        geometry.input_rows * geometry.input_channels;
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
        padded_channels, count);
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

void launch_build_freect_filter(float* spatial_filter, int length,
    float sample_spacing, const FreeCtFilterConfig& config,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const auto launch = policy.make1D(static_cast<size_t>(length));
    build_freect_filter_kernel<<<launch.grid, launch.block, 0, stream>>>(
        spatial_filter, length, sample_spacing, config);
    YK_CUDA_KERNEL_CHECK();
}

void launch_multiply_filter(cufftComplex* data, const cufftComplex* filter,
    int complex_channels, int batch, float scale,
    const SKernelLaunchPolicy& policy, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(complex_channels) * batch;
    const auto launch = policy.make1D(count);
    multiply_filter_kernel<<<launch.grid, launch.block, 0, stream>>>(data, filter,
        complex_channels, batch, scale, count);
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
