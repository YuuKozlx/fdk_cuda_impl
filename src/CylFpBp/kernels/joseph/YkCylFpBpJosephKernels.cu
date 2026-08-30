#include "CylFpBp/kernels/joseph/YkCylJosephLaunch.cuh"

#include <cmath>

#include "common/cuda/operators/YkOperatorKernelTypes.cuh"
#include "global/YkMacro.hpp"

namespace YK::CylFpBp::detail {
namespace {

__device__ float3 add3(float3 a, float3 b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 sub3(float3 a, float3 b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 mul3(float3 a, float s)
{ return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float dot3(float3 a, float3 b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float norm3(float3 a) { return sqrtf(dot3(a, a)); }
__device__ float3 to3(float4 a) { return make_float3(a.x, a.y, a.z); }

__device__ float3 detector_pixel(const SKernelView& g, int channel, int row)
{
    const float delta = (channel - g.principal_u) * g.channel_angle_step_rad;
    float sine = 0.f, cosine = 0.f;
    sincosf(delta, &sine, &cosine);
    const float3 surface = add3(to3(g.cylinder_center),
        add3(mul3(to3(g.radial_unit), g.radius_mm * cosine),
             mul3(to3(g.tangent_unit), g.radius_mm * sine)));
    return add3(surface, mul3(to3(g.detector_v), row - g.principal_v));
}

__device__ bool ray_box(float3 source, float3 direction, const SVolGeom& vg,
    float& t_min, float& t_max)
{
    const float3 origin = vg.origin();
    const float3 lower = make_float3(origin.x - 0.5f * vg.vox_x,
        origin.y - 0.5f * vg.vox_y, origin.z - 0.5f * vg.vox_z);
    const float3 upper = make_float3(origin.x + (vg.Nx - 0.5f) * vg.vox_x,
        origin.y + (vg.Ny - 0.5f) * vg.vox_y,
        origin.z + (vg.Nz - 0.5f) * vg.vox_z);
    t_min = 0.f;
    t_max = 1.f;
    const float s[3] = { source.x, source.y, source.z };
    const float d[3] = { direction.x, direction.y, direction.z };
    const float lo[3] = { lower.x, lower.y, lower.z };
    const float hi[3] = { upper.x, upper.y, upper.z };
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1e-12f) {
            if (s[axis] < lo[axis] || s[axis] > hi[axis]) return false;
            continue;
        }
        float a = (lo[axis] - s[axis]) / d[axis];
        float b = (hi[axis] - s[axis]) / d[axis];
        if (a > b) { const float tmp = a; a = b; b = tmp; }
        t_min = fmaxf(t_min, a);
        t_max = fminf(t_max, b);
        if (t_max <= t_min) return false;
    }
    return t_max > 0.f;
}

struct JosephWeights {
    size_t indices[4]{};
    float weights[4]{};
};

// 主轴只选择一个最近体素层，另外两个轴做双线性插值。因此每个采样点
// 只有 4 个读写位置；FP 和 BP 共用此函数，保证二者使用相同离散权重。
__device__ bool make_joseph_weights(float3 p, int main_axis,
    const SVolGeom& vg, JosephWeights& result)
{
    const size_t slice = static_cast<size_t>(vg.Nx) * vg.Ny;
    if (main_axis == 0) {
        const int x = static_cast<int>(floorf(p.x + 0.5f));
        if (x < 0 || x >= vg.Nx || p.y < 0.f || p.y > vg.Ny - 1.f ||
            p.z < 0.f || p.z > vg.Nz - 1.f) return false;
        const int y0 = min(static_cast<int>(floorf(p.y)), vg.Ny - 1);
        const int z0 = min(static_cast<int>(floorf(p.z)), vg.Nz - 1);
        const int y1 = min(y0 + 1, vg.Ny - 1);
        const int z1 = min(z0 + 1, vg.Nz - 1);
        const float ty = p.y - y0, tz = p.z - z0;
        result.indices[0] = static_cast<size_t>(z0) * slice + static_cast<size_t>(y0) * vg.Nx + x;
        result.indices[1] = static_cast<size_t>(z0) * slice + static_cast<size_t>(y1) * vg.Nx + x;
        result.indices[2] = static_cast<size_t>(z1) * slice + static_cast<size_t>(y0) * vg.Nx + x;
        result.indices[3] = static_cast<size_t>(z1) * slice + static_cast<size_t>(y1) * vg.Nx + x;
        result.weights[0] = (1.f - ty) * (1.f - tz);
        result.weights[1] = ty * (1.f - tz);
        result.weights[2] = (1.f - ty) * tz;
        result.weights[3] = ty * tz;
        return true;
    }
    if (main_axis == 1) {
        const int y = static_cast<int>(floorf(p.y + 0.5f));
        if (y < 0 || y >= vg.Ny || p.x < 0.f || p.x > vg.Nx - 1.f ||
            p.z < 0.f || p.z > vg.Nz - 1.f) return false;
        const int x0 = min(static_cast<int>(floorf(p.x)), vg.Nx - 1);
        const int z0 = min(static_cast<int>(floorf(p.z)), vg.Nz - 1);
        const int x1 = min(x0 + 1, vg.Nx - 1);
        const int z1 = min(z0 + 1, vg.Nz - 1);
        const float tx = p.x - x0, tz = p.z - z0;
        result.indices[0] = static_cast<size_t>(z0) * slice + static_cast<size_t>(y) * vg.Nx + x0;
        result.indices[1] = static_cast<size_t>(z0) * slice + static_cast<size_t>(y) * vg.Nx + x1;
        result.indices[2] = static_cast<size_t>(z1) * slice + static_cast<size_t>(y) * vg.Nx + x0;
        result.indices[3] = static_cast<size_t>(z1) * slice + static_cast<size_t>(y) * vg.Nx + x1;
        result.weights[0] = (1.f - tx) * (1.f - tz);
        result.weights[1] = tx * (1.f - tz);
        result.weights[2] = (1.f - tx) * tz;
        result.weights[3] = tx * tz;
        return true;
    }
    const int z = static_cast<int>(floorf(p.z + 0.5f));
    if (z < 0 || z >= vg.Nz || p.x < 0.f || p.x > vg.Nx - 1.f ||
        p.y < 0.f || p.y > vg.Ny - 1.f) return false;
    const int x0 = min(static_cast<int>(floorf(p.x)), vg.Nx - 1);
    const int y0 = min(static_cast<int>(floorf(p.y)), vg.Ny - 1);
    const int x1 = min(x0 + 1, vg.Nx - 1);
    const int y1 = min(y0 + 1, vg.Ny - 1);
    const float tx = p.x - x0, ty = p.y - y0;
    result.indices[0] = static_cast<size_t>(z) * slice + static_cast<size_t>(y0) * vg.Nx + x0;
    result.indices[1] = static_cast<size_t>(z) * slice + static_cast<size_t>(y0) * vg.Nx + x1;
    result.indices[2] = static_cast<size_t>(z) * slice + static_cast<size_t>(y1) * vg.Nx + x0;
    result.indices[3] = static_cast<size_t>(z) * slice + static_cast<size_t>(y1) * vg.Nx + x1;
    result.weights[0] = (1.f - tx) * (1.f - ty);
    result.weights[1] = tx * (1.f - ty);
    result.weights[2] = (1.f - tx) * ty;
    result.weights[3] = tx * ty;
    return true;
}

__device__ int select_main_axis(float3 voxel_direction)
{
    const float ax = fabsf(voxel_direction.x);
    const float ay = fabsf(voxel_direction.y);
    const float az = fabsf(voxel_direction.z);
    return ax >= ay && ax >= az ? 0 : (ay >= az ? 1 : 2);
}

__device__ float main_component(float3 value, int axis)
{ return axis == 0 ? value.x : (axis == 1 ? value.y : value.z); }

// Point 纹理只负责缓存读取，插值仍使用 make_joseph_weights() 生成的完整
// 浮点权重。因此该路径与 Joseph BP 使用完全相同的离散权重。
__device__ float point_texture_joseph(cudaTextureObject_t texture, float3 p,
    int main_axis, const SVolGeom& vg)
{
    JosephWeights weights{};
    if (!make_joseph_weights(p, main_axis, vg, weights)) return 0.f;
    const size_t slice = static_cast<size_t>(vg.Nx) * vg.Ny;
    float value = 0.f;
    for (int i = 0; i < 4; ++i) {
        const int z = static_cast<int>(weights.indices[i] / slice);
        const size_t in_slice = weights.indices[i] -
            static_cast<size_t>(z) * slice;
        const int y = static_cast<int>(in_slice / vg.Nx);
        const int x = static_cast<int>(in_slice - static_cast<size_t>(y) * vg.Nx);
        value += tex3D<float>(texture, x + 0.5f, y + 0.5f, z + 0.5f) *
            weights.weights[i];
    }
    return value;
}

__global__ void joseph_forward_kernel(const float* __restrict__ volume,
    float* __restrict__ projection, const SKernelView* __restrict__ geometry,
    int rows, int channels, SVolGeom vg, float samples_per_voxel,
    size_t count, CudaOp::EWriteMode write_mode)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t t = index / channels;
        const int row = static_cast<int>(t % rows);
        const int view = static_cast<int>(t / rows);
        const float3 source = to3(geometry[view].source);
        const float3 direction = sub3(detector_pixel(geometry[view], channel, row), source);
        const float ray_length = norm3(direction);
        float t_min = 0.f, t_max = 0.f, sum = 0.f;
        if (ray_box(source, direction, vg, t_min, t_max)) {
            const float3 voxel_direction = make_float3(
                direction.x * vg.tmp_rcp_vox_x,
                direction.y * vg.tmp_rcp_vox_y,
                direction.z * vg.tmp_rcp_vox_z);
            const int main_axis = select_main_axis(voxel_direction);
            const float main_span = fabsf(main_component(voxel_direction, main_axis)) *
                (t_max - t_min);
            const int samples = max(1, static_cast<int>(ceilf(
                main_span * samples_per_voxel)));
            const float dt = (t_max - t_min) / samples;
            const float sample_length = ray_length * dt;
            const float3 origin = vg.origin();
            const float first_t = t_min + 0.5f * dt;
            const float3 first = add3(source, mul3(direction, first_t));
            float3 voxel_position = make_float3(
                (first.x - origin.x) * vg.tmp_rcp_vox_x,
                (first.y - origin.y) * vg.tmp_rcp_vox_y,
                (first.z - origin.z) * vg.tmp_rcp_vox_z);
            const float3 voxel_step = mul3(voxel_direction, dt);
            for (int sample = 0; sample < samples; ++sample) {
                JosephWeights weights{};
                if (make_joseph_weights(voxel_position, main_axis, vg, weights))
                    for (int i = 0; i < 4; ++i)
                        sum += __ldg(volume + weights.indices[i]) *
                            weights.weights[i] * sample_length;
                voxel_position = add3(voxel_position, voxel_step);
            }
        }
        if (CudaOp::accumulates(write_mode)) projection[index] += sum;
        else projection[index] = sum;
    }
}

template <bool MatchedPointTexture>
__global__ void joseph_forward_texture_kernel(cudaTextureObject_t volume_texture,
    float* __restrict__ projection, const SKernelView* __restrict__ geometry,
    int rows, int channels, SVolGeom vg, float samples_per_voxel,
    size_t count, CudaOp::EWriteMode write_mode)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t t = index / channels;
        const int row = static_cast<int>(t % rows);
        const int view = static_cast<int>(t / rows);
        const float3 source = to3(geometry[view].source);
        const float3 direction = sub3(detector_pixel(geometry[view], channel, row), source);
        const float ray_length = norm3(direction);
        float t_min = 0.f, t_max = 0.f, sum = 0.f;
        if (ray_box(source, direction, vg, t_min, t_max)) {
            const float3 voxel_direction = make_float3(
                direction.x * vg.tmp_rcp_vox_x,
                direction.y * vg.tmp_rcp_vox_y,
                direction.z * vg.tmp_rcp_vox_z);
            const int main_axis = select_main_axis(voxel_direction);
            const float main_span = fabsf(main_component(voxel_direction, main_axis)) *
                (t_max - t_min);
            const int samples = max(1, static_cast<int>(ceilf(
                main_span * samples_per_voxel)));
            const float dt = (t_max - t_min) / samples;
            const float sample_length = ray_length * dt;
            const float3 origin = vg.origin();
            const float first_t = t_min + 0.5f * dt;
            const float3 first = add3(source, mul3(direction, first_t));
            float3 p = make_float3(
                (first.x - origin.x) * vg.tmp_rcp_vox_x,
                (first.y - origin.y) * vg.tmp_rcp_vox_y,
                (first.z - origin.z) * vg.tmp_rcp_vox_z);
            const float3 step = mul3(voxel_direction, dt);
            for (int sample = 0; sample < samples; ++sample) {
                if constexpr (MatchedPointTexture) {
                    sum += point_texture_joseph(volume_texture, p, main_axis, vg) *
                        sample_length;
                }
                else {
                    float3 texture_position = make_float3(
                        p.x + 0.5f, p.y + 0.5f, p.z + 0.5f);
                    bool valid = false;
                    if (main_axis == 0) {
                        const int layer = static_cast<int>(floorf(p.x + 0.5f));
                        valid = layer >= 0 && layer < vg.Nx && p.y >= 0.f &&
                            p.y <= vg.Ny - 1.f && p.z >= 0.f &&
                            p.z <= vg.Nz - 1.f;
                        texture_position.x = layer + 0.5f;
                    }
                    else if (main_axis == 1) {
                        const int layer = static_cast<int>(floorf(p.y + 0.5f));
                        valid = layer >= 0 && layer < vg.Ny && p.x >= 0.f &&
                            p.x <= vg.Nx - 1.f && p.z >= 0.f &&
                            p.z <= vg.Nz - 1.f;
                        texture_position.y = layer + 0.5f;
                    }
                    else {
                        const int layer = static_cast<int>(floorf(p.z + 0.5f));
                        valid = layer >= 0 && layer < vg.Nz && p.x >= 0.f &&
                            p.x <= vg.Nx - 1.f && p.y >= 0.f &&
                            p.y <= vg.Ny - 1.f;
                        texture_position.z = layer + 0.5f;
                    }
                    if (valid)
                        sum += tex3D<float>(volume_texture, texture_position.x,
                            texture_position.y, texture_position.z) * sample_length;
                }
                p = add3(p, step);
            }
        }
        if (CudaOp::accumulates(write_mode)) projection[index] += sum;
        else projection[index] = sum;
    }
}

__global__ void joseph_backproject_kernel(const float* __restrict__ projection,
    float* __restrict__ volume, const SKernelView* __restrict__ geometry,
    int rows, int channels, SVolGeom vg, float samples_per_voxel, size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float ray_value = projection[index];
        if (ray_value == 0.f) continue;
        const int channel = static_cast<int>(index % channels);
        const size_t t = index / channels;
        const int row = static_cast<int>(t % rows);
        const int view = static_cast<int>(t / rows);
        const float3 source = to3(geometry[view].source);
        const float3 direction = sub3(detector_pixel(geometry[view], channel, row), source);
        const float ray_length = norm3(direction);
        float t_min = 0.f, t_max = 0.f;
        if (!ray_box(source, direction, vg, t_min, t_max)) continue;
        const float3 voxel_direction = make_float3(
            direction.x * vg.tmp_rcp_vox_x,
            direction.y * vg.tmp_rcp_vox_y,
            direction.z * vg.tmp_rcp_vox_z);
        const int main_axis = select_main_axis(voxel_direction);
        const float main_span = fabsf(main_component(voxel_direction, main_axis)) *
            (t_max - t_min);
        const int samples = max(1, static_cast<int>(ceilf(
            main_span * samples_per_voxel)));
        const float dt = (t_max - t_min) / samples;
        const float contribution = ray_value * ray_length * dt;
        const float3 origin = vg.origin();
        const float first_t = t_min + 0.5f * dt;
        const float3 first = add3(source, mul3(direction, first_t));
        float3 voxel_position = make_float3(
            (first.x - origin.x) * vg.tmp_rcp_vox_x,
            (first.y - origin.y) * vg.tmp_rcp_vox_y,
            (first.z - origin.z) * vg.tmp_rcp_vox_z);
        const float3 voxel_step = mul3(voxel_direction, dt);
        for (int sample = 0; sample < samples; ++sample) {
            JosephWeights weights{};
            if (make_joseph_weights(voxel_position, main_axis, vg, weights))
                for (int i = 0; i < 4; ++i)
                    if (weights.weights[i] != 0.f)
                        atomicAdd(volume + weights.indices[i],
                            contribution * weights.weights[i]);
            voxel_position = add3(voxel_position, voxel_step);
        }
    }
}

// 投影数组使用 Point 纹理读取，BP 保留完整浮点 Joseph 散射权重，不进行
// 近似权重处理。它是软件浮点 Joseph FP 的离散转置；与硬件纹理 FP 配对
// 时不再严格 matched，原因是 FP 的 Linear 纹理会量化插值小数。
__global__ void joseph_backproject_texture_kernel(
    cudaTextureObject_t projection_texture, float* __restrict__ volume,
    const SKernelView* __restrict__ geometry, int rows, int channels,
    SVolGeom vg, float samples_per_voxel, size_t count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t t = index / channels;
        const int row = static_cast<int>(t % rows);
        const int view = static_cast<int>(t / rows);
        const float ray_value = tex3D<float>(projection_texture,
            channel + 0.5f, row + 0.5f, view + 0.5f);
        if (ray_value == 0.f) continue;
        const float3 source = to3(geometry[view].source);
        const float3 direction = sub3(detector_pixel(geometry[view], channel, row), source);
        const float ray_length = norm3(direction);
        float t_min = 0.f, t_max = 0.f;
        if (!ray_box(source, direction, vg, t_min, t_max)) continue;
        const float3 voxel_direction = make_float3(
            direction.x * vg.tmp_rcp_vox_x,
            direction.y * vg.tmp_rcp_vox_y,
            direction.z * vg.tmp_rcp_vox_z);
        const int main_axis = select_main_axis(voxel_direction);
        const float main_span = fabsf(main_component(voxel_direction, main_axis)) *
            (t_max - t_min);
        const int samples = max(1, static_cast<int>(ceilf(
            main_span * samples_per_voxel)));
        const float dt = (t_max - t_min) / samples;
        const float contribution = ray_value * ray_length * dt;
        const float3 origin = vg.origin();
        const float first_t = t_min + 0.5f * dt;
        const float3 first = add3(source, mul3(direction, first_t));
        float3 voxel_position = make_float3(
            (first.x - origin.x) * vg.tmp_rcp_vox_x,
            (first.y - origin.y) * vg.tmp_rcp_vox_y,
            (first.z - origin.z) * vg.tmp_rcp_vox_z);
        const float3 voxel_step = mul3(voxel_direction, dt);
        for (int sample = 0; sample < samples; ++sample) {
            JosephWeights weights{};
            if (make_joseph_weights(voxel_position, main_axis, vg, weights))
                for (int i = 0; i < 4; ++i)
                    if (weights.weights[i] != 0.f)
                        atomicAdd(volume + weights.indices[i],
                            contribution * weights.weights[i]);
            voxel_position = add3(voxel_position, voxel_step);
        }
    }
}

} // namespace

void launch_main_axis_forward(const float* volume, float* projection,
    const SKernelView* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    joseph_forward_kernel<<<launch.grid, launch.block, 0, stream>>>(volume,
        projection, geometry, rows, channels, volume_geometry,
        config.samples_per_voxel, count, CudaOp::writeMode(accumulate));
    YK_CUDA_KERNEL_CHECK();
}

void launch_main_axis_forward_texture(cudaTextureObject_t volume_texture,
    float* projection, const SKernelView* geometry, int views, int rows,
    int channels, const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    joseph_forward_texture_kernel<false><<<launch.grid, launch.block, 0, stream>>>(
        volume_texture, projection, geometry, rows, channels, volume_geometry,
        config.samples_per_voxel, count, CudaOp::writeMode(accumulate));
    YK_CUDA_KERNEL_CHECK();
}

void launch_main_axis_forward_texture_matched(
    cudaTextureObject_t point_volume_texture, float* projection,
    const SKernelView* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    joseph_forward_texture_kernel<true><<<launch.grid, launch.block, 0, stream>>>(
        point_volume_texture, projection, geometry, rows, channels,
        volume_geometry, config.samples_per_voxel, count,
        CudaOp::writeMode(accumulate));
    YK_CUDA_KERNEL_CHECK();
}

void launch_main_axis_backproject(const float* projection, float* volume,
    const SKernelView* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    CudaOp::clearIfOverwrite(volume,
        static_cast<size_t>(volume_geometry.Nx) * volume_geometry.Ny *
            volume_geometry.Nz,
        CudaOp::writeMode(accumulate), stream);
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    joseph_backproject_kernel<<<launch.grid, launch.block, 0, stream>>>(projection,
        volume, geometry, rows, channels, volume_geometry,
        config.samples_per_voxel, count);
    YK_CUDA_KERNEL_CHECK();
}

void launch_main_axis_backproject_texture(cudaTextureObject_t projection_texture,
    float* volume, const SKernelView* geometry, int views, int rows,
    int channels, const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    CudaOp::clearIfOverwrite(volume,
        static_cast<size_t>(volume_geometry.Nx) * volume_geometry.Ny *
            volume_geometry.Nz,
        CudaOp::writeMode(accumulate), stream);
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    joseph_backproject_texture_kernel<<<launch.grid, launch.block, 0, stream>>>(
        projection_texture, volume, geometry, rows, channels, volume_geometry,
        config.samples_per_voxel, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
