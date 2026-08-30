#include "CylFpBp/kernels/legacy/YkCylLegacyLaunch.cuh"

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

__device__ float3 detector_pixel(const SKernelView& g,
    int channel, int row)
{
    const float delta = (channel - g.principal_u) *
        g.channel_angle_step_rad;
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
    // 参数 t=0 为源点，t=1 为探测器像素；不沿射线越过探测器继续积分。
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

__device__ float sample_trilinear(const float* __restrict__ volume,
    const SVolGeom& vg, float3 voxel_position)
{
    const float x = voxel_position.x;
    const float y = voxel_position.y;
    const float z = voxel_position.z;
    if (x < 0.f || x > vg.Nx - 1.f || y < 0.f || y > vg.Ny - 1.f ||
        z < 0.f || z > vg.Nz - 1.f) return 0.f;
    const int x0 = min(static_cast<int>(floorf(x)), vg.Nx - 1);
    const int y0 = min(static_cast<int>(floorf(y)), vg.Ny - 1);
    const int z0 = min(static_cast<int>(floorf(z)), vg.Nz - 1);
    const int x1 = min(x0 + 1, vg.Nx - 1);
    const int y1 = min(y0 + 1, vg.Ny - 1);
    const int z1 = min(z0 + 1, vg.Nz - 1);
    const float tx = x - x0, ty = y - y0, tz = z - z0;
    const size_t slice = static_cast<size_t>(vg.Nx) * vg.Ny;
    auto at = [&](int ix, int iy, int iz) {
        return __ldg(volume + static_cast<size_t>(iz) * slice +
            static_cast<size_t>(iy) * vg.Nx + ix);
    };
    const float c00 = at(x0, y0, z0) + tx * (at(x1, y0, z0) - at(x0, y0, z0));
    const float c10 = at(x0, y1, z0) + tx * (at(x1, y1, z0) - at(x0, y1, z0));
    const float c01 = at(x0, y0, z1) + tx * (at(x1, y0, z1) - at(x0, y0, z1));
    const float c11 = at(x0, y1, z1) + tx * (at(x1, y1, z1) - at(x0, y1, z1));
    const float c0 = c00 + ty * (c10 - c00);
    const float c1 = c01 + ty * (c11 - c01);
    return c0 + tz * (c1 - c0);
}

__device__ void scatter_trilinear(float* volume, const SVolGeom& vg,
    float3 voxel_position, float value)
{
    const float x = voxel_position.x;
    const float y = voxel_position.y;
    const float z = voxel_position.z;
    if (x < 0.f || x > vg.Nx - 1.f || y < 0.f || y > vg.Ny - 1.f ||
        z < 0.f || z > vg.Nz - 1.f) return;
    const int x0 = min(static_cast<int>(floorf(x)), vg.Nx - 1);
    const int y0 = min(static_cast<int>(floorf(y)), vg.Ny - 1);
    const int z0 = min(static_cast<int>(floorf(z)), vg.Nz - 1);
    const int x1 = min(x0 + 1, vg.Nx - 1);
    const int y1 = min(y0 + 1, vg.Ny - 1);
    const int z1 = min(z0 + 1, vg.Nz - 1);
    const float tx = x - x0, ty = y - y0, tz = z - z0;
    const size_t slice = static_cast<size_t>(vg.Nx) * vg.Ny;
    const int xs[2] = { x0, x1 }, ys[2] = { y0, y1 }, zs[2] = { z0, z1 };
    const float wx[2] = { 1.f - tx, tx }, wy[2] = { 1.f - ty, ty };
    const float wz[2] = { 1.f - tz, tz };
    for (int iz = 0; iz < 2; ++iz) for (int iy = 0; iy < 2; ++iy)
        for (int ix = 0; ix < 2; ++ix) {
            const float weight = wx[ix] * wy[iy] * wz[iz];
            if (weight == 0.f) continue;
            atomicAdd(volume + static_cast<size_t>(zs[iz]) * slice +
                static_cast<size_t>(ys[iy]) * vg.Nx + xs[ix], value * weight);
        }
}

__device__ int ray_samples(float ray_length, float t_min, float t_max,
    const SVolGeom& vg, float samples_per_voxel)
{
    const float length = ray_length * (t_max - t_min);
    const float min_voxel = fminf(vg.vox_x, fminf(vg.vox_y, vg.vox_z));
    return max(1, static_cast<int>(ceilf(length * samples_per_voxel / min_voxel)));
}

__global__ void forward_kernel(const float* __restrict__ volume,
    float* __restrict__ projection, const SKernelView* __restrict__ geometry,
    int views, int rows, int channels,
    SVolGeom vg, float samples_per_voxel, size_t count,
    CudaOp::EWriteMode write_mode)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t t = index / channels;
        const int row = static_cast<int>(t % rows);
        const int view = static_cast<int>(t / rows);
        const float3 source = to3(geometry[view].source);
        const float3 detector = detector_pixel(geometry[view], channel, row);
        const float3 direction = sub3(detector, source);
        const float ray_length = norm3(direction);
        float t_min = 0.f, t_max = 0.f;
        float sum = 0.f;
        if (ray_box(source, direction, vg, t_min, t_max)) {
            const int samples = ray_samples(ray_length, t_min, t_max, vg,
                samples_per_voxel);
            const float dt = (t_max - t_min) / samples;
            const float sample_length = ray_length * dt;
            const float3 origin = vg.origin();
            const float first_t = t_min + 0.5f * dt;
            const float3 first = add3(source, mul3(direction, first_t));
            float3 voxel_position = make_float3(
                (first.x - origin.x) * vg.tmp_rcp_vox_x,
                (first.y - origin.y) * vg.tmp_rcp_vox_y,
                (first.z - origin.z) * vg.tmp_rcp_vox_z);
            const float3 voxel_step = make_float3(
                direction.x * dt * vg.tmp_rcp_vox_x,
                direction.y * dt * vg.tmp_rcp_vox_y,
                direction.z * dt * vg.tmp_rcp_vox_z);
            for (int sample = 0; sample < samples; ++sample) {
                sum += sample_trilinear(volume, vg, voxel_position) * sample_length;
                voxel_position = add3(voxel_position, voxel_step);
            }
        }
        if (CudaOp::accumulates(write_mode)) projection[index] += sum;
        else projection[index] = sum;
    }
}

__global__ void backproject_kernel(const float* __restrict__ projection,
    float* __restrict__ volume, const SKernelView* __restrict__ geometry,
    int views, int rows, int channels,
    SVolGeom vg, float samples_per_voxel, size_t count)
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
        const float3 detector = detector_pixel(geometry[view], channel, row);
        const float3 direction = sub3(detector, source);
        const float ray_length = norm3(direction);
        float t_min = 0.f, t_max = 0.f;
        if (!ray_box(source, direction, vg, t_min, t_max)) continue;
        const int samples = ray_samples(ray_length, t_min, t_max, vg,
            samples_per_voxel);
        const float dt = (t_max - t_min) / samples;
        const float contribution = ray_value * ray_length * dt;
        const float3 origin = vg.origin();
        const float first_t = t_min + 0.5f * dt;
        const float3 first = add3(source, mul3(direction, first_t));
        float3 voxel_position = make_float3(
            (first.x - origin.x) * vg.tmp_rcp_vox_x,
            (first.y - origin.y) * vg.tmp_rcp_vox_y,
            (first.z - origin.z) * vg.tmp_rcp_vox_z);
        const float3 voxel_step = make_float3(
            direction.x * dt * vg.tmp_rcp_vox_x,
            direction.y * dt * vg.tmp_rcp_vox_y,
            direction.z * dt * vg.tmp_rcp_vox_z);
        for (int sample = 0; sample < samples; ++sample) {
            scatter_trilinear(volume, vg, voxel_position, contribution);
            voxel_position = add3(voxel_position, voxel_step);
        }
    }
}

} // namespace

void launch_precomputed_forward(const float* volume, float* projection,
    const SKernelView* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    forward_kernel<<<launch.grid, launch.block, 0, stream>>>(volume, projection,
        geometry, views, rows, channels, volume_geometry,
        config.samples_per_voxel, count, CudaOp::writeMode(accumulate));
    YK_CUDA_KERNEL_CHECK();
}

void launch_precomputed_backproject(const float* projection, float* volume,
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
    backproject_kernel<<<launch.grid, launch.block, 0, stream>>>(projection, volume,
        geometry, views, rows, channels, volume_geometry,
        config.samples_per_voxel, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
