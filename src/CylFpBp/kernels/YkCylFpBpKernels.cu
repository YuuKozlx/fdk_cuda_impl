#include "CylFpBp/kernels/YkCylFpBpLaunch.cuh"

#include <cmath>

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
__device__ float3 normalize3(float3 a)
{
    const float n = norm3(a);
    return n > 1e-12f ? mul3(a, 1.f / n) : make_float3(0.f, 0.f, 0.f);
}
__device__ float3 to3(float4 a) { return make_float3(a.x, a.y, a.z); }

__device__ float3 detector_pixel(const SCylConeProjGeomVec& g,
    int channel, int row)
{
    const float3 center = to3(g.detector_principal);
    const float3 tangent_unit = normalize3(to3(g.detector_u_tangent));
    const float3 v = to3(g.detector_v);
    const float arc_pixel = norm3(to3(g.detector_u_tangent));
    const float delta = (channel - g.principal_u) * arc_pixel / g.radius_mm;
    const float3 normal = normalize3(sub3(center, to3(g.source)));
    // ASTRA cyl_cone_vec: cylinder center = detector center - radial_axis.
    const float3 cylinder_center = sub3(center, mul3(normal, g.radius_mm));
    const float3 surface = add3(cylinder_center,
        add3(mul3(normal, g.radius_mm * cosf(delta)),
             mul3(tangent_unit, g.radius_mm * sinf(delta))));
    return add3(surface, mul3(v, row - g.principal_v));
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

__device__ float sample_trilinear(const float* volume, const SVolGeom& vg,
    float3 position)
{
    const float3 origin = vg.origin();
    const float x = (position.x - origin.x) / vg.vox_x;
    const float y = (position.y - origin.y) / vg.vox_y;
    const float z = (position.z - origin.z) / vg.vox_z;
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
        return volume[static_cast<size_t>(iz) * slice +
            static_cast<size_t>(iy) * vg.Nx + ix];
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
    float3 position, float value)
{
    const float3 origin = vg.origin();
    const float x = (position.x - origin.x) / vg.vox_x;
    const float y = (position.y - origin.y) / vg.vox_y;
    const float z = (position.z - origin.z) / vg.vox_z;
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

__device__ int ray_samples(float3 direction, float t_min, float t_max,
    const SVolGeom& vg, float samples_per_voxel)
{
    const float length = norm3(direction) * (t_max - t_min);
    const float min_voxel = fminf(vg.vox_x, fminf(vg.vox_y, vg.vox_z));
    return max(1, static_cast<int>(ceilf(length * samples_per_voxel / min_voxel)));
}

__global__ void forward_kernel(const float* volume, float* projection,
    const SCylConeProjGeomVec* geometry, int views, int rows, int channels,
    SVolGeom vg, float samples_per_voxel, size_t count, bool accumulate)
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
        float t_min = 0.f, t_max = 0.f;
        float sum = 0.f;
        if (ray_box(source, direction, vg, t_min, t_max)) {
            const int samples = ray_samples(direction, t_min, t_max, vg,
                samples_per_voxel);
            const float dt = (t_max - t_min) / samples;
            const float sample_length = norm3(direction) * dt;
            for (int sample = 0; sample < samples; ++sample) {
                const float ray_t = t_min + (sample + 0.5f) * dt;
                sum += sample_trilinear(volume, vg,
                    add3(source, mul3(direction, ray_t))) * sample_length;
            }
        }
        if (accumulate) projection[index] += sum;
        else projection[index] = sum;
    }
}

__global__ void backproject_kernel(const float* projection, float* volume,
    const SCylConeProjGeomVec* geometry, int views, int rows, int channels,
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
        float t_min = 0.f, t_max = 0.f;
        if (!ray_box(source, direction, vg, t_min, t_max)) continue;
        const int samples = ray_samples(direction, t_min, t_max, vg,
            samples_per_voxel);
        const float dt = (t_max - t_min) / samples;
        const float contribution = ray_value * norm3(direction) * dt;
        for (int sample = 0; sample < samples; ++sample) {
            const float ray_t = t_min + (sample + 0.5f) * dt;
            scatter_trilinear(volume, vg, add3(source, mul3(direction, ray_t)),
                contribution);
        }
    }
}

} // namespace

void launch_forward(const float* volume, float* projection,
    const SCylConeProjGeomVec* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    forward_kernel<<<launch.grid, launch.block, 0, stream>>>(volume, projection,
        geometry, views, rows, channels, volume_geometry,
        config.samples_per_voxel, count, accumulate);
    YK_CUDA_KERNEL_CHECK();
}

void launch_backproject(const float* projection, float* volume,
    const SCylConeProjGeomVec* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, const Config& config,
    cudaStream_t stream, bool accumulate)
{
    if (!accumulate) {
        YK_CUDA_CHECK(cudaMemsetAsync(volume, 0,
            static_cast<size_t>(volume_geometry.Nx) * volume_geometry.Ny *
            volume_geometry.Nz * sizeof(float), stream));
    }
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const auto launch = config.launch.make1D(count);
    backproject_kernel<<<launch.grid, launch.block, 0, stream>>>(projection, volume,
        geometry, views, rows, channels, volume_geometry,
        config.samples_per_voxel, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
