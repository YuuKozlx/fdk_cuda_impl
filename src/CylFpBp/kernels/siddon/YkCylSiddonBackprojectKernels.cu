#include "CylFpBp/kernels/siddon/YkCylSiddonLaunch.cuh"

#include <cmath>

#include "global/YkMacro.hpp"

namespace YK::CylFpBp::detail {
namespace {

__device__ float3 add3(float3 a, float3 b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 mul3(float3 a, float s)
{ return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float dot3(float3 a, float3 b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 to3(float4 a) { return make_float3(a.x, a.y, a.z); }

// 与 Cyl Siddon FP 读取同一份逐通道基射线。这里只叠加探测器行偏移，
// 没有改变圆柱等角通道对应的射线，也没有改用体素驱动反查。
__device__ float3 prepared_ray(const SKernelView& g,
    const SSiddonChannelRay& channel_ray, int row)
{
    return add3(to3(channel_ray.ray_at_principal_row),
        mul3(to3(g.detector_v), row - g.principal_v));
}

__device__ bool clip_box(float3 source, float3 ray, const SVolGeom& vg,
    float& tmin, float& tmax)
{
    const float3 origin = vg.origin();
    const float lo[3] = {origin.x - .5f * vg.vox_x, origin.y - .5f * vg.vox_y,
        origin.z - .5f * vg.vox_z};
    const float hi[3] = {origin.x + (vg.Nx - .5f) * vg.vox_x,
        origin.y + (vg.Ny - .5f) * vg.vox_y,
        origin.z + (vg.Nz - .5f) * vg.vox_z};
    const float s[3] = {source.x, source.y, source.z};
    const float d[3] = {ray.x, ray.y, ray.z};
    tmin = 0.f;
    tmax = 1.f;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1e-12f) {
            if (s[axis] < lo[axis] || s[axis] > hi[axis]) return false;
            continue;
        }
        float a = (lo[axis] - s[axis]) / d[axis];
        float b = (hi[axis] - s[axis]) / d[axis];
        if (a > b) { const float temp = a; a = b; b = temp; }
        tmin = fmaxf(tmin, a);
        tmax = fminf(tmax, b);
    }
    return tmax > tmin;
}

__global__ void cyl_siddon_backproject_kernel(cudaTextureObject_t projection,
    float* volume, const SKernelView* geometry,
    const SSiddonChannelRay* channel_rays, int views, int rows, int channels,
    SVolGeom vg)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(views) * rows * channels;
    if (index >= count) return;
    const int channel = static_cast<int>(index % channels);
    const size_t view_row = index / channels;
    const int row = static_cast<int>(view_row % rows);
    const int view = static_cast<int>(view_row / rows);
    const float ray_value = tex3D<float>(projection, channel + .5f,
        row + .5f, view + .5f);
    if (ray_value == 0.f) return;

    const float3 source = to3(geometry[view].source);
    const float3 ray = prepared_ray(geometry[view],
        channel_rays[static_cast<size_t>(view) * channels + channel], row);
    float tmin = 0.f, tmax = 0.f;
    if (!clip_box(source, ray, vg, tmin, tmax)) return;
    const float3 origin = vg.origin();
    const float3 entry = add3(source, mul3(ray, tmin));
    int ix = max(0, min(vg.Nx - 1, static_cast<int>(floorf(
        (entry.x - origin.x) / vg.vox_x + .5f))));
    int iy = max(0, min(vg.Ny - 1, static_cast<int>(floorf(
        (entry.y - origin.y) / vg.vox_y + .5f))));
    int iz = max(0, min(vg.Nz - 1, static_cast<int>(floorf(
        (entry.z - origin.z) / vg.vox_z + .5f))));
    const int step_x = ray.x >= 0.f ? 1 : -1;
    const int step_y = ray.y >= 0.f ? 1 : -1;
    const int step_z = ray.z >= 0.f ? 1 : -1;
    const bool valid_x = fabsf(ray.x) > 1e-12f;
    const bool valid_y = fabsf(ray.y) > 1e-12f;
    const bool valid_z = fabsf(ray.z) > 1e-12f;
    const float bx = origin.x + (ix + (step_x > 0 ? .5f : -.5f)) * vg.vox_x;
    const float by = origin.y + (iy + (step_y > 0 ? .5f : -.5f)) * vg.vox_y;
    const float bz = origin.z + (iz + (step_z > 0 ? .5f : -.5f)) * vg.vox_z;
    constexpr float infinity = 1e30f;
    float tx = valid_x ? (bx - source.x) / ray.x : infinity;
    float ty = valid_y ? (by - source.y) / ray.y : infinity;
    float tz = valid_z ? (bz - source.z) / ray.z : infinity;
    const float dtx = valid_x ? fabsf(vg.vox_x / ray.x) : infinity;
    const float dty = valid_y ? fabsf(vg.vox_y / ray.y) : infinity;
    const float dtz = valid_z ? fabsf(vg.vox_z / ray.z) : infinity;
    const float ray_length = sqrtf(dot3(ray, ray));
    const size_t plane = static_cast<size_t>(vg.Nx) * vg.Ny;
    float current = tmin;
    while (current < tmax && ix >= 0 && ix < vg.Nx && iy >= 0 && iy < vg.Ny &&
        iz >= 0 && iz < vg.Nz) {
        const float next = fminf(fminf(tx, ty), fminf(tz, tmax));
        const size_t voxel = static_cast<size_t>(iz) * plane +
            static_cast<size_t>(iy) * vg.Nx + ix;
        atomicAdd(volume + voxel, ray_value * (next - current) * ray_length);
        current = next;
        if (tx <= ty && tx <= tz) { ix += step_x; tx += dtx; }
        else if (ty <= tz) { iy += step_y; ty += dty; }
        else { iz += step_z; tz += dtz; }
    }
}

} // namespace

void launch_cyl_siddon_backproject(cudaTextureObject_t projection, float* volume,
    const SKernelView* geometry, const SSiddonChannelRay* channel_rays,
    int views, int rows, int channels, const SVolGeom& volume_geometry,
    bool accumulate, cudaStream_t stream)
{
    if (!accumulate) {
        const size_t bytes = static_cast<size_t>(volume_geometry.Nx) *
            volume_geometry.Ny * volume_geometry.Nz * sizeof(float);
        YK_CUDA_CHECK(cudaMemsetAsync(volume, 0, bytes, stream));
    }
    const size_t count = static_cast<size_t>(views) * rows * channels;
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    cyl_siddon_backproject_kernel<<<grid, block, 0, stream>>>(projection, volume,
        geometry, channel_rays, views, rows, channels, volume_geometry);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
