#include "DLTFpBp/kernels/YkDltSiddonLaunch.cuh"

#include <algorithm>

#include "DLTFpBp/kernels/YkDltSiddonTraversal.cuh"
#include "global/YkKernelLaunchPolicy.hpp"
#include "global/YkMacro.hpp"
#include <math_constants.h>

namespace YK::DltFpBp {
namespace detail {

struct ForwardAccumulator {
    const float* volume = nullptr;
    float value = 0.f;

    __device__ __forceinline__ void add(size_t index, float length)
    { value = fmaf(volume[index], length, value); }
};

struct BackAccumulator {
    float* volume = nullptr;
    float projection_value = 0.f;

    __device__ __forceinline__ void add(size_t index, float length)
    { atomicAdd(volume + index, projection_value * length); }
};

__global__ void dltSiddonForwardKernel(const float* volume, float* projection,
    const SDltRayGeometry* geometry, SVolGeom volume_geometry,
    int channels, int rows, int views, bool accumulate)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    const int view = blockIdx.z;
    if (u >= channels || v >= rows || view >= views) return;
    ForwardAccumulator accumulator{volume, 0.f};
    traverseDltSiddonRay(geometry[view], u, v, volume_geometry, accumulator);
    const size_t index = (static_cast<size_t>(view) * rows + v) * channels + u;
    if (accumulate) projection[index] += accumulator.value;
    else projection[index] = accumulator.value;
}

__global__ void dltSiddonBackKernel(const float* projection, float* volume,
    const SDltRayGeometry* geometry, SVolGeom volume_geometry,
    int channels, int rows, int views)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    const int view = blockIdx.z;
    if (u >= channels || v >= rows || view >= views) return;
    const size_t index = (static_cast<size_t>(view) * rows + v) * channels + u;
    BackAccumulator accumulator{volume, projection[index]};
    traverseDltSiddonRay(geometry[view], u, v, volume_geometry, accumulator);
}

template <bool Bilinear>
__device__ __forceinline__ float sampleVoxelBackProjection(
    const float* projection, int channels, int rows, int view,
    float u, float v)
{
    if constexpr (Bilinear) {
        const int u0 = static_cast<int>(floorf(u));
        const int v0 = static_cast<int>(floorf(v));
        const int u1 = min(u0 + 1, channels - 1);
        const int v1 = min(v0 + 1, rows - 1);
        const float du = u - static_cast<float>(u0);
        const float dv = v - static_cast<float>(v0);
        const size_t base = static_cast<size_t>(view) * rows * channels;
        const float p00 = __ldg(projection + base + static_cast<size_t>(v0) * channels + u0);
        const float p10 = __ldg(projection + base + static_cast<size_t>(v0) * channels + u1);
        const float p01 = __ldg(projection + base + static_cast<size_t>(v1) * channels + u0);
        const float p11 = __ldg(projection + base + static_cast<size_t>(v1) * channels + u1);
        const float top = fmaf(du, p10 - p00, p00);
        const float bottom = fmaf(du, p11 - p01, p01);
        return fmaf(dv, bottom - top, top);
    }
    const int iu = static_cast<int>(floorf(u + 0.5f));
    const int iv = static_cast<int>(floorf(v + 0.5f));
    const size_t index = (static_cast<size_t>(view) * rows + iv) * channels + iu;
    return __ldg(projection + index);
}

__device__ __forceinline__ bool clipVoxelRay(
    const float3& source, const float3& ray, const SVolGeom& volume,
    float& near_parameter, float& far_parameter)
{
    const float3 first_center = volume.origin();
    const float3 lower = make_float3(first_center.x - 0.5f * volume.vox_x,
        first_center.y - 0.5f * volume.vox_y,
        first_center.z - 0.5f * volume.vox_z);
    const float3 upper = make_float3(lower.x + volume.Nx * volume.vox_x,
        lower.y + volume.Ny * volume.vox_y,
        lower.z + volume.Nz * volume.vox_z);
    near_parameter = 0.f;
    far_parameter = CUDART_INF_F;
    const auto clip_axis = [&](float origin, float direction,
                               float minimum, float maximum) {
        if (fabsf(direction) <= 1e-12f)
            return origin >= minimum && origin <= maximum;
        float first = (minimum - origin) / direction;
        float second = (maximum - origin) / direction;
        if (first > second) {
            const float swap = first;
            first = second;
            second = swap;
        }
        near_parameter = fmaxf(near_parameter, first);
        far_parameter = fminf(far_parameter, second);
        return near_parameter < far_parameter;
    };
    return clip_axis(source.x, ray.x, lower.x, upper.x) &&
        clip_axis(source.y, ray.y, lower.y, upper.y) &&
        clip_axis(source.z, ray.z, lower.z, upper.z) &&
        isfinite(far_parameter) && near_parameter < far_parameter;
}

template <bool Bilinear>
__global__ void dltVoxelBackKernel(const float* projection, float* volume,
    const SDltVoxelBackGeometry* geometry, SVolGeom volume_geometry,
    int channels, int rows, int views, bool accumulate)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= volume_geometry.Nx || iy >= volume_geometry.Ny ||
        iz >= volume_geometry.Nz) return;

    const float3 origin = volume_geometry.origin();
    const float3 point = make_float3(
        origin.x + ix * volume_geometry.vox_x,
        origin.y + iy * volume_geometry.vox_y,
        origin.z + iz * volume_geometry.vox_z);
    float result = 0.f;

    for (int view = 0; view < views; ++view) {
        const SDltVoxelBackGeometry& g = geometry[view];
        const float3 delta = make_float3(point.x - g.source.x,
            point.y - g.source.y, point.z - g.source.z);
        const float a = g.inverseRow0.x * delta.x +
            g.inverseRow0.y * delta.y + g.inverseRow0.z * delta.z;
        if (!(a > 1e-8f) || !isfinite(a)) continue;
        const float b = g.inverseRow1.x * delta.x +
            g.inverseRow1.y * delta.y + g.inverseRow1.z * delta.z;
        const float c = g.inverseRow2.x * delta.x +
            g.inverseRow2.y * delta.y + g.inverseRow2.z * delta.z;
        const float u = b / a;
        const float v = c / a;
        if (!isfinite(u) || !isfinite(v)) continue;

        if constexpr (Bilinear) {
            if (u < 0.f || u > static_cast<float>(channels - 1) ||
                v < 0.f || v > static_cast<float>(rows - 1)) continue;
        }
        else {
            const int iu = static_cast<int>(floorf(u + 0.5f));
            const int iv = static_cast<int>(floorf(v + 0.5f));
            if (iu < 0 || iu >= channels || iv < 0 || iv >= rows) continue;
        }

        const float3 ray = make_float3(
            g.ray00.x + u * g.rayU.x + v * g.rayV.x,
            g.ray00.y + u * g.rayU.y + v * g.rayV.y,
            g.ray00.z + u * g.rayU.z + v * g.rayV.z);
        const float ray_length = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
        if (!(ray_length > 1e-12f) || !isfinite(ray_length)) continue;

        float near_parameter = 0.f, far_parameter = 0.f;
        if (!clipVoxelRay(make_float3(g.source.x, g.source.y, g.source.z),
                ray, volume_geometry, near_parameter, far_parameter)) continue;
        const float chord = (far_parameter - near_parameter) * ray_length;
        if (!(chord > 0.f) || !isfinite(chord)) continue;
        result = fmaf(sampleVoxelBackProjection<Bilinear>(projection, channels,
            rows, view, u, v), chord, result);
    }

    const size_t index = (static_cast<size_t>(iz) * volume_geometry.Ny + iy) *
        volume_geometry.Nx + ix;
    if (accumulate) volume[index] += result;
    else volume[index] = result;
}

} // namespace detail

void dltSiddonForwardLaunch(const float* volume, float* projection,
    const SDltRayGeometry* geometry, const SVolGeom& volume_geometry,
    int channels, int rows, int views, bool accumulate, cudaStream_t stream)
{
    if (!volume || !projection || !geometry || channels <= 0 || rows <= 0 ||
        views <= 0) return;
    SKernelLaunchPolicy policy;
    const int maximum_chunk = policy.maxAngleChunk();
    const size_t view_elements = static_cast<size_t>(channels) * rows;
    const dim3 block(16, 16, 1);
    const dim3 xy((channels + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y, 1);
    for (int base = 0; base < views;) {
        const int count = std::min(maximum_chunk, views - base);
        const dim3 grid(xy.x, xy.y, count);
        detail::dltSiddonForwardKernel<<<grid, block, 0, stream>>>(volume,
            projection + static_cast<size_t>(base) * view_elements,
            geometry + base, volume_geometry, channels, rows, count, accumulate);
        base += count;
    }
    YK_CUDA_KERNEL_CHECK();
}

void dltSiddonBackLaunch(const float* projection, float* volume,
    const SDltRayGeometry* geometry, const SVolGeom& volume_geometry,
    int channels, int rows, int views, cudaStream_t stream)
{
    if (!volume || !projection || !geometry || channels <= 0 || rows <= 0 ||
        views <= 0) return;
    SKernelLaunchPolicy policy;
    const int maximum_chunk = policy.maxAngleChunk();
    const size_t view_elements = static_cast<size_t>(channels) * rows;
    const dim3 block(16, 16, 1);
    const dim3 xy((channels + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y, 1);
    for (int base = 0; base < views;) {
        const int count = std::min(maximum_chunk, views - base);
        const dim3 grid(xy.x, xy.y, count);
        detail::dltSiddonBackKernel<<<grid, block, 0, stream>>>(
            projection + static_cast<size_t>(base) * view_elements, volume,
            geometry + base, volume_geometry, channels, rows, count);
        base += count;
    }
    YK_CUDA_KERNEL_CHECK();
}

void dltVoxelBackLaunch(const float* projection, float* volume,
    const SDltVoxelBackGeometry* geometry, const SVolGeom& volume_geometry,
    int channels, int rows, int views, bool bilinear, bool accumulate,
    cudaStream_t stream)
{
    if (!projection || !volume || !geometry || channels <= 0 || rows <= 0 ||
        views <= 0 || volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
        volume_geometry.Nz <= 0) return;
    const size_t volume_elements = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    if (!accumulate)
        YK_CUDA_CHECK(cudaMemsetAsync(volume, 0,
            volume_elements * sizeof(float), stream));
    const dim3 block(8, 8, 4);
    const dim3 grid(
        (volume_geometry.Nx + block.x - 1) / block.x,
        (volume_geometry.Ny + block.y - 1) / block.y,
        (volume_geometry.Nz + block.z - 1) / block.z);
    if (bilinear)
        detail::dltVoxelBackKernel<true><<<grid, block, 0, stream>>>(
            projection, volume, geometry, volume_geometry, channels, rows,
            views, accumulate);
    else
        detail::dltVoxelBackKernel<false><<<grid, block, 0, stream>>>(
            projection, volume, geometry, volume_geometry, channels, rows,
            views, accumulate);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::DltFpBp
