#include "DLTFpBp/kernels/YkDltSiddonLaunch.cuh"

#include <algorithm>

#include "DLTFpBp/kernels/YkDltSiddonTraversal.cuh"
#include "global/YkKernelLaunchPolicy.hpp"
#include "global/YkMacro.hpp"

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

} // namespace YK::DltFpBp
