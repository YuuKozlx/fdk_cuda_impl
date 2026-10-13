#include "YkFlatFootprintLaunch.cuh"

#include <cmath>

#include "common/cuda/operators/YkOperatorKernelTypes.cuh"
#include "global/YkMacro.hpp"

namespace YK::Footprint {
namespace {

struct FootprintSample {
    float u_min = 0.f;
    float u_max = 0.f;
    float v_min = 0.f;
    float v_max = 0.f;
    float weight_scale = 0.f;
    bool valid = false;
};

// Both directions call this function.  Keeping detector coordinates, support,
// clipping and ray-length scaling in one routine is what makes the pair a
// transpose of one discrete matrix rather than merely a similar FP/BP pair.
__device__ float affine_dot(const float4& coefficient, float3 point)
{
    return coefficient.x * point.x + coefficient.y * point.y +
        coefficient.z * point.z + coefficient.w;
}

__device__ FootprintSample make_sample(const CudaOp::SFootprintView& view,
    float3 point, const SVolGeom& volume)
{
    FootprintSample result{};
    const float denominator = affine_dot(view.denominator, point);
    if (fabsf(denominator) <= 1e-12f) return result;
    const float reciprocal = __frcp_rn(denominator);
    const float u_numerator = affine_dot(view.u_numerator, point);
    const float v_numerator = affine_dot(view.v_numerator, point);
    const float detector_x = u_numerator * reciprocal;
    const float detector_y = v_numerator * reciprocal;
    if (!isfinite(detector_x) || !isfinite(detector_y)) return result;

    // Linearize the projective map at the voxel center.  The sum of the
    // absolute derivatives gives a separable, axis-aligned detector footprint
    // for the voxel.  Pixel overlap is intentionally not normalized: its sum
    // is the projected footprint area, matching the number of detector rays
    // represented by the voxel rather than treating every voxel as one ray.
    const float reciprocal_squared = reciprocal * reciprocal;
    const float du_dx = (view.u_numerator.x * denominator -
        u_numerator * view.denominator.x) * reciprocal_squared;
    const float du_dy = (view.u_numerator.y * denominator -
        u_numerator * view.denominator.y) * reciprocal_squared;
    const float du_dz = (view.u_numerator.z * denominator -
        u_numerator * view.denominator.z) * reciprocal_squared;
    const float dv_dx = (view.v_numerator.x * denominator -
        v_numerator * view.denominator.x) * reciprocal_squared;
    const float dv_dy = (view.v_numerator.y * denominator -
        v_numerator * view.denominator.y) * reciprocal_squared;
    const float dv_dz = (view.v_numerator.z * denominator -
        v_numerator * view.denominator.z) * reciprocal_squared;
    const float half_u = 0.5f * (fabsf(du_dx) * volume.vox_x +
        fabsf(du_dy) * volume.vox_y + fabsf(du_dz) * volume.vox_z);
    const float half_v = 0.5f * (fabsf(dv_dx) * volume.vox_x +
        fabsf(dv_dy) * volume.vox_y + fabsf(dv_dz) * volume.vox_z);
    if (!(half_u > 0.f) || !(half_v > 0.f) || !isfinite(half_u) ||
        !isfinite(half_v)) return result;

    result.u_min = detector_x - half_u;
    result.u_max = detector_x + half_u;
    result.v_min = detector_y - half_v;
    result.v_max = detector_y + half_v;
    const float bounding_area = 4.f * half_u * half_v;
    const float jacobian_x = du_dy * dv_dz - du_dz * dv_dy;
    const float jacobian_y = du_dz * dv_dx - du_dx * dv_dz;
    const float jacobian_z = du_dx * dv_dy - du_dy * dv_dx;
    const float ray_area_jacobian = sqrtf(jacobian_x * jacobian_x +
        jacobian_y * jacobian_y + jacobian_z * jacobian_z);
    result.weight_scale = volume.vox_x * volume.vox_y * volume.vox_z *
        ray_area_jacobian / bounding_area;
    result.valid = isfinite(result.weight_scale);
    return result;
}

__device__ float pixel_overlap(float footprint_min, float footprint_max,
    int pixel)
{
    const float pixel_min = pixel - 0.5f;
    const float pixel_max = pixel + 0.5f;
    return fmaxf(0.f, fminf(footprint_max, pixel_max) -
        fmaxf(footprint_min, pixel_min));
}

__device__ float3 voxel_center(size_t index, const SVolGeom& volume)
{
    const int x = static_cast<int>(index % volume.Nx);
    const size_t quotient = index / volume.Nx;
    const int y = static_cast<int>(quotient % volume.Ny);
    const int z = static_cast<int>(quotient / volume.Ny);
    const float3 origin = volume.origin();
    return make_float3(origin.x + x * volume.vox_x,
        origin.y + y * volume.vox_y,
        origin.z + z * volume.vox_z);
}

__device__ double atomic_add_double(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* integer_address = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *integer_address;
    unsigned long long assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(integer_address, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__device__ void splat(double* projection, int views, int channels, int rows,
    int view, int u, int v, double value)
{
    if (view < 0 || view >= views || u < 0 || u >= channels ||
        v < 0 || v >= rows) return;
    const size_t index = (static_cast<size_t>(view) * rows + v) * channels + u;
    atomic_add_double(projection + index, value);
}

__device__ float read(const float* projection, int views, int channels,
    int rows, int view, int u, int v)
{
    if (view < 0 || view >= views || u < 0 || u >= channels ||
        v < 0 || v >= rows) return 0.f;
    const size_t index = (static_cast<size_t>(view) * rows + v) * channels + u;
    return projection[index];
}

__global__ void footprint_forward_kernel(const float* volume,
    double* projection, const CudaOp::SFootprintView* views,
    SVolGeom volume_geometry, int view_count, int channels, int rows,
    size_t voxel_count)
{
    const int view = blockIdx.y;
    if (view >= view_count) return;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < voxel_count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float value = volume[index];
        if (value == 0.f) continue;
        const FootprintSample sample = make_sample(views[view],
            voxel_center(index, volume_geometry), volume_geometry);
        if (!sample.valid) continue;
        const float scaled = value * sample.weight_scale;
        const int u_begin = static_cast<int>(ceilf(sample.u_min - 0.5f));
        const int u_end = static_cast<int>(floorf(sample.u_max + 0.5f));
        const int v_begin = static_cast<int>(ceilf(sample.v_min - 0.5f));
        const int v_end = static_cast<int>(floorf(sample.v_max + 0.5f));
        for (int v = v_begin; v <= v_end; ++v) {
            const float wv = pixel_overlap(sample.v_min, sample.v_max, v);
            for (int u = u_begin; u <= u_end; ++u) {
                const float wu = pixel_overlap(sample.u_min, sample.u_max, u);
                splat(projection, view_count, channels, rows, view, u, v,
                    static_cast<double>(scaled) * wu * wv);
            }
        }
    }
}

__global__ void convert_projection_kernel(const double* source,
    float* destination, size_t count, bool accumulate)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float value = static_cast<float>(source[index]);
        if (accumulate) destination[index] += value;
        else destination[index] = value;
    }
}

__global__ void footprint_back_kernel(const float* projection,
    float* volume, const CudaOp::SFootprintView* views,
    SVolGeom volume_geometry, int view_count, int channels, int rows,
    size_t voxel_count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < voxel_count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float3 point = voxel_center(index, volume_geometry);
        float sum = 0.f;
        for (int view = 0; view < view_count; ++view) {
            const FootprintSample sample = make_sample(views[view], point,
                volume_geometry);
            if (!sample.valid) continue;
            const int u_begin = static_cast<int>(ceilf(sample.u_min - 0.5f));
            const int u_end = static_cast<int>(floorf(sample.u_max + 0.5f));
            const int v_begin = static_cast<int>(ceilf(sample.v_min - 0.5f));
            const int v_end = static_cast<int>(floorf(sample.v_max + 0.5f));
            float detector_value = 0.f;
            for (int v = v_begin; v <= v_end; ++v) {
                const float wv = pixel_overlap(sample.v_min, sample.v_max, v);
                for (int u = u_begin; u <= u_end; ++u) {
                    const float wu = pixel_overlap(sample.u_min, sample.u_max,
                        u);
                    detector_value += read(projection, view_count, channels,
                        rows, view, u, v) * wu * wv;
                }
            }
            sum += detector_value * sample.weight_scale;
        }
        volume[index] += sum;
    }
}

int block_count(size_t count, int threads)
{
    const size_t needed = (count + threads - 1) / threads;
    return static_cast<int>(needed < 65535 ? needed : 65535);
}

} // namespace

void forward_launch(const float* volume, float* projection,
    const CudaOp::SFootprintView* views, const SVolGeom& volume_geometry,
    int view_count, int channels, int rows, bool accumulate,
    cudaStream_t stream)
{
    const size_t voxel_count = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    const size_t plane_count = static_cast<size_t>(channels) * rows;
    constexpr int max_batch_views = 16;
    const int allocation_views = view_count < max_batch_views ? view_count :
        max_batch_views;
    double* accumulator = nullptr;
    YK_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&accumulator),
        static_cast<size_t>(allocation_views) * plane_count * sizeof(double),
        stream));
    constexpr int threads = 256;
    for (int view_begin = 0; view_begin < view_count;
         view_begin += max_batch_views) {
        const int remaining_views = view_count - view_begin;
        const int batch_views = remaining_views < max_batch_views ?
            remaining_views : max_batch_views;
        const size_t batch_count = static_cast<size_t>(batch_views) * plane_count;
        YK_CUDA_CHECK(cudaMemsetAsync(accumulator, 0,
            batch_count * sizeof(double), stream));
        const dim3 grid(block_count(voxel_count, threads), batch_views);
        footprint_forward_kernel<<<grid, threads, 0, stream>>>(volume,
            accumulator, views + view_begin, volume_geometry, batch_views,
            channels, rows, voxel_count);
        YK_CUDA_KERNEL_CHECK();
        convert_projection_kernel<<<block_count(batch_count, threads), threads,
            0, stream>>>(accumulator, projection +
                static_cast<size_t>(view_begin) * plane_count, batch_count,
                accumulate);
        YK_CUDA_KERNEL_CHECK();
    }
    YK_CUDA_CHECK(cudaFreeAsync(accumulator, stream));
}

void back_launch(const float* projection, float* volume,
    const CudaOp::SFootprintView* views, const SVolGeom& volume_geometry,
    int view_count, int channels, int rows, bool accumulate,
    cudaStream_t stream)
{
    const size_t voxel_count = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    CudaOp::clearIfOverwrite(volume, voxel_count,
        CudaOp::writeMode(accumulate), stream);
    constexpr int threads = 256;
    footprint_back_kernel<<<block_count(voxel_count, threads), threads, 0,
        stream>>>(projection, volume, views, volume_geometry, view_count,
        channels, rows, voxel_count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::Footprint
