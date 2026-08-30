#include "CylFpBp/kernels/siddon/YkCylSiddonLaunch.cuh"

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
__device__ float3 to3(float4 a) { return make_float3(a.x, a.y, a.z); }

// 将源点到体素中心的射线映射到圆柱探测器连续坐标。该映射只负责确定
// 投影样本；Siddon 权重由同一条射线与当前体素盒的实际交长给出。
__device__ bool map_voxel_to_detector(const SVoxelDrivenView& view,
    float3 voxel, float& channel, float& row, float3& detector,
    float* intersection_t = nullptr)
{
    const float3 source = to3(view.source);
    const float3 axis = to3(view.axis_unit);
    const float3 direction = sub3(voxel, source);
    const float3 source_perp = to3(view.source_perp);
    const float3 direction_perp = sub3(direction,
        mul3(axis, dot3(direction, axis)));
    const float a = dot3(direction_perp, direction_perp);
    if (a < 1e-16f) return false;
    const float b = 2.f * dot3(source_perp, direction_perp);
    const float c = view.source_perp_norm_sq - view.radius_mm * view.radius_mm;
    const float discriminant = b * b - 4.f * a * c;
    if (discriminant < 0.f) return false;
    const float root = view.source_on_axis ? 0.f :
        sqrtf(fmaxf(discriminant, 0.f));
    // 源在圆柱轴上时退化为 t=R/|d_perp|，省去判别式和二次方程。
    const float t = view.source_on_axis
        ? view.radius_mm * rsqrtf(a)
        : fmaxf((-b - root) / (2.f * a), (-b + root) / (2.f * a));
    if (!(t > 0.f)) return false;
    if (intersection_t) *intersection_t = t;
    detector = add3(source, mul3(direction, t));

    const float3 relative = sub3(detector, to3(view.cylinder_center));
    const float axial = dot3(relative, axis);
    const float3 radial = sub3(relative, mul3(axis, axial));
    const float angle = atan2f(dot3(radial, to3(view.tangent_unit)),
        dot3(radial, to3(view.radial_unit)));
    channel = view.principal_u + angle * view.inv_channel_angle_step_rad;
    row = view.principal_v + axial * view.inv_row_step_mm;
    return true;
}

__device__ float voxel_segment_length(float3 source, float3 ray,
    int x, int y, int z, const SVolGeom& vg)
{
    const float3 origin = vg.origin();
    const float center[3] = {origin.x + x * vg.vox_x,
        origin.y + y * vg.vox_y, origin.z + z * vg.vox_z};
    const float half[3] = {.5f * vg.vox_x, .5f * vg.vox_y, .5f * vg.vox_z};
    const float s[3] = {source.x, source.y, source.z};
    const float d[3] = {ray.x, ray.y, ray.z};
    float tmin = 0.f, tmax = 1.f;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1e-12f) {
            if (s[axis] < center[axis] - half[axis] ||
                s[axis] > center[axis] + half[axis]) return 0.f;
            continue;
        }
        float a = (center[axis] - half[axis] - s[axis]) / d[axis];
        float b = (center[axis] + half[axis] - s[axis]) / d[axis];
        if (a > b) { const float tmp = a; a = b; b = tmp; }
        tmin = fmaxf(tmin, a);
        tmax = fminf(tmax, b);
    }
    return tmax > tmin ? (tmax - tmin) * sqrtf(dot3(ray, ray)) : 0.f;
}

__device__ bool voxel_xy_interval(float3 source, float3 ray, int x, int y,
    const SVolGeom& vg, float& tmin, float& tmax)
{
    const float3 origin = vg.origin();
    const float lo[2] = {origin.x + (x - .5f) * vg.vox_x,
        origin.y + (y - .5f) * vg.vox_y};
    const float hi[2] = {origin.x + (x + .5f) * vg.vox_x,
        origin.y + (y + .5f) * vg.vox_y};
    const float s[2] = {source.x, source.y};
    const float d[2] = {ray.x, ray.y};
    tmin = 0.f; tmax = 1.f;
    for (int axis = 0; axis < 2; ++axis) {
        if (fabsf(d[axis]) < 1e-12f) {
            if (s[axis] < lo[axis] || s[axis] > hi[axis]) return false;
            continue;
        }
        float a = (lo[axis] - s[axis]) / d[axis];
        float b = (hi[axis] - s[axis]) / d[axis];
        if (a > b) { const float tmp = a; a = b; b = tmp; }
        tmin = fmaxf(tmin, a); tmax = fminf(tmax, b);
    }
    return tmax > tmin;
}

__device__ float voxel_segment_from_xy(float3 source, float3 ray, int z,
    const SVolGeom& vg, float xy_min, float xy_max)
{
    const float center = vg.origin().z + z * vg.vox_z;
    float z_min = 0.f, z_max = 1.f;
    if (fabsf(ray.z) < 1e-12f) {
        if (source.z < center - .5f * vg.vox_z ||
            source.z > center + .5f * vg.vox_z) return 0.f;
    }
    else {
        z_min = (center - .5f * vg.vox_z - source.z) / ray.z;
        z_max = (center + .5f * vg.vox_z - source.z) / ray.z;
        if (z_min > z_max) { const float tmp = z_min; z_min = z_max; z_max = tmp; }
    }
    const float begin = fmaxf(xy_min, z_min);
    const float end = fminf(xy_max, z_max);
    return end > begin ? (end - begin) * sqrtf(dot3(ray, ray)) : 0.f;
}

__device__ float3 detector_pixel(const SVoxelDrivenView& view, int channel,
    int row)
{
    const float angle = (channel - view.principal_u) /
        view.inv_channel_angle_step_rad;
    float sine = 0.f, cosine = 0.f;
    sincosf(angle, &sine, &cosine);
    const float axial = (row - view.principal_v) / view.inv_row_step_mm;
    return add3(to3(view.cylinder_center), add3(
        mul3(to3(view.radial_unit), view.radius_mm * cosine),
        add3(mul3(to3(view.tangent_unit), view.radius_mm * sine),
            mul3(to3(view.axis_unit), axial))));
}

template<int ZSize>
__global__ void siddon_voxel_v2_kernel(cudaTextureObject_t projection,
    const SVoxelDrivenView* __restrict__ geometry, float* __restrict__ volume,
    int views, SVolGeom vg)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int first_z = blockIdx.z * ZSize;
    if (x >= vg.Nx || y >= vg.Ny || first_z >= vg.Nz) return;
    const float3 origin = vg.origin();
    const float wx = origin.x + x * vg.vox_x;
    const float wy = origin.y + y * vg.vox_y;
    float sums[ZSize]{};
    for (int index = 0; index < views; ++index) {
        const SVoxelDrivenView& view = geometry[index];
        const float3 axis = to3(view.axis_unit);
        if (fabsf(axis.x) < 1e-6f && fabsf(axis.y) < 1e-6f) {
            const float3 first_voxel = make_float3(wx, wy,
                origin.z + first_z * vg.vox_z);
            float channel = 0.f, first_row = 0.f, intersection_t = 0.f;
            float3 first_detector{};
            if (!map_voxel_to_detector(view, first_voxel, channel, first_row,
                    first_detector, &intersection_t)) continue;
            const float first_axial = (first_row - view.principal_v) /
                view.inv_row_step_mm;
            const float3 channel_point = sub3(first_detector,
                mul3(axis, first_axial));
            const float3 source = to3(view.source);
            const float3 base_ray = sub3(channel_point, source);
            float xy_min = 0.f, xy_max = 0.f;
            if (!voxel_xy_interval(source, base_ray, x, y, vg,
                    xy_min, xy_max)) continue;
            const float row_increment = intersection_t * vg.vox_z * axis.z *
                view.inv_row_step_mm;
#pragma unroll
            for (int local_z = 0; local_z < ZSize; ++local_z) {
                const int z = first_z + local_z;
                if (z >= vg.Nz) continue;
                const float row = first_row + local_z * row_increment;
                const float axial = (row - view.principal_v) /
                    view.inv_row_step_mm;
                const float3 ray = sub3(add3(channel_point,
                    mul3(axis, axial)), source);
                sums[local_z] += tex3D<float>(projection, channel + .5f,
                    row + .5f, index + .5f) * voxel_segment_from_xy(source,
                    ray, z, vg, xy_min, xy_max);
            }
            continue;
        }
#pragma unroll
        for (int local_z = 0; local_z < ZSize; ++local_z) {
            const int z = first_z + local_z;
            if (z >= vg.Nz) continue;
            const float3 voxel = make_float3(wx, wy,
                origin.z + z * vg.vox_z);
            float channel = 0.f, row = 0.f;
            float3 detector{};
            if (!map_voxel_to_detector(view, voxel, channel, row, detector))
                continue;
            const float3 source = to3(view.source);
            const float3 ray = sub3(detector, source);
            sums[local_z] += tex3D<float>(projection, channel + .5f,
                row + .5f, index + .5f) * voxel_segment_length(source,
                ray, x, y, z, vg);
        }
    }
    const size_t plane = static_cast<size_t>(vg.Nx) * vg.Ny;
#pragma unroll
    for (int local_z = 0; local_z < ZSize; ++local_z) {
        const int z = first_z + local_z;
        if (z < vg.Nz)
            volume[static_cast<size_t>(z) * plane +
                static_cast<size_t>(y) * vg.Nx + x] += sums[local_z];
    }
}

template<int ZSize>
__global__ void siddon_voxel_v3_kernel(cudaTextureObject_t projection,
    const SVoxelDrivenView* __restrict__ geometry, float* __restrict__ volume,
    const SCylVoxelChannelRay* __restrict__ channel_rays, int views,
    int channels, SVolGeom vg)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int first_z = blockIdx.z * ZSize;
    if (x >= vg.Nx || y >= vg.Ny || first_z >= vg.Nz) return;
    const float3 origin = vg.origin();
    const float wx = origin.x + x * vg.vox_x;
    const float wy = origin.y + y * vg.vox_y;
    float sums[ZSize]{};
    for (int index = 0; index < views; ++index) {
        const SVoxelDrivenView& view = geometry[index];
        const float3 axis = to3(view.axis_unit);
        // 标准 CT 圆柱轴与体积 Z 轴对齐时，固定 (x,y,view) 的通道和
        // 横向求交参数不随 z 改变。ZSize 个体素共用一次圆柱求交和
        // 一次通道 sin/cos；一般姿态继续走下方完整映射。
        if (fabsf(axis.x) < 1e-6f && fabsf(axis.y) < 1e-6f) {
            const float3 first_voxel = make_float3(wx, wy,
                origin.z + first_z * vg.vox_z);
            float channel = 0.f, first_row = 0.f, intersection_t = 0.f;
            float3 ignored_detector{};
            if (!map_voxel_to_detector(view, first_voxel, channel, first_row,
                    ignored_detector, &intersection_t)) continue;
            const int iu = __float2int_rn(channel);
            const float3 source = to3(view.source);
            const float3 channel_point = to3(channel_rays[
                static_cast<size_t>(index) * channels + iu].detector_at_principal_row);
            const float3 base_ray = sub3(channel_point, source);
            float xy_min = 0.f, xy_max = 0.f;
            if (!voxel_xy_interval(source, base_ray, x, y, vg, xy_min, xy_max))
                continue;
            const float row_increment = intersection_t * vg.vox_z * axis.z *
                view.inv_row_step_mm;
#pragma unroll
            for (int local_z = 0; local_z < ZSize; ++local_z) {
                const int z = first_z + local_z;
                if (z >= vg.Nz) continue;
                const int iv = __float2int_rn(first_row +
                    local_z * row_increment);
                const float axial = (iv - view.principal_v) /
                    view.inv_row_step_mm;
                const float3 ray = sub3(add3(channel_point,
                    mul3(axis, axial)), source);
                sums[local_z] += tex3D<float>(projection, iu + .5f,
                    iv + .5f, index + .5f) * voxel_segment_from_xy(source,
                    ray, z, vg, xy_min, xy_max);
            }
            continue;
        }
#pragma unroll
        for (int local_z = 0; local_z < ZSize; ++local_z) {
            const int z = first_z + local_z;
            if (z >= vg.Nz) continue;
            const float3 voxel = make_float3(wx, wy, origin.z + z * vg.vox_z);
            float channel = 0.f, row = 0.f;
            float3 detector{};
            if (!map_voxel_to_detector(view, voxel, channel, row,
                    detector)) continue;
            const int iu = __float2int_rn(channel);
            const int iv = __float2int_rn(row);
            const float3 source = to3(view.source);
            // V3 与 Flat Siddon V3 一致：取最近探测器像素，并用该像素
            // 中心对应射线计算体素盒交长，不能沿用连续坐标射线。
            const float3 ray = sub3(detector_pixel(view, iu, iv),
                source);
            sums[local_z] += tex3D<float>(projection, iu + .5f, iv + .5f,
                index + .5f) * voxel_segment_length(source, ray, x, y, z, vg);
        }
    }
    const size_t plane = static_cast<size_t>(vg.Nx) * vg.Ny;
#pragma unroll
    for (int local_z = 0; local_z < ZSize; ++local_z) {
        const int z = first_z + local_z;
        if (z < vg.Nz)
            volume[static_cast<size_t>(z) * plane +
                static_cast<size_t>(y) * vg.Nx + x] += sums[local_z];
    }
}

void clear_volume(float* volume, const SVolGeom& vg, bool accumulate,
    cudaStream_t stream)
{
    CudaOp::clearIfOverwrite(volume,
        static_cast<size_t>(vg.Nx) * vg.Ny * vg.Nz,
        CudaOp::writeMode(accumulate), stream);
}

} // namespace

void launch_cyl_siddon_backproject_v2(cudaTextureObject_t projection,
    float* volume, const SVoxelDrivenView* geometry, int views,
    const SVolGeom& vg, bool accumulate, cudaStream_t stream)
{
    clear_volume(volume, vg, accumulate, stream);
    constexpr int z_size = 4;
    const dim3 block(16, 16, 1);
    const dim3 grid((vg.Nx + block.x - 1) / block.x,
        (vg.Ny + block.y - 1) / block.y, (vg.Nz + z_size - 1) / z_size);
    siddon_voxel_v2_kernel<z_size><<<grid, block, 0, stream>>>(projection, geometry,
        volume, views, vg);
    YK_CUDA_KERNEL_CHECK();
}

void launch_cyl_siddon_backproject_v3(cudaTextureObject_t projection,
    float* volume, const SVoxelDrivenView* geometry,
    const SCylVoxelChannelRay* channel_rays, int views, int channels,
    const SVolGeom& vg,
    bool accumulate, cudaStream_t stream)
{
    clear_volume(volume, vg, accumulate, stream);
    constexpr int z_size = 4;
    const dim3 block(16, 16, 1);
    const dim3 grid((vg.Nx + block.x - 1) / block.x,
        (vg.Ny + block.y - 1) / block.y, (vg.Nz + z_size - 1) / z_size);
    siddon_voxel_v3_kernel<z_size><<<grid, block, 0, stream>>>(projection,
        geometry, volume, channel_rays, views, channels, vg);
    YK_CUDA_KERNEL_CHECK();
}

template<int ZSize>
__global__ void siddon_voxel_v3_standard_kernel(cudaTextureObject_t projection,
    const SVoxelDrivenView* __restrict__ geometry, float* __restrict__ volume,
    const SCylVoxelChannelRay* __restrict__ channel_rays, int views,
    int channels, SVolGeom vg)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int first_z = blockIdx.z * ZSize;
    if (x >= vg.Nx || y >= vg.Ny || first_z >= vg.Nz) return;
    const float3 origin = vg.origin();
    const float wx = origin.x + x * vg.vox_x;
    const float wy = origin.y + y * vg.vox_y;
    float sums[ZSize]{};
    for (int index = 0; index < views; ++index) {
        const SVoxelDrivenView& view = geometry[index];
        const float3 axis = to3(view.axis_unit);
        const float3 first_voxel = make_float3(wx, wy,
            origin.z + first_z * vg.vox_z);
        float channel = 0.f, first_row = 0.f, intersection_t = 0.f;
        float3 ignored{};
        if (!map_voxel_to_detector(view, first_voxel, channel, first_row,
                ignored, &intersection_t)) continue;
        const int iu = __float2int_rn(channel);
        const float3 source = to3(view.source);
        const float3 channel_point = to3(channel_rays[
            static_cast<size_t>(index) * channels + iu].detector_at_principal_row);
        const float3 base_ray = sub3(channel_point, source);
        float xy_min = 0.f, xy_max = 0.f;
        if (!voxel_xy_interval(source, base_ray, x, y, vg, xy_min, xy_max)) continue;
        const float row_increment = intersection_t * vg.vox_z * axis.z *
            view.inv_row_step_mm;
#pragma unroll
        for (int local_z = 0; local_z < ZSize; ++local_z) {
            const int z = first_z + local_z;
            if (z >= vg.Nz) continue;
            const int iv = __float2int_rn(first_row + local_z * row_increment);
            const float axial = (iv - view.principal_v) / view.inv_row_step_mm;
            const float3 ray = sub3(add3(channel_point, mul3(axis, axial)), source);
            sums[local_z] += tex3D<float>(projection, iu + .5f, iv + .5f,
                index + .5f) * voxel_segment_from_xy(source, ray, z, vg,
                xy_min, xy_max);
        }
    }
    const size_t plane = static_cast<size_t>(vg.Nx) * vg.Ny;
#pragma unroll
    for (int local_z = 0; local_z < ZSize; ++local_z) {
        const int z = first_z + local_z;
        if (z < vg.Nz) volume[static_cast<size_t>(z) * plane +
            static_cast<size_t>(y) * vg.Nx + x] += sums[local_z];
    }
}

void launch_cyl_siddon_backproject_v3_standard(cudaTextureObject_t projection,
    float* volume, const SVoxelDrivenView* geometry,
    const SCylVoxelChannelRay* channel_rays, int views, int channels,
    const SVolGeom& vg, bool accumulate, cudaStream_t stream)
{
    clear_volume(volume, vg, accumulate, stream);
    constexpr int z_size = 4;
    const dim3 block(16, 16, 1);
    const dim3 grid((vg.Nx + block.x - 1) / block.x,
        (vg.Ny + block.y - 1) / block.y, (vg.Nz + z_size - 1) / z_size);
    siddon_voxel_v3_standard_kernel<z_size><<<grid, block, 0, stream>>>(
        projection, geometry, volume, channel_rays, views, channels, vg);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
