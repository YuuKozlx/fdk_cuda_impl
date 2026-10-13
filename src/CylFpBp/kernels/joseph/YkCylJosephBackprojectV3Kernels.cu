#include "CylFpBp/kernels/joseph/YkCylJosephBackprojectV3Launch.cuh"

#include <algorithm>
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

// 求源点到体素射线与无限圆柱面的探测器侧交点。圆柱轴可能不经过源点，
// 因而必须解一般二次方程。若源点位于圆柱外，射线可能有两个正根；较大根
// 对应远侧探测器表面，与 detector_principal 的定义一致。
__device__ bool intersect_detector_cylinder(const SVoxelDrivenView& view,
    float3 voxel, float3& intersection)
{
    const float3 source = to3(view.source);
    const float3 axis = to3(view.axis_unit);
    const float3 cylinder_to_source = sub3(source, to3(view.cylinder_center));
    const float3 direction = sub3(voxel, source);
    const float3 source_perp = sub3(cylinder_to_source,
        mul3(axis, dot3(cylinder_to_source, axis)));
    const float3 direction_perp = sub3(direction,
        mul3(axis, dot3(direction, axis)));
    const float a = dot3(direction_perp, direction_perp);
    if (a < 1e-16f) return false;
    const float b = 2.f * dot3(source_perp, direction_perp);
    const float c = dot3(source_perp, source_perp) -
        view.radius_mm * view.radius_mm;
    const float discriminant = b * b - 4.f * a * c;
    if (discriminant < 0.f) return false;
    const float root = sqrtf(fmaxf(discriminant, 0.f));
    const float inverse_2a = 0.5f / a;
    const float t0 = (-b - root) * inverse_2a;
    const float t1 = (-b + root) * inverse_2a;
    const float ray_t = fmaxf(t0, t1);
    if (!(ray_t > 0.f)) return false;
    intersection = add3(source, mul3(direction, ray_t));
    return true;
}

__device__ float physical_main_axis_length(float3 direction,
    const SVolGeom& volume_geometry)
{
    const float dx = direction.x * volume_geometry.tmp_rcp_vox_x;
    const float dy = direction.y * volume_geometry.tmp_rcp_vox_y;
    const float dz = direction.z * volume_geometry.tmp_rcp_vox_z;
    const float ax = fabsf(dx), ay = fabsf(dy), az = fabsf(dz);
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, slope1 = 0.f, slope2 = 0.f;
    if (ax >= ay && ax >= az && ax > 1e-12f) {
        v0 = volume_geometry.vox_x;
        v1 = volume_geometry.vox_y;
        v2 = volume_geometry.vox_z;
        slope1 = dy / dx;
        slope2 = dz / dx;
    }
    else if (ay >= az && ay > 1e-12f) {
        v0 = volume_geometry.vox_y;
        v1 = volume_geometry.vox_x;
        v2 = volume_geometry.vox_z;
        slope1 = dx / dy;
        slope2 = dz / dy;
    }
    else if (az > 1e-12f) {
        v0 = volume_geometry.vox_z;
        v1 = volume_geometry.vox_x;
        v2 = volume_geometry.vox_y;
        slope1 = dx / dz;
        slope2 = dy / dz;
    }
    else return 0.f;
    const float physical1 = slope1 * v1;
    const float physical2 = slope2 * v2;
    return sqrtf(v0 * v0 + physical1 * physical1 + physical2 * physical2);
}

template<int ZSize>
__global__ void voxel_driven_v3_kernel(cudaTextureObject_t projection_texture,
    const SVoxelDrivenView* __restrict__ geometry, float* __restrict__ volume,
    int views, SVolGeom volume_geometry)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int first_z = static_cast<int>(blockIdx.z) * ZSize;
    if (x >= volume_geometry.Nx || y >= volume_geometry.Ny ||
        first_z >= volume_geometry.Nz) return;

    const float3 origin = volume_geometry.origin();
    const float world_x = origin.x + x * volume_geometry.vox_x;
    const float world_y = origin.y + y * volume_geometry.vox_y;
    float sums[ZSize]{};

    for (int view_index = 0; view_index < views; ++view_index) {
        const SVoxelDrivenView& view = geometry[view_index];
#pragma unroll
        for (int local_z = 0; local_z < ZSize; ++local_z) {
            const int z = first_z + local_z;
            if (z >= volume_geometry.Nz) continue;
            const float3 voxel = make_float3(world_x, world_y,
                origin.z + z * volume_geometry.vox_z);
            float3 detector{};
            if (!intersect_detector_cylinder(view, voxel, detector)) continue;

            const float3 relative = sub3(detector, to3(view.cylinder_center));
            const float axial_mm = dot3(relative, to3(view.axis_unit));
            const float3 radial_point = sub3(relative,
                mul3(to3(view.axis_unit), axial_mm));
            const float cosine_component = dot3(radial_point,
                to3(view.radial_unit));
            const float sine_component = dot3(radial_point,
                to3(view.tangent_unit));
            const float angle = atan2f(sine_component, cosine_component);
            const float channel = view.principal_u + angle *
                view.inv_channel_angle_step_rad;
            const float row = view.principal_v + axial_mm *
                view.inv_row_step_mm;
            const float projection = tex3D<float>(projection_texture,
                channel + 0.5f, row + 0.5f, view_index + 0.5f);
            const float3 direction = sub3(voxel, to3(view.source));
            sums[local_z] += projection *
                physical_main_axis_length(direction, volume_geometry);
        }
    }

    const size_t slice = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny;
#pragma unroll
    for (int local_z = 0; local_z < ZSize; ++local_z) {
        const int z = first_z + local_z;
        if (z >= volume_geometry.Nz) continue;
        volume[static_cast<size_t>(z) * slice +
            static_cast<size_t>(y) * volume_geometry.Nx + x] += sums[local_z];
    }
}

__device__ float component(float3 value, int axis)
{ return axis == 0 ? value.x : (axis == 1 ? value.y : value.z); }

__device__ int dimension(const SVolGeom& volume, int axis)
{ return axis == 0 ? volume.Nx : (axis == 1 ? volume.Ny : volume.Nz); }

__device__ float voxel_size(const SVolGeom& volume, int axis)
{ return axis == 0 ? volume.vox_x : (axis == 1 ? volume.vox_y : volume.vox_z); }

__device__ bool project_to_detector(const SVoxelDrivenView& view, float3 point,
    float& channel, float& row)
{
    float3 detector{};
    if (!intersect_detector_cylinder(view, point, detector)) return false;
    const float3 relative = sub3(detector, to3(view.cylinder_center));
    const float axial = dot3(relative, to3(view.axis_unit));
    const float3 radial = sub3(relative,
        mul3(to3(view.axis_unit), axial));
    const float angle = atan2f(dot3(radial, to3(view.tangent_unit)),
        dot3(radial, to3(view.radial_unit)));
    channel = view.principal_u + angle * view.inv_channel_angle_step_rad;
    row = view.principal_v + axial * view.inv_row_step_mm;
    return isfinite(channel) && isfinite(row);
}

__device__ float3 index_to_world(float3 index, const SVolGeom& volume)
{
    const float3 origin = volume.origin();
    return make_float3(origin.x + index.x * volume.vox_x,
        origin.y + index.y * volume.vox_y,
        origin.z + index.z * volume.vox_z);
}

__device__ float3 world_to_index(float3 point, const SVolGeom& volume)
{
    const float3 origin = volume.origin();
    return make_float3((point.x - origin.x) * volume.tmp_rcp_vox_x,
        (point.y - origin.y) * volume.tmp_rcp_vox_y,
        (point.z - origin.z) * volume.tmp_rcp_vox_z);
}

__device__ float sample_layer(cudaTextureObject_t texture, float3 p, int axis,
    const SVolGeom& volume)
{
    int fixed = 0, a0 = 0, a1 = 0, b0 = 0, b1 = 0;
    float ta = 0.f, tb = 0.f;
    if (axis == 0) {
        fixed = static_cast<int>(floorf(p.x + 0.5f));
        if (fixed < 0 || fixed >= volume.Nx || p.y < 0.f ||
            p.y > volume.Ny - 1.f || p.z < 0.f || p.z > volume.Nz - 1.f)
            return 0.f;
        a0 = static_cast<int>(floorf(p.y)); a1 = min(a0 + 1, volume.Ny - 1);
        b0 = static_cast<int>(floorf(p.z)); b1 = min(b0 + 1, volume.Nz - 1);
        ta = p.y - a0; tb = p.z - b0;
        return tex3D<float>(texture, fixed + 0.5f, a0 + 0.5f, b0 + 0.5f) * (1.f-ta)*(1.f-tb) +
            tex3D<float>(texture, fixed + 0.5f, a1 + 0.5f, b0 + 0.5f) * ta*(1.f-tb) +
            tex3D<float>(texture, fixed + 0.5f, a0 + 0.5f, b1 + 0.5f) * (1.f-ta)*tb +
            tex3D<float>(texture, fixed + 0.5f, a1 + 0.5f, b1 + 0.5f) * ta*tb;
    }
    if (axis == 1) {
        fixed = static_cast<int>(floorf(p.y + 0.5f));
        if (fixed < 0 || fixed >= volume.Ny || p.x < 0.f ||
            p.x > volume.Nx - 1.f || p.z < 0.f || p.z > volume.Nz - 1.f)
            return 0.f;
        a0 = static_cast<int>(floorf(p.x)); a1 = min(a0 + 1, volume.Nx - 1);
        b0 = static_cast<int>(floorf(p.z)); b1 = min(b0 + 1, volume.Nz - 1);
        ta = p.x - a0; tb = p.z - b0;
        return tex3D<float>(texture, a0 + 0.5f, fixed + 0.5f, b0 + 0.5f) * (1.f-ta)*(1.f-tb) +
            tex3D<float>(texture, a1 + 0.5f, fixed + 0.5f, b0 + 0.5f) * ta*(1.f-tb) +
            tex3D<float>(texture, a0 + 0.5f, fixed + 0.5f, b1 + 0.5f) * (1.f-ta)*tb +
            tex3D<float>(texture, a1 + 0.5f, fixed + 0.5f, b1 + 0.5f) * ta*tb;
    }
    fixed = static_cast<int>(floorf(p.z + 0.5f));
    if (fixed < 0 || fixed >= volume.Nz || p.x < 0.f ||
        p.x > volume.Nx - 1.f || p.y < 0.f || p.y > volume.Ny - 1.f)
        return 0.f;
    a0 = static_cast<int>(floorf(p.x)); a1 = min(a0 + 1, volume.Nx - 1);
    b0 = static_cast<int>(floorf(p.y)); b1 = min(b0 + 1, volume.Ny - 1);
    ta = p.x - a0; tb = p.y - b0;
    return tex3D<float>(texture, a0 + 0.5f, b0 + 0.5f, fixed + 0.5f) * (1.f-ta)*(1.f-tb) +
        tex3D<float>(texture, a1 + 0.5f, b0 + 0.5f, fixed + 0.5f) * ta*(1.f-tb) +
        tex3D<float>(texture, a0 + 0.5f, b1 + 0.5f, fixed + 0.5f) * (1.f-ta)*tb +
        tex3D<float>(texture, a1 + 0.5f, b1 + 0.5f, fixed + 0.5f) * ta*tb;
}

__device__ float weight_for_voxel(float3 p, int axis, int x, int y, int z,
    const SVolGeom& volume)
{
    if (axis == 0) {
        if (static_cast<int>(floorf(p.x + 0.5f)) != x || p.y < 0.f ||
            p.y > volume.Ny - 1.f || p.z < 0.f || p.z > volume.Nz - 1.f)
            return 0.f;
        const int y0 = static_cast<int>(floorf(p.y));
        const int z0 = static_cast<int>(floorf(p.z));
        const int y1 = min(y0 + 1, volume.Ny - 1);
        const int z1 = min(z0 + 1, volume.Nz - 1);
        const float wy = (y == y0 ? 1.f - (p.y-y0) : 0.f) +
            (y == y1 ? p.y-y0 : 0.f);
        const float wz = (z == z0 ? 1.f - (p.z-z0) : 0.f) +
            (z == z1 ? p.z-z0 : 0.f);
        return wy * wz;
    }
    if (axis == 1) {
        if (static_cast<int>(floorf(p.y + 0.5f)) != y || p.x < 0.f ||
            p.x > volume.Nx - 1.f || p.z < 0.f || p.z > volume.Nz - 1.f)
            return 0.f;
        const int x0 = static_cast<int>(floorf(p.x));
        const int z0 = static_cast<int>(floorf(p.z));
        const int x1 = min(x0 + 1, volume.Nx - 1);
        const int z1 = min(z0 + 1, volume.Nz - 1);
        const float wx = (x == x0 ? 1.f - (p.x-x0) : 0.f) +
            (x == x1 ? p.x-x0 : 0.f);
        const float wz = (z == z0 ? 1.f - (p.z-z0) : 0.f) +
            (z == z1 ? p.z-z0 : 0.f);
        return wx * wz;
    }
    if (static_cast<int>(floorf(p.z + 0.5f)) != z || p.x < 0.f ||
        p.x > volume.Nx - 1.f || p.y < 0.f || p.y > volume.Ny - 1.f)
        return 0.f;
    const int x0 = static_cast<int>(floorf(p.x));
    const int y0 = static_cast<int>(floorf(p.y));
    const int x1 = min(x0 + 1, volume.Nx - 1);
    const int y1 = min(y0 + 1, volume.Ny - 1);
    const float wx = (x == x0 ? 1.f - (p.x-x0) : 0.f) +
        (x == x1 ? p.x-x0 : 0.f);
    const float wy = (y == y0 ? 1.f - (p.y-y0) : 0.f) +
        (y == y1 ? p.y-y0 : 0.f);
    return wx * wy;
}

__global__ void layered_forward_kernel(cudaTextureObject_t volume_texture,
    float* projection, const SVoxelDrivenView* geometry,
    const SSiddonChannelRay* channel_rays, int views, int rows, int channels,
    SVolGeom volume, CudaOp::EWriteMode write_mode)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t quotient = index / channels;
        const int row = static_cast<int>(quotient % rows);
        const int view_index = static_cast<int>(quotient / rows);
        const SVoxelDrivenView& view = geometry[view_index];
        const float3 source = to3(view.source);
        const float3 principal_direction = to3(channel_rays[
            static_cast<size_t>(view_index) * channels + channel]
            .ray_at_principal_row);
        const float3 direction = add3(principal_direction,
            mul3(to3(view.axis_unit),
                (row - view.principal_v) / view.inv_row_step_mm));
        const int axis = static_cast<int>(view.main_axis);
        const float denominator = component(direction, axis);
        float sum = 0.f;
        if (fabsf(denominator) > 1e-12f) {
            const float3 origin = volume.origin();
            const float first = component(origin, axis);
            const float spacing = voxel_size(volume, axis);
            const int layers = dimension(volume, axis);
            const float length = physical_main_axis_length(direction, volume);
            for (int layer = 0; layer < layers; ++layer) {
                const float t = (first + layer * spacing -
                    component(source, axis)) / denominator;
                if (t <= 0.f || t >= 1.f) continue;
                sum += sample_layer(volume_texture,
                    world_to_index(add3(source, mul3(direction, t)), volume),
                    axis, volume) * length;
            }
        }
        if (CudaOp::accumulates(write_mode)) projection[index] += sum;
        else projection[index] = sum;
    }
}

__global__ void exact_gather_kernel(cudaTextureObject_t projection_texture,
    float* volume_output, const SVoxelDrivenView* geometry,
    const SSiddonChannelRay* channel_rays, int views, int rows, int channels,
    SVolGeom volume)
{
    const size_t voxel_count = static_cast<size_t>(volume.Nx) * volume.Ny * volume.Nz;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < voxel_count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(index % volume.Nx);
        const size_t quotient = index / volume.Nx;
        const int y = static_cast<int>(quotient % volume.Ny);
        const int z = static_cast<int>(quotient / volume.Ny);
        const float3 voxel_index = make_float3(static_cast<float>(x),
            static_cast<float>(y), static_cast<float>(z));
        const float3 voxel = index_to_world(voxel_index, volume);
        float sum = 0.f;
        for (int view_index = 0; view_index < views; ++view_index) {
            const SVoxelDrivenView& view = geometry[view_index];
            const float3 source = to3(view.source);
            const int first_axis = static_cast<int>(view.main_axis);
            const int end_axis = first_axis + 1;
            for (int axis = first_axis; axis < end_axis; ++axis) {
              // Joseph has compact support. Project the voxel centre once and
              // validate only nearby detector-centre rays; do not search a
              // detector-space box obtained from four extra projections.
              float detector_channel = 0.f, detector_row = 0.f;
              if (!project_to_detector(view, voxel, detector_channel,
                  detector_row)) continue;
              constexpr int support_radius = 3;
              const int center_channel = static_cast<int>(floorf(
                  detector_channel + 0.5f));
              const int center_row = static_cast<int>(floorf(
                  detector_row + 0.5f));
              const int channel_begin = max(0, center_channel - support_radius);
              const int channel_end = min(channels - 1,
                  center_channel + support_radius);
              const int row_begin = max(0, center_row - support_radius);
              const int row_end = min(rows - 1, center_row + support_radius);
              for (int row = row_begin; row <= row_end; ++row) {
                for (int channel = channel_begin; channel <= channel_end; ++channel) {
                    const float3 principal_direction = to3(channel_rays[
                        static_cast<size_t>(view_index) * channels + channel]
                        .ray_at_principal_row);
                    const float3 direction = add3(principal_direction,
                        mul3(to3(view.axis_unit),
                            (row - view.principal_v) / view.inv_row_step_mm));
                    const float denominator = component(direction, axis);
                    if (fabsf(denominator) < 1e-12f) continue;
                    const float t = (component(voxel, axis) -
                        component(source, axis)) / denominator;
                    if (t <= 0.f || t >= 1.f) continue;
                    const float weight = weight_for_voxel(world_to_index(
                        add3(source, mul3(direction, t)), volume), axis,
                        x, y, z, volume);
                    if (weight == 0.f) continue;
                    const float projection = tex3D<float>(projection_texture,
                        channel + 0.5f, row + 0.5f, view_index + 0.5f);
                    sum += projection * weight *
                        physical_main_axis_length(direction, volume);
                }
              }
            }
        }
        volume_output[index] += sum;
    }
}

__device__ double footprint_atomic_add_double(double* address, double value)
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

__device__ void footprint_splat(double* projection, int views, int rows,
    int channels, int view, int channel, int row, double value)
{
    if (view < 0 || view >= views || channel < 0 || channel >= channels ||
        row < 0 || row >= rows) return;
    footprint_atomic_add_double(projection +
        (static_cast<size_t>(view) * rows + row) * channels + channel, value);
}

__device__ float footprint_read(const float* projection, int views, int rows,
    int channels, int view, int channel, int row)
{
    if (view < 0 || view >= views || channel < 0 || channel >= channels ||
        row < 0 || row >= rows) return 0.f;
    return projection[(static_cast<size_t>(view) * rows + row) * channels +
        channel];
}

__global__ void footprint_forward_kernel(const float* volume_input,
    double* projection, const SVoxelDrivenView* geometry, int views, int rows,
    int channels, SVolGeom volume, size_t voxel_count)
{
    const int view_index = blockIdx.y;
    if (view_index >= views) return;
    const SVoxelDrivenView& view = geometry[view_index];
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < voxel_count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float value = volume_input[index];
        if (value == 0.f) continue;
        const int x = static_cast<int>(index % volume.Nx);
        const size_t quotient = index / volume.Nx;
        const int y = static_cast<int>(quotient % volume.Ny);
        const int z = static_cast<int>(quotient / volume.Ny);
        const float3 voxel = index_to_world(make_float3(
            static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)),
            volume);
        float channel_f = 0.f, row_f = 0.f;
        if (!project_to_detector(view, voxel, channel_f, row_f)) continue;
        const int channel0 = static_cast<int>(floorf(channel_f));
        const int row0 = static_cast<int>(floorf(row_f));
        const float wc = channel_f - channel0;
        const float wr = row_f - row0;
        const float length = physical_main_axis_length(
            sub3(voxel, to3(view.source)), volume);
        const float scaled = value * length;
        footprint_splat(projection, views, rows, channels, view_index,
            channel0, row0, static_cast<double>(scaled) * (1.f-wc) * (1.f-wr));
        footprint_splat(projection, views, rows, channels, view_index,
            channel0+1, row0, static_cast<double>(scaled) * wc * (1.f-wr));
        footprint_splat(projection, views, rows, channels, view_index,
            channel0, row0+1, static_cast<double>(scaled) * (1.f-wc) * wr);
        footprint_splat(projection, views, rows, channels, view_index,
            channel0+1, row0+1, static_cast<double>(scaled) * wc * wr);
    }
}

__global__ void footprint_convert_projection_kernel(const double* source,
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
    float* volume_output, const SVoxelDrivenView* geometry, int views,
    int rows, int channels, SVolGeom volume, size_t voxel_count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < voxel_count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(index % volume.Nx);
        const size_t quotient = index / volume.Nx;
        const int y = static_cast<int>(quotient % volume.Ny);
        const int z = static_cast<int>(quotient / volume.Ny);
        const float3 voxel = index_to_world(make_float3(
            static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)),
            volume);
        float sum = 0.f;
        for (int view_index = 0; view_index < views; ++view_index) {
            const SVoxelDrivenView& view = geometry[view_index];
            float channel_f = 0.f, row_f = 0.f;
            if (!project_to_detector(view, voxel, channel_f, row_f)) continue;
            const int channel0 = static_cast<int>(floorf(channel_f));
            const int row0 = static_cast<int>(floorf(row_f));
            const float wc = channel_f - channel0;
            const float wr = row_f - row0;
            const float detector_value =
                footprint_read(projection, views, rows, channels, view_index,
                    channel0, row0) * (1.f-wc) * (1.f-wr) +
                footprint_read(projection, views, rows, channels, view_index,
                    channel0+1, row0) * wc * (1.f-wr) +
                footprint_read(projection, views, rows, channels, view_index,
                    channel0, row0+1) * (1.f-wc) * wr +
                footprint_read(projection, views, rows, channels, view_index,
                    channel0+1, row0+1) * wc * wr;
            sum += detector_value * physical_main_axis_length(
                sub3(voxel, to3(view.source)), volume);
        }
        volume_output[index] += sum;
    }
}

} // namespace

void launch_joseph_backproject_v3(cudaTextureObject_t projection_texture,
    const SVoxelDrivenView* geometry, float* volume, int views,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    CudaOp::clearIfOverwrite(volume,
        static_cast<size_t>(volume_geometry.Nx) * volume_geometry.Ny *
            volume_geometry.Nz,
        CudaOp::writeMode(accumulate), stream);
    constexpr int z_size = 4;
    const dim3 block(16, 16, 1);
    const dim3 grid((volume_geometry.Nx + block.x - 1) / block.x,
        (volume_geometry.Ny + block.y - 1) / block.y,
        (volume_geometry.Nz + z_size - 1) / z_size);
    voxel_driven_v3_kernel<z_size><<<grid, block, 0, stream>>>(
        projection_texture, geometry, volume, views, volume_geometry);
    YK_CUDA_KERNEL_CHECK();
}

void launch_joseph_layered_forward(cudaTextureObject_t volume_texture,
    float* projection, const SVoxelDrivenView* geometry,
    const SSiddonChannelRay* channel_rays, int views, int rows, int channels,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(views) * rows * channels;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((count + threads - 1) / threads,
        65535));
    layered_forward_kernel<<<blocks, threads, 0, stream>>>(volume_texture,
        projection, geometry, channel_rays, views, rows, channels,
        volume_geometry, CudaOp::writeMode(accumulate));
    YK_CUDA_KERNEL_CHECK();
}

void launch_joseph_exact_gather(cudaTextureObject_t projection_texture,
    float* volume, const SVoxelDrivenView* geometry,
    const SSiddonChannelRay* channel_rays, int views, int rows, int channels,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    const size_t count = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    CudaOp::clearIfOverwrite(volume, count, CudaOp::writeMode(accumulate), stream);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((count + threads - 1) / threads,
        65535));
    exact_gather_kernel<<<blocks, threads, 0, stream>>>(projection_texture,
        volume, geometry, channel_rays, views, rows, channels, volume_geometry);
    YK_CUDA_KERNEL_CHECK();
}

void launch_footprint_forward(const float* volume, float* projection,
    const SVoxelDrivenView* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    const size_t voxel_count = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    const size_t plane_count = static_cast<size_t>(rows) * channels;
    constexpr int max_batch_views = 16;
    const int allocation_views = views < max_batch_views ? views :
        max_batch_views;
    double* accumulator = nullptr;
    YK_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&accumulator),
        static_cast<size_t>(allocation_views) * plane_count * sizeof(double),
        stream));
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>(
        (voxel_count + threads - 1) / threads, 65535));
    for (int view_begin = 0; view_begin < views;
         view_begin += max_batch_views) {
        const int batch_views = std::min(max_batch_views, views - view_begin);
        const size_t batch_count = static_cast<size_t>(batch_views) * plane_count;
        YK_CUDA_CHECK(cudaMemsetAsync(accumulator, 0,
            batch_count * sizeof(double), stream));
        footprint_forward_kernel<<<dim3(blocks, batch_views), threads, 0,
            stream>>>(volume, accumulator, geometry + view_begin, batch_views,
                rows, channels, volume_geometry, voxel_count);
        YK_CUDA_KERNEL_CHECK();
        footprint_convert_projection_kernel<<<static_cast<int>(std::min<size_t>(
            (batch_count + threads - 1) / threads, 65535)), threads, 0,
            stream>>>(accumulator, projection +
                static_cast<size_t>(view_begin) * plane_count, batch_count,
                accumulate);
        YK_CUDA_KERNEL_CHECK();
    }
    YK_CUDA_CHECK(cudaFreeAsync(accumulator, stream));
}

void launch_footprint_back(const float* projection, float* volume,
    const SVoxelDrivenView* geometry, int views, int rows, int channels,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    const size_t voxel_count = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    CudaOp::clearIfOverwrite(volume, voxel_count,
        CudaOp::writeMode(accumulate), stream);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>(
        (voxel_count + threads - 1) / threads, 65535));
    footprint_back_kernel<<<blocks, threads, 0, stream>>>(projection, volume,
        geometry, views, rows, channels, volume_geometry, voxel_count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
