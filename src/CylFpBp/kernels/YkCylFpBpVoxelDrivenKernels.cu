#include "CylFpBp/kernels/YkCylVoxelDrivenLaunch.cuh"

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

} // namespace

void launch_voxel_driven_v3(cudaTextureObject_t projection_texture,
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

} // namespace YK::CylFpBp::detail
