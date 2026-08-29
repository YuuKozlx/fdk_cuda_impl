#include "CylFpBp/kernels/YkCylFdkLaunch.cuh"

#include <cmath>

#include "common/cuda/operators/YkOperatorKernelTypes.cuh"
#include "global/YkMacro.hpp"

namespace YK::CylFpBp::detail {
namespace {

__global__ void cyl_fdk_preweight_kernel(const float* input, float* output,
    const SCylFdkView* geometry, int channels, int rows, int views)
{
    const size_t count = static_cast<size_t>(channels) * rows * views;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t view_row = index / channels;
        const int row = static_cast<int>(view_row % rows);
        const int view_index = static_cast<int>(view_row / rows);
        const SCylFdkView& view = geometry[view_index];
        const float gamma = view.detector_center_angle_rad +
            (static_cast<float>(channel) - view.principal_u) *
                (1.f / view.inv_channel_angle_step_rad);
        const float q = view.detector_axial_offset_mm +
            (static_cast<float>(row) - view.principal_v) /
                view.inv_row_step_mm;
        const float cosine = cosf(gamma);
        const float cone = view.radius_mm /
            sqrtf(fmaxf(view.radius_mm * view.radius_mm + q * q, 1e-20f));
        output[index] = input[index] * cosine * cone * view.dtheta;
    }
}

__device__ float3 sub3(float3 a, float3 b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float dot3(float3 a, float3 b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 to3(float4 a) { return make_float3(a.x, a.y, a.z); }

struct DetectorCoordinate {
    float channel = 0.f;
    float row = 0.f;
    float axial_mm = 0.f;
    float source_distance_squared = 0.f;
    float central_depth_mm = 0.f;
};

// R=SDD 时圆柱轴经过源点。源点到体素方向 d 与圆柱面的交点比例为
// R/|d_perp|；U 是横向方向角，V 是交点沿圆柱轴的物理坐标。
__device__ bool map_voxel_to_detector(const SCylFdkView& view, float3 voxel,
    DetectorCoordinate& coordinate)
{
    const float3 direction = sub3(voxel, to3(view.source));
    const float axial_direction = dot3(direction, to3(view.axis_unit));
    const float radial_direction = dot3(direction, to3(view.radial_unit));
    const float tangent_direction = dot3(direction, to3(view.tangent_unit));
    const float transverse_squared = radial_direction * radial_direction +
        tangent_direction * tangent_direction;
    if (!(radial_direction > 1e-8f) || !(transverse_squared > 1e-12f))
        return false;

    const float transverse = sqrtf(transverse_squared);
    const float detector_scale = view.radius_mm / transverse;
    coordinate.axial_mm = axial_direction * detector_scale;
    coordinate.channel = view.principal_u +
        atan2f(tangent_direction, radial_direction) *
        view.inv_channel_angle_step_rad;
    coordinate.row = view.principal_v +
        (coordinate.axial_mm - view.detector_axial_offset_mm) *
            view.inv_row_step_mm;
    coordinate.source_distance_squared = transverse_squared +
        axial_direction * axial_direction;
    coordinate.central_depth_mm = dot3(direction, to3(view.depth_unit));
    return true;
}

struct FdkWeight {
    __device__ static float evaluate(const SCylFdkView& view,
        const DetectorCoordinate& coordinate, const SVolGeom&)
    {
        const float depth_squared = coordinate.central_depth_mm *
            coordinate.central_depth_mm;
        return depth_squared > 1e-12f
            ? view.sid_mm * view.sid_mm / depth_squared : 0.f;
    }
};

struct FdkMatchedWeight {
    __device__ static float evaluate(const SCylFdkView& view,
        const DetectorCoordinate& coordinate, const SVolGeom& volume_geometry)
    {
        if (!(coordinate.source_distance_squared > 1e-12f)) return 0.f;
        // 一般曲面的 detector Jacobian 为 L^3/(n·ray)。源中心圆柱上
        // n·ray=R，L²=R²+v_det²。再乘体素体积/探测器像素面积，得到与
        // 平板 BP_FDK_matched 相同约定下的圆柱权重。
        const float detector_distance_squared =
            view.radius_mm * view.radius_mm +
            coordinate.axial_mm * coordinate.axial_mm;
        const float detector_distance = sqrtf(detector_distance_squared);
        const float voxel_volume = volume_geometry.vox_x *
            volume_geometry.vox_y * volume_geometry.vox_z;
        return detector_distance_squared * detector_distance /
            (view.radius_mm * coordinate.source_distance_squared) *
            voxel_volume * view.inverse_pixel_area;
    }
};

template<typename Weight, int ZSize>
__global__ void cyl_fdk_bp_kernel(cudaTextureObject_t projection_texture,
    const SCylFdkView* __restrict__ geometry, float* __restrict__ volume,
    int views, SVolGeom volume_geometry, CudaOp::EWriteMode write_mode)
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
        const SCylFdkView& view = geometry[view_index];
#pragma unroll
        for (int local_z = 0; local_z < ZSize; ++local_z) {
            const int z = first_z + local_z;
            if (z >= volume_geometry.Nz) continue;
            const float3 voxel = make_float3(world_x, world_y,
                origin.z + z * volume_geometry.vox_z);
            DetectorCoordinate coordinate{};
            if (!map_voxel_to_detector(view, voxel, coordinate)) continue;
            const float projection = tex3D<float>(projection_texture,
                coordinate.channel + 0.5f, coordinate.row + 0.5f,
                view_index + 0.5f);
            sums[local_z] += projection *
                Weight::evaluate(view, coordinate, volume_geometry);
        }
    }

    const size_t slice = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny;
#pragma unroll
    for (int local_z = 0; local_z < ZSize; ++local_z) {
        const int z = first_z + local_z;
        if (z >= volume_geometry.Nz) continue;
        const size_t index = static_cast<size_t>(z) * slice +
            static_cast<size_t>(y) * volume_geometry.Nx + x;
        volume[index] = (CudaOp::accumulates(write_mode) ? volume[index] : 0.f) +
            sums[local_z];
    }
}

template<typename Weight>
void launch(cudaTextureObject_t projection_texture,
    const SCylFdkView* geometry, float* volume, int views,
    const SVolGeom& volume_geometry, cudaStream_t stream,
    CudaOp::EWriteMode write_mode)
{
    constexpr int z_size = 4;
    const dim3 block(16, 16, 1);
    const dim3 grid((volume_geometry.Nx + block.x - 1) / block.x,
        (volume_geometry.Ny + block.y - 1) / block.y,
        (volume_geometry.Nz + z_size - 1) / z_size);
    cyl_fdk_bp_kernel<Weight, z_size><<<grid, block, 0, stream>>>(
        projection_texture, geometry, volume, views, volume_geometry, write_mode);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace

void launch_cyl_fdk_preweight(const float* input, float* output,
    const SCylFdkView* geometry, int channels, int rows, int views,
    cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(channels) * rows * views;
    const int blocks = static_cast<int>(std::min<size_t>(
        (count + 255) / 256, 65535));
    cyl_fdk_preweight_kernel<<<blocks, 256, 0, stream>>>(
        input, output, geometry, channels, rows, views);
    YK_CUDA_KERNEL_CHECK();
}

void launch_cyl_fdk_bp(cudaTextureObject_t projection_texture,
    const SCylFdkView* geometry, float* volume, int views,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    launch<FdkWeight>(projection_texture, geometry, volume, views,
        volume_geometry, stream, CudaOp::writeMode(accumulate));
}

void launch_cyl_fdk_matched_bp(cudaTextureObject_t projection_texture,
    const SCylFdkView* geometry, float* volume, int views,
    const SVolGeom& volume_geometry, cudaStream_t stream, bool accumulate)
{
    launch<FdkMatchedWeight>(projection_texture, geometry, volume, views,
        volume_geometry, stream, CudaOp::writeMode(accumulate));
}

} // namespace YK::CylFpBp::detail
