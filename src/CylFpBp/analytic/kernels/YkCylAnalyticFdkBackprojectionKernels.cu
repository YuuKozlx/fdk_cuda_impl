#include "CylFpBp/analytic/kernels/YkCylAnalyticFdkLaunch.cuh"

#include <cmath>

#include "global/YkMacro.hpp"

namespace YK::CylFpBp::detail {
namespace {

__device__ float3 sub3(float3 a, float3 b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float dot3(float3 a, float3 b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 to3(float4 a) { return make_float3(a.x, a.y, a.z); }

__global__ void cyl_analytic_fdk_bp_kernel(cudaTextureObject_t projection,
    const SCylFdkView* geometry, float* volume, int views, SVolGeom vg,
    bool accumulate)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= vg.Nx || y >= vg.Ny || z >= vg.Nz) return;
    const float3 origin = vg.origin();
    const float3 voxel = make_float3(origin.x + x * vg.vox_x,
        origin.y + y * vg.vox_y, origin.z + z * vg.vox_z);
    float sum = 0.f;
    for (int i = 0; i < views; ++i) {
        const SCylFdkView& view = geometry[i];
        const float3 d = sub3(voxel, to3(view.source));
        const float axial = dot3(d, to3(view.axis_unit));
        const float radial = dot3(d, to3(view.radial_unit));
        const float tangent = dot3(d, to3(view.tangent_unit));
        const float l2 = radial * radial + tangent * tangent;
        if (!(radial > 1e-8f) || !(l2 > 1e-12f)) continue;
        const float scale = view.radius_mm / sqrtf(l2);
        const float gamma = atan2f(tangent, radial);
        const float q = axial * scale;
        // 解析管线的投影数组坐标已经以 detectorCenter 对应的 principal_u
        // 为原点。gamma 是相对于该 view 的 radial_unit 的局部扇角；旧版
        // Cyl-FDK 也采用这个坐标约定。detector_center_angle_rad 只属于
        // 一般 FDK-style BP 的几何元数据，不能在这里再次扣除，否则会
        // 把采样整体平移到探测器边界外并触发 border=0。
        const float u = view.principal_u +
            gamma * view.inv_channel_angle_step_rad;
        const float v = view.principal_v +
            (q - view.detector_axial_offset_mm) * view.inv_row_step_mm;
        // R=SDD 约束描述的是 map 后虚拟探测器的曲率，并不把 SDD 变成
        // 源到旋转中心的距离。解析 FDK 深度权重仍以 SID 为标尺，并使用
        // 体素在中央射线方向上的深度；用 SDD^2/l2 会在等中心额外放大
        // (SDD/SID)^2，直接破坏材料值量级。
        const float central_depth = dot3(d, to3(view.depth_unit));
        const float central_depth_squared = central_depth * central_depth;
        const float depth_weight = central_depth_squared > 1e-12f
            ? view.sid_mm * view.sid_mm / central_depth_squared : 0.f;
        // dtheta 已在解析预加权阶段乘入投影；此处只做一次深度补偿和
        // 视图累加，避免重复角度积分权重导致结果随视图数缩放。
        sum += tex3D<float>(projection, u + .5f, v + .5f, i + .5f) *
            depth_weight;
    }
    const size_t index = (static_cast<size_t>(z) * vg.Ny + y) * vg.Nx + x;
    volume[index] = accumulate ? volume[index] + sum : sum;
}

} // namespace

void launch_cyl_analytic_fdk_bp(cudaTextureObject_t projection,
    const SCylFdkView* geometry, float* volume, int views,
    const SVolGeom& vg, cudaStream_t stream, bool accumulate)
{
    const dim3 block(8, 8, 4);
    const dim3 grid((vg.Nx + block.x - 1) / block.x,
        (vg.Ny + block.y - 1) / block.y, (vg.Nz + block.z - 1) / block.z);
    cyl_analytic_fdk_bp_kernel<<<grid, block, 0, stream>>>(projection, geometry,
        volume, views, vg, accumulate);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::CylFpBp::detail
