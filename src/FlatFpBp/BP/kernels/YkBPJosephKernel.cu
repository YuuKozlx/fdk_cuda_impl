#include "YkFlatJosephBpLaunch.cuh"
#include "global/YkMacro.hpp"
#include "YkBPHelpers.cuh"
#include "common/cuda/operators/YkSamplingReaders.cuh"
#include <FlatFpBp/BP/YkBPCommon.cuh>

namespace YK {
    namespace Bp {
        namespace detail {
            // ray driven 
            // 缺陷原子操作多
            template<typename DIR, int kStepDenom, typename SinoReader>
            __global__ void joseph_bp_kernel(SinoReader sino,
                const SConeProjGeomVec* d_views_vox,   // 全局指针
                float* d_vol,
                SVolGeom                g,
                int Nu, int Nv,
                int startSlice,
                int startAngle, int endAngle,      // 全局角度范围
                float fMainAxisVox)
            {
                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
                const int angle = startAngle + localAngle;
                if (angle >= endAngle) return;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int detV = (blockIdx.x / nUBlocks) * kDetBlockV + threadIdx.z;  // ← 并行
                if (detV >= Nv) return;

                const SConeProjGeomVec& v = d_views_vox[angle];
                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
                const float fDetUX = v.detU.x, fDetUY = v.detU.y, fDetUZ = v.detU.z;
                const float fDetVX = v.detV.x, fDetVY = v.detV.y, fDetVZ = v.detV.z;
                const float fDetSX = v.detS.x;
                const float fDetSY = v.detS.y;
                const float fDetSZ = v.detS.z;

                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);
                const int nSlicesOrig = nSlices / kStepDenom;
                const int endSlice = min(startSlice + kBlockSlices, nSlices);
                constexpr float step = 1.f / (float)kStepDenom;

                // detV 直接用，不再 for 循环
                const float fDetX = fDetSX + detU * fDetUX + detV * fDetVX;
                const float fDetY = fDetSY + detU * fDetUY + detV * fDetVY;
                const float fDetZ = fDetSZ + detU * fDetUZ + detV * fDetVZ;

                const float den = DIR::c0(fSrcX, fSrcY, fSrcZ)
                    - DIR::c0(fDetX, fDetY, fDetZ);
                if (fabsf(den) < 1e-8f) return;

                const float rcp_den = __frcp_rn(den);
                const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ)
                    - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_den;
                const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ)
                    - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_den;
                const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ)
                    - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ)
                    - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                const float fDistCorr = __fsqrt_rn(a1 * a1 + a2 * a2 + 1.f);

                const float proj_val = sino.read(angle, detV, detU, Nu, Nv);
                const float contrib_base = proj_val * fDistCorr * step * fMainAxisVox;

                float f0 = startSlice * step + 0.5f * step;
                float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                    + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
                float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                    + b2 + 0.5f * nDim2 - 0.5f + 0.5f;

                const float a1_step = a1 * step;
                const float a2_step = a2 * step;

                for (int s = startSlice; s < endSlice; ++s)
                {
                    const int i0 = (int)floorf(f0 - 0.5f);
                    const int i1 = (int)floorf(f1 - 0.5f);
                    const int i2 = (int)floorf(f2 - 0.5f);

                    const float w1 = fmaxf(0.f, fminf(1.f, (f1 - 0.5f) - i1));
                    const float w2 = fmaxf(0.f, fminf(1.f, (f2 - 0.5f) - i2));

                    if (i0 >= 0 && i0 < nSlicesOrig) {
                        auto add = [&](int j1, int j2, float w) {
                            if (j1 < 0 || j1 >= nDim1) return;
                            if (j2 < 0 || j2 >= nDim2) return;
                            const int3 vox = DIR::toVoxel(i0, j1, j2);
                            if (vox.x < 0 || vox.x >= g.Nx) return;
                            if (vox.y < 0 || vox.y >= g.Ny) return;
                            if (vox.z < 0 || vox.z >= g.Nz) return;
                            const size_t idx = (size_t)vox.z * g.Ny * g.Nx
                                + (size_t)vox.y * g.Nx
                                + (size_t)vox.x;
                            atomicAdd(&d_vol[idx], contrib_base * w); // 原子操作多，速度慢。
                            };
                        add(i1, i2, (1.f - w1) * (1.f - w2));
                        add(i1 + 1, i2, w1 * (1.f - w2));
                        add(i1, i2 + 1, (1.f - w1) * w2);
                        add(i1 + 1, i2 + 1, w1 * w2);
                    }

                    f0 += step;
                    f1 += a1_step;
                    f2 += a2_step;
                }
            }



            template<typename DIR, int kStepDenom, typename SinoReader>
            static void joseph_bp_launch_group_impl(
                SinoReader sino,
                const SConeProjGeomVec* d_views_vox,
                float* d_vol,
                const SVolGeom& g,
                int Nu, int Nv,
                int startAngle, int endAngle,
                cudaStream_t stream)
            {
                const int Na = endAngle - startAngle;
                if (Na <= 0) return;

                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
                const int nABlocks = (Na + kAnglesPerBlock - 1) / kAnglesPerBlock;

                dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV);
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                const float fVox = DIR::voxSize(g);

                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    joseph_bp_kernel<DIR, kStepDenom><<<grid, block, 0, stream>>>(
                        sino,
                        d_views_vox,
                        d_vol, g, Nu, Nv,
                        s, startAngle, endAngle,
                        fVox);
                }
            }

            // ── dispatch：d_views_vox 全局指针，不偏移 ──────────────────────
            template<typename SinoReader>
            static void joseph_bp_dispatch(
                SinoReader sino,
                const std::vector<SConeProjGeomVec>& h_views,
                const SConeProjGeomVec* d_views_vox,   // 全局指针
                float* d_vol,
                const SVolGeom& g,
                int Na, int Nu, int Nv,
                cudaStream_t stream,
                BpStepSuperSample ss)
            {
                // 中心射线由源点和探测器几何中心派生，不接受独立方向输入。
                const bool validGeometry = forEachAxisRun(h_views, Na, Nu, Nv,
                    [&](MainAxis ax, CudaOp::SViewRange range) {
                    const int i = range.first;
                    const int j = range.end();
                    switch (ax) {
                    case MainAxis::X:
                        switch (ss) {
                        case BpStepSuperSample::x1:
                            joseph_bp_launch_group_impl<DirX, 1>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x2:
                            joseph_bp_launch_group_impl<DirX, 2>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x4:
                            joseph_bp_launch_group_impl<DirX, 4>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        }
                        break;
                    case MainAxis::Y:
                        switch (ss) {
                        case BpStepSuperSample::x1:
                            joseph_bp_launch_group_impl<DirY, 1>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x2:
                            joseph_bp_launch_group_impl<DirY, 2>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x4:
                            joseph_bp_launch_group_impl<DirY, 4>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        }
                        break;
                    case MainAxis::Z:
                        switch (ss) {
                        case BpStepSuperSample::x1:
                            joseph_bp_launch_group_impl<DirZ, 1>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x2:
                            joseph_bp_launch_group_impl<DirZ, 2>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x4:
                            joseph_bp_launch_group_impl<DirZ, 4>(
                                sino, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        }
                        break;
                    }
                });
                if (!validGeometry)
                    YK_LOGE("Joseph BP launch rejected invalid projection geometry");
            }

        } // namespace detail

        void joseph_bp_launch(
            const float* d_sino,
            const std::vector<SConeProjGeomVec>& h_views,
            const SConeProjGeomVec* d_views_vox,
            float* d_vol,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            BpStepSuperSample ss)
        {
            CudaOp::clearIfOverwrite(d_vol,
                static_cast<size_t>(g.Nx) * g.Ny * g.Nz,
                CudaOp::writeMode(accumulate), stream);
            detail::joseph_bp_dispatch(
                CudaOp::RawProjectionPointReader{ d_sino, 0 }, h_views,
                d_views_vox, d_vol,
                g, Na, Nu, Nv, stream, ss);
            YK_CUDA_KERNEL_CHECK();
        }

        void joseph_bp_launch(
            cudaTextureObject_t                  sinoTex,
            const std::vector<SConeProjGeomVec>& h_views,
            const SConeProjGeomVec* d_views_vox,
            float* d_vol,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            BpStepSuperSample ss)
        {
            CudaOp::clearIfOverwrite(d_vol,
                static_cast<size_t>(g.Nx) * g.Ny * g.Nz,
                CudaOp::writeMode(accumulate), stream);
            detail::joseph_bp_dispatch(
                CudaOp::TextureProjectionPointReader{ sinoTex, 0 }, h_views,
                d_views_vox, d_vol,
                g, Na, Nu, Nv, stream, ss);
            YK_CUDA_KERNEL_CHECK();
        }

    } // namespace Bp
} // namespace YK

namespace YK::Bp {
namespace {

__device__ float3 flat_add(float3 a, float3 b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 flat_sub(float3 a, float3 b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 flat_mul(float3 a, float s)
{ return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float flat_dot(float3 a, float3 b)
{ return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ float3 flat_cross(float3 a, float3 b)
{ return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ float3 flat_xyz(float4 a) { return make_float3(a.x, a.y, a.z); }
__device__ float flat_component(float3 a, int axis)
{ return axis == 0 ? a.x : (axis == 1 ? a.y : a.z); }
__device__ int flat_dimension(const SVolGeom& g, int axis)
{ return axis == 0 ? g.Nx : (axis == 1 ? g.Ny : g.Nz); }
__device__ float flat_voxel_size(const SVolGeom& g, int axis)
{ return axis == 0 ? g.vox_x : (axis == 1 ? g.vox_y : g.vox_z); }

__device__ int flat_main_axis(const SConeProjGeomVec& view, int channels,
    int rows)
{
    const float3 source = flat_xyz(view.src);
    const float3 detector = flat_add(flat_add(flat_xyz(view.detS),
        flat_mul(flat_xyz(view.detU), 0.5f * (channels - 1))),
        flat_mul(flat_xyz(view.detV), 0.5f * (rows - 1)));
    const float3 direction = flat_sub(detector, source);
    const float ax = fabsf(direction.x), ay = fabsf(direction.y),
        az = fabsf(direction.z);
    return ax >= ay && ax >= az ? 0 : (ay >= az ? 1 : 2);
}

__device__ bool flat_project(const SConeProjGeomVec& view, float3 point,
    float& channel, float& row)
{
    const float3 source = flat_xyz(view.src);
    const float3 det_s = flat_xyz(view.detS);
    const float3 det_u = flat_xyz(view.detU);
    const float3 det_v = flat_xyz(view.detV);
    const float3 normal = flat_cross(det_u, det_v);
    const float3 direction = flat_sub(point, source);
    const float denominator = flat_dot(direction, normal);
    if (fabsf(denominator) < 1e-12f) return false;
    const float t = flat_dot(flat_sub(det_s, source), normal) / denominator;
    if (t <= 0.f) return false;
    const float3 relative = flat_sub(flat_add(source,
        flat_mul(direction, t)), det_s);
    const float u2 = flat_dot(det_u, det_u);
    const float v2 = flat_dot(det_v, det_v);
    if (u2 < 1e-16f || v2 < 1e-16f) return false;
    channel = flat_dot(relative, det_u) / u2;
    row = flat_dot(relative, det_v) / v2;
    return isfinite(channel) && isfinite(row);
}

__device__ float flat_joseph_weight(const SConeProjGeomVec& view,
    int channel, int row, int axis, int x, int y, int z,
    const SVolGeom& volume, float& physical_length)
{
    const float3 source = flat_xyz(view.src);
    const float3 detector = flat_add(flat_add(flat_xyz(view.detS),
        flat_mul(flat_xyz(view.detU), static_cast<float>(channel))),
        flat_mul(flat_xyz(view.detV), static_cast<float>(row)));
    const float3 direction = flat_sub(detector, source);
    const float denominator = flat_component(direction, axis);
    if (fabsf(denominator) < 1e-12f) return 0.f;
    const int axis1 = axis == 0 ? 1 : 0;
    const int axis2 = axis == 2 ? 1 : 2;
    const float slope1 = flat_component(direction, axis1) / denominator;
    const float slope2 = flat_component(direction, axis2) / denominator;
    const int index0 = axis == 0 ? x : (axis == 1 ? y : z);
    const int index1 = axis1 == 0 ? x : (axis1 == 1 ? y : z);
    const int index2 = axis2 == 0 ? x : (axis2 == 1 ? y : z);
    const float centered0 = index0 - 0.5f * (flat_dimension(volume, axis)-1);
    const float intercept1 = flat_component(source, axis1) -
        slope1 * flat_component(source, axis);
    const float intercept2 = flat_component(source, axis2) -
        slope2 * flat_component(source, axis);
    const float sample1 = slope1 * centered0 + intercept1 +
        0.5f * (flat_dimension(volume, axis1)-1);
    const float sample2 = slope2 * centered0 + intercept2 +
        0.5f * (flat_dimension(volume, axis2)-1);
    const float weight1 = fmaxf(0.f, 1.f - fabsf(sample1-index1));
    const float weight2 = fmaxf(0.f, 1.f - fabsf(sample2-index2));
    const float v0 = flat_voxel_size(volume, axis);
    const float v1 = flat_voxel_size(volume, axis1);
    const float v2 = flat_voxel_size(volume, axis2);
    physical_length = sqrtf(v0*v0 + slope1*slope1*v1*v1 +
        slope2*slope2*v2*v2);
    return weight1 * weight2;
}

__global__ void flat_joseph_gather_kernel(cudaTextureObject_t projection,
    const SConeProjGeomVec* views_world,
    const SConeProjGeomVec* views_voxel, float* volume_output,
    SVolGeom volume, int views, int channels, int rows, size_t voxel_count)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < voxel_count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(index % volume.Nx);
        const size_t quotient = index / volume.Nx;
        const int y = static_cast<int>(quotient % volume.Ny);
        const int z = static_cast<int>(quotient / volume.Ny);
        const float3 origin = volume.origin();
        const float3 voxel = make_float3(origin.x+x*volume.vox_x,
            origin.y+y*volume.vox_y, origin.z+z*volume.vox_z);
        float sum = 0.f;
        for (int view_index = 0; view_index < views; ++view_index) {
            float detector_u = 0.f, detector_v = 0.f;
            if (!flat_project(views_world[view_index], voxel,
                detector_u, detector_v)) continue;
            const int center_u = static_cast<int>(floorf(detector_u + 0.5f));
            const int center_v = static_cast<int>(floorf(detector_v + 0.5f));
            const int axis = flat_main_axis(views_world[view_index], channels, rows);
            constexpr int radius = 3;
            for (int v = max(0, center_v-radius);
                 v <= min(rows-1, center_v+radius); ++v) {
                for (int u = max(0, center_u-radius);
                     u <= min(channels-1, center_u+radius); ++u) {
                    float length = 0.f;
                    const float weight = flat_joseph_weight(
                        views_voxel[view_index], u, v, axis, x, y, z,
                        volume, length);
                    if (weight == 0.f) continue;
                    sum += tex3D<float>(projection, u+0.5f, v+0.5f,
                        view_index+0.5f) * weight * length;
                }
            }
        }
        volume_output[index] += sum;
    }
}

} // namespace

void joseph_bp_gather_launch(cudaTextureObject_t projection_texture,
    const SConeProjGeomVec* d_views_world,
    const SConeProjGeomVec* d_views_voxel, float* volume,
    const SVolGeom& volume_geometry, int views, int channels, int rows,
    bool accumulate, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(volume_geometry.Nx) *
        volume_geometry.Ny * volume_geometry.Nz;
    CudaOp::clearIfOverwrite(volume, count, CudaOp::writeMode(accumulate), stream);
    constexpr int threads = 256;
    const size_t needed_blocks = (count + threads - 1) / threads;
    const int blocks = static_cast<int>(needed_blocks < 65535 ?
        needed_blocks : 65535);
    flat_joseph_gather_kernel<<<blocks, threads, 0, stream>>>(projection_texture,
        d_views_world, d_views_voxel, volume, volume_geometry, views,
        channels, rows, count);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::Bp

// 体素驱动 Joseph BP。兼容入口仍使用 v2 名称。
namespace YK {
    namespace Bp {
        namespace detail {

            //joseph 距离权重的反投影，消除attom_addmicAdd的原子操作，改为每个线程处理一个体素切片（主轴方向）
            template<int ZSIZE>
            __global__ void joseph_bp_v2_kernel(
                cudaTextureObject_t                  sinoTex,
                const SConeProjGeomVec* __restrict__ d_views,
                float* __restrict__                  d_vol,
                SVolGeom                             vg,
                int                                  startAngle,
                int                                  endAngle)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vg.Nx || y >= vg.Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= vg.Nz) return;

                const float fX = vg.origin().x + x * vg.vox_x;
                const float fY = vg.origin().y + y * vg.vox_y;

                const float rcp_vox_x = vg.tmp_rcp_vox_x;
                const float rcp_vox_y = vg.tmp_rcp_vox_y;
                const float rcp_vox_z = vg.tmp_rcp_vox_z;

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int angle = startAngle; angle < endAngle; ++angle)
                {
                    const SConeProjGeomVec& v = d_views[angle];
                    const float ia_rel = (float)(angle - startAngle) + 0.5f;

                    // 探测器法向量（Z 循环外）
                    const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
                    const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
                    const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;

                    const float SDD_plane =
                        (v.detS.x - v.src.x) * nx +
                        (v.detS.y - v.src.y) * ny +
                        (v.detS.z - v.src.z) * nz;

                    const float rcp_U2 = __frcp_rn(
                        v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z);
                    const float rcp_V2 = __frcp_rn(
                        v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z);

                    // 射线 XY 分量（Z 循环外）
                    const float dx = fX - v.src.x;
                    const float dy = fY - v.src.y;

                    // 体素坐标系下的 XY 分量（Z 循环外）
                    const float dx_vox = dx * rcp_vox_x;
                    const float dy_vox = dy * rcp_vox_y;
                    const float absDX_vox = fabsf(dx_vox);
                    const float absDY_vox = fabsf(dy_vox);

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz)
                    {
                        const int zIdx = startZ + iz;
                        if (zIdx >= vg.Nz) continue;

                        const float worldZ = vg.origin().z + zIdx * vg.vox_z;
                        const float dz = worldZ - v.src.z;

                        // 射线与探测器平面的交点
                        const float denom_n = dx * nx + dy * ny + dz * nz;
                        if (fabsf(denom_n) < 1e-8f) continue;

                        const float t = __fdividef(SDD_plane, denom_n);
                        if (t <= 0.f) continue;

                        const float px = v.src.x + t * dx - v.detS.x;
                        const float py = v.src.y + t * dy - v.detS.y;
                        const float pz = v.src.z + t * dz - v.detS.z;

                        const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) * rcp_U2;
                        const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) * rcp_V2;

                        // 体素坐标系下的 Z 分量
                        const float dz_vox = dz * rcp_vox_z;
                        const float absDZ_vox = fabsf(dz_vox);


                        // ── 主轴判断 + a1/a2 + 物理弦长（一体，替换原来三段）──────────
                        float v0, v1, v2, a1, a2;

                        if (absDX_vox >= absDY_vox && absDX_vox >= absDZ_vox) {
                            // 主轴 X：a1↔Y, a2↔Z
                            v0 = vg.vox_x; v1 = vg.vox_y; v2 = vg.vox_z;
                            a1 = dy_vox / dx_vox;
                            a2 = dz_vox / dx_vox;
                        }
                        else if (absDY_vox >= absDZ_vox) {
                            // 主轴 Y：a1↔X, a2↔Z
                            v0 = vg.vox_y; v1 = vg.vox_x; v2 = vg.vox_z;
                            a1 = dx_vox / dy_vox;
                            a2 = dz_vox / dy_vox;
                        }
                        else {
                            // 主轴 Z：a1↔X, a2↔Y
                            v0 = vg.vox_z; v1 = vg.vox_x; v2 = vg.vox_y;
                            a1 = dx_vox / dz_vox;
                            a2 = dy_vox / dz_vox;
                        }

                        // 物理弦长：各方向用各自的物理体素尺寸
                        // 各向同性时 v0=v1=v2=vox → fPhysLen = vox·sqrt(a1²+a2²+1) = 原值 ✓
                        const float a1v1 = a1 * v1;
                        const float a2v2 = a2 * v2;
                        const float fPhysLen = __fsqrt_rn(v0 * v0 + a1v1 * a1v1 + a2v2 * a2v2);

                        const float p = tex3D<float>(sinoTex, fu + 0.5f, fv + 0.5f, ia_rel);
                        Z[iz] += p * fPhysLen;
                    }
                }

#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) continue;
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                        + (size_t)y * vg.Nx
                        + (size_t)x;
                    d_vol[idx] += Z[iz];
                }
            }




            template<int ZSIZE>
            static void joseph_bp_v2_launch_impl(
                cudaTextureObject_t     sinoTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_vol,
                const SVolGeom& vg,
                int startAngle, int endAngle,
                cudaStream_t stream)
            {
                if (endAngle <= startAngle) return;

                const dim3 block(16, 16, 1);
                const dim3 grid(
                    (vg.Nx + block.x - 1) / block.x,
                    (vg.Ny + block.y - 1) / block.y,
                    (vg.Nz + ZSIZE - 1) / ZSIZE);

                joseph_bp_v2_kernel<ZSIZE> << <grid, block, 0, stream >> > (
                    sinoTex, d_views_vox, d_vol, vg,
                    startAngle, endAngle);
            }

        } // namespace detail

        // ── 对外接口 ─────────────────────────────────────────────────────────────────

        void joseph_bp_v2_launch(
            cudaTextureObject_t                  sinoTex,
            const SConeProjGeomVec* d_views_vox,
            float* d_vol,
            const SVolGeom& vg,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream
        )
        {
            CudaOp::clearIfOverwrite(d_vol,
                static_cast<size_t>(vg.Nx) * vg.Ny * vg.Nz,
                CudaOp::writeMode(accumulate), stream);
            detail::joseph_bp_v2_launch_impl<4>(
                sinoTex, d_views_vox, d_vol, vg,
                0, Na, stream);
            YK_CUDA_KERNEL_CHECK();
        }

    } // namespace Bp
} // namespace YK



// 放射系数版
namespace YK {
    namespace Bp {
        namespace detail {

            template<int ZSIZE>
            __global__ void joseph_bp_v3_kernel(
                cudaTextureObject_t                  sinoTex,
                const SConeProjGeomVec* __restrict__ d_views,
                const FdkAffineCoeff* __restrict__ d_coeffs,
                float* __restrict__                  d_vol,
                SVolGeom                             vg,
                int                                  startAngle,
                int                                  endAngle)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vg.Nx || y >= vg.Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= vg.Nz) return;

                const float fX = vg.origin().x + x * vg.vox_x;
                const float fY = vg.origin().y + y * vg.vox_y;

                const float rcp_vox_x = vg.tmp_rcp_vox_x;
                const float rcp_vox_y = vg.tmp_rcp_vox_y;
                const float rcp_vox_z = vg.tmp_rcp_vox_z;

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int angle = startAngle; angle < endAngle; ++angle)
                {
                    const SConeProjGeomVec& v = d_views[angle];
                    const FdkAffineCoeff& c = d_coeffs[angle];
                    const float ia_rel = (float)(angle - startAngle) + 0.5f;

                    // Z循环外：XY部分预计算
                    const float denXY = c.Cd_w + c.Cd_x * fX + c.Cd_y * fY;
                    const float uNumXY = c.Cu_w + c.Cu_x * fX + c.Cu_y * fY;
                    const float vNumXY = c.Cv_w + c.Cv_x * fX + c.Cv_y * fY;

                    // Z方向步长
                    const float denStep = c.Cd_z * vg.vox_z;
                    const float uNumStep = c.Cu_z * vg.vox_z;
                    const float vNumStep = c.Cv_z * vg.vox_z;

                    // 初始值退一步，循环内先加再用
                    const float fZ0 = vg.origin().z + (startZ - 1) * vg.vox_z;
                    float den = denXY + c.Cd_z * fZ0;
                    float uNum = uNumXY + c.Cu_z * fZ0;
                    float vNum = vNumXY + c.Cv_z * fZ0;

                    // 主轴XY分量（Z循环外）
                    const float dx_vox = (fX - v.src.x) * rcp_vox_x;
                    const float dy_vox = (fY - v.src.y) * rcp_vox_y;
                    const float absDX_vox = fabsf(dx_vox);
                    const float absDY_vox = fabsf(dy_vox);

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz)
                    {
                        den += denStep;
                        uNum += uNumStep;
                        vNum += vNumStep;

                        const int zIdx = startZ + iz;
                        if (zIdx >= vg.Nz) continue;
                        if (den < 1e-8f) continue;

                        const float fr = __frcp_rn(den);
                        const float fu = uNum * fr;
                        const float fv = vNum * fr;

                        const float worldZ = vg.origin().z + zIdx * vg.vox_z;
                        const float dz_vox = (worldZ - v.src.z) * rcp_vox_z;
                        const float absDZ_vox = fabsf(dz_vox);

                        // ── 主轴判断 + 物理体素尺寸 + a1/a2 ──────────────────────
                        float v0, v1, v2, a1, a2;
                        if (absDX_vox >= absDY_vox && absDX_vox >= absDZ_vox) {
                            v0 = vg.vox_x; v1 = vg.vox_y; v2 = vg.vox_z;
                            a1 = dy_vox / dx_vox;
                            a2 = dz_vox / dx_vox;
                        }
                        else if (absDY_vox >= absDZ_vox) {
                            v0 = vg.vox_y; v1 = vg.vox_x; v2 = vg.vox_z;
                            a1 = dx_vox / dy_vox;
                            a2 = dz_vox / dy_vox;
                        }
                        else {
                            v0 = vg.vox_z; v1 = vg.vox_x; v2 = vg.vox_y;
                            a1 = dx_vox / dz_vox;
                            a2 = dy_vox / dz_vox;
                        }

                        // ── 物理弦长（各向异性精确）──────────────────────────────
                        // fPhysLen = sqrt(v0² + (a1·v1)² + (a2·v2)²)
                        // 各向同性时 v0=v1=v2=vox：fPhysLen = vox·sqrt(a1²+a2²+1)
                        //   = fMainAxisVox·fDistCorr  ← 与原来完全等价 ✓
                        const float a1v1 = a1 * v1;
                        const float a2v2 = a2 * v2;
                        const float fPhysLen = __fsqrt_rn(v0 * v0 + a1v1 * a1v1 + a2v2 * a2v2);

                        const float p = tex3D<float>(sinoTex, fu + 0.5f, fv + 0.5f, ia_rel);
                        Z[iz] += p * fPhysLen;   // 原: p * fMainAxisVox * fDistCorr
                    }
                }

#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) continue;
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                        + (size_t)y * vg.Nx
                        + (size_t)x;
                    d_vol[idx] += Z[iz];
                }
            }

            template<int ZSIZE>
            static void joseph_bp_v3_launch_impl(
                cudaTextureObject_t     sinoTex,
                const SConeProjGeomVec* d_views,
                const FdkAffineCoeff* d_coeffs,
                float* d_vol,
                const SVolGeom& vg,
                int startAngle, int endAngle,
                cudaStream_t stream)
            {
                if (endAngle <= startAngle) return;

                const dim3 block(16, 16, 1);
                const dim3 grid(
                    (vg.Nx + block.x - 1) / block.x,
                    (vg.Ny + block.y - 1) / block.y,
                    (vg.Nz + ZSIZE - 1) / ZSIZE);

                joseph_bp_v3_kernel<ZSIZE> << <grid, block, 0, stream >> > (
                    sinoTex, d_views, d_coeffs, d_vol, vg,
                    startAngle, endAngle);
            }

        }; // namespace detail

        void joseph_bp_v3_launch(
            cudaTextureObject_t      sinoTex,
            const SConeProjGeomVec* d_views,
            const FdkAffineCoeff* d_coeffs,
            float* d_vol,
            const SVolGeom& vg,
            int Na,
            bool accumulate,
            cudaStream_t stream)
        {
            CudaOp::clearIfOverwrite(d_vol,
                static_cast<size_t>(vg.Nx) * vg.Ny * vg.Nz,
                CudaOp::writeMode(accumulate), stream);
            detail::joseph_bp_v3_launch_impl<4>(
                sinoTex, d_views, d_coeffs, d_vol, vg,
                0, Na, stream);
            YK_CUDA_KERNEL_CHECK();
        }

    } // namespace Bp
} // namespace YK


