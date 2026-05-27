//#include "YkBpJosephLaunch.cuh"
//#include "../../global/YkMacro.hpp"
//#include "YkBPHelpers.cuh"
//#include <BP/YkBPCommon.cuh>
//
//namespace YK {
//    namespace Bp {
//        namespace detail {
//
//            struct DirX {
//                __host__ __device__ static float c0(float x, float y, float z) { return x; }
//                __host__ __device__ static float c1(float x, float y, float z) { return y; }
//                __host__ __device__ static float c2(float x, float y, float z) { return z; }
//                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Nx; }
//                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Ny; }
//                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Nz; }
//                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_x; }
//                __device__ static float sampleVol(cudaTextureObject_t tex,
//                    float f0, float f1, float f2)
//                {
//                    return tex3D<float>(tex, f0, f1, f2);
//                }
//                __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
//                {
//                    return make_int3(s, d1, d2);
//                }
//            };
//
//            struct DirY {
//                __host__ __device__ static float c0(float x, float y, float z) { return y; }
//                __host__ __device__ static float c1(float x, float y, float z) { return x; }
//                __host__ __device__ static float c2(float x, float y, float z) { return z; }
//                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Ny; }
//                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Nx; }
//                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Nz; }
//                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_y; }
//                __device__ static float sampleVol(cudaTextureObject_t tex,
//                    float f0, float f1, float f2)
//                {
//                    return tex3D<float>(tex, f1, f0, f2);
//                }
//                __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
//                {
//                    return make_int3(d1, s, d2);
//                }
//            };
//
//            struct DirZ {
//                __host__ __device__ static float c0(float x, float y, float z) { return z; }
//                __host__ __device__ static float c1(float x, float y, float z) { return x; }
//                __host__ __device__ static float c2(float x, float y, float z) { return y; }
//                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Nz; }
//                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Nx; }
//                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Ny; }
//                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_z; }
//                __device__ static float sampleVol(cudaTextureObject_t tex,
//                    float f0, float f1, float f2)
//                {
//                    return tex3D<float>(tex, f1, f2, f0);
//                }
//                __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
//                {
//                    return make_int3(d1, d2, s);
//                }
//            };
//
//
//
//            template<typename DIR, int kStepDenom, bool UseTex>
//            __global__ void joseph_bp_kernel(const float* d_sino,
//                cudaTextureObject_t     sinoTex,
//                const SConeProjGeomVec* d_views_vox,   // 全局指针
//                float* d_vol,
//                SVolGeom                g,
//                int Nu, int Nv,
//                int startSlice,
//                int startAngle, int endAngle,      // 全局角度范围
//                float fMainAxisVox)
//            {
//                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
//                const int angle = startAngle + localAngle;
//                if (angle >= endAngle) return;
//
//                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
//                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
//                if (detU >= Nu) return;
//
//                const int detV = (blockIdx.x / nUBlocks) * kDetBlockV + threadIdx.z;  // ← 并行
//                if (detV >= Nv) return;
//
//                const SConeProjGeomVec& v = d_views_vox[angle];
//                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
//                const float fDetUX = v.detU.x, fDetUY = v.detU.y, fDetUZ = v.detU.z;
//                const float fDetVX = v.detV.x, fDetVY = v.detV.y, fDetVZ = v.detV.z;
//                const float fDetSX = v.detS.x;
//                const float fDetSY = v.detS.y;
//                const float fDetSZ = v.detS.z;
//
//                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
//                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
//                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);
//                const int nSlicesOrig = nSlices / kStepDenom;
//                const int endSlice = min(startSlice + kBlockSlices, nSlices);
//                constexpr float step = 1.f / (float)kStepDenom;
//
//                // detV 直接用，不再 for 循环
//                const float fDetX = fDetSX + detU * fDetUX + detV * fDetVX;
//                const float fDetY = fDetSY + detU * fDetUY + detV * fDetVY;
//                const float fDetZ = fDetSZ + detU * fDetUZ + detV * fDetVZ;
//
//                const float den = DIR::c0(fSrcX, fSrcY, fSrcZ)
//                    - DIR::c0(fDetX, fDetY, fDetZ);
//                if (fabsf(den) < 1e-8f) return;
//
//                const float rcp_den = __frcp_rn(den);
//                const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ)
//                    - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_den;
//                const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ)
//                    - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_den;
//                const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ)
//                    - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
//                const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ)
//                    - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);
//
//                const float fDistCorr = __fsqrt_rn(a1 * a1 + a2 * a2 + 1.f);
//
//                float proj_val;
//                if constexpr (UseTex) {
//                    proj_val = tex3D<float>(sinoTex, detU + 0.5f, detV + 0.5f, angle + 0.5f);
//                }
//                else {
//                    proj_val = d_sino[((size_t)angle * Nv + detV) * Nu + detU];
//                }
//                const float contrib_base = proj_val * fDistCorr * step * fMainAxisVox;
//
//                float f0 = startSlice * step + 0.5f * step;
//                float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
//                    + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
//                float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
//                    + b2 + 0.5f * nDim2 - 0.5f + 0.5f;
//
//                const float a1_step = a1 * step;
//                const float a2_step = a2 * step;
//
//                for (int s = startSlice; s < endSlice; ++s)
//                {
//                    const int i0 = (int)floorf(f0 - 0.5f);
//                    const int i1 = (int)floorf(f1 - 0.5f);
//                    const int i2 = (int)floorf(f2 - 0.5f);
//
//                    const float w1 = fmaxf(0.f, fminf(1.f, (f1 - 0.5f) - i1));
//                    const float w2 = fmaxf(0.f, fminf(1.f, (f2 - 0.5f) - i2));
//
//                    if (i0 >= 0 && i0 < nSlicesOrig) {
//                        auto add = [&](int j1, int j2, float w) {
//                            if (j1 < 0 || j1 >= nDim1) return;
//                            if (j2 < 0 || j2 >= nDim2) return;
//                            const int3 vox = DIR::toVoxel(i0, j1, j2);
//                            if (vox.x < 0 || vox.x >= g.Nx) return;
//                            if (vox.y < 0 || vox.y >= g.Ny) return;
//                            if (vox.z < 0 || vox.z >= g.Nz) return;
//                            const size_t idx = (size_t)vox.z * g.Ny * g.Nx
//                                + (size_t)vox.y * g.Nx
//                                + (size_t)vox.x;
//                            atomicAdd(&d_vol[idx], contrib_base * w);
//                            };
//                        add(i1, i2, (1.f - w1) * (1.f - w2));
//                        add(i1 + 1, i2, w1 * (1.f - w2));
//                        add(i1, i2 + 1, (1.f - w1) * w2);
//                        add(i1 + 1, i2 + 1, w1 * w2);
//                    }
//
//                    f0 += step;
//                    f1 += a1_step;
//                    f2 += a2_step;
//                }
//            }
//
//            
//
//            template<typename DIR, int kStepDenom, bool UseTex>
//            static void joseph_bp_launch_group_impl(
//                const float* d_sino,
//                cudaTextureObject_t     sinoTex,
//                const SConeProjGeomVec* d_views_vox,
//                float* d_vol,
//                const SVolGeom& g,
//                int Nu, int Nv,
//                int startAngle, int endAngle,
//                cudaStream_t stream)
//            {
//                const int Na = endAngle - startAngle;
//                if (Na <= 0) return;
//
//                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
//
//                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
//                const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
//                const int nABlocks = (Na + kAnglesPerBlock - 1) / kAnglesPerBlock;
//
//                dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV);
//                dim3 grid(nUBlocks * nVBlocks, nABlocks);
//
//                const float fVox = DIR::voxSize(g);
//
//                for (int s = 0; s < nSlices; s += kBlockSlices)
//                {
//                    joseph_bp_kernel<DIR, kStepDenom, UseTex> << <grid, block, 0, stream >> > (
//                        d_sino, sinoTex,
//                        d_views_vox,
//                        d_vol, g, Nu, Nv,
//                        s, startAngle, endAngle,
//                        fVox);
//                }
//            }
//
//            // ── dispatch：d_views_vox 全局指针，不偏移 ──────────────────────
//            template<bool UseTex>
//            static void joseph_bp_dispatch(
//                const float* d_sino,
//                cudaTextureObject_t                  sinoTex,
//                const std::vector<SConeProjGeomVec>& h_views,
//                const SConeProjGeomVec* d_views_vox,   // 全局指针
//                float* d_vol,
//                const SVolGeom& g,
//                int Na, int Nu, int Nv,
//                cudaStream_t stream,
//                BpStepSuperSample ss)
//            {
//                int i = 0;
//                while (i < Na)
//                {
//                    const MainAxis ax = getMainAxis(h_views[i].src);
//                    int j = i + 1;
//                    while (j < Na && getMainAxis(h_views[j].src) == ax) ++j;
//
//                    switch (ax) {
//                    case MainAxis::X:
//                        switch (ss) {
//                        case BpStepSuperSample::x1:
//                            joseph_bp_launch_group_impl<DirX, 1, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        case BpStepSuperSample::x2:
//                            joseph_bp_launch_group_impl<DirX, 2, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        case BpStepSuperSample::x4:
//                            joseph_bp_launch_group_impl<DirX, 4, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        }
//                        break;
//                    case MainAxis::Y:
//                        switch (ss) {
//                        case BpStepSuperSample::x1:
//                            joseph_bp_launch_group_impl<DirY, 1, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        case BpStepSuperSample::x2:
//                            joseph_bp_launch_group_impl<DirY, 2, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        case BpStepSuperSample::x4:
//                            joseph_bp_launch_group_impl<DirY, 4, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        }
//                        break;
//                    case MainAxis::Z:
//                        switch (ss) {
//                        case BpStepSuperSample::x1:
//                            joseph_bp_launch_group_impl<DirZ, 1, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        case BpStepSuperSample::x2:
//                            joseph_bp_launch_group_impl<DirZ, 2, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        case BpStepSuperSample::x4:
//                            joseph_bp_launch_group_impl<DirZ, 4, UseTex>(
//                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
//                                i, j, stream); break;
//                        }
//                        break;
//                    }
//                    i = j;
//                }
//            }
//
//        } // namespace detail
//
//        void joseph_bp_launch(
//            const float* d_sino,
//            const std::vector<SConeProjGeomVec>& h_views,
//            const SConeProjGeomVec* d_views_vox,
//            float* d_vol,
//            const SVolGeom& g,
//            int Na, int Nu, int Nv,
//            bool accumulate,
//            cudaStream_t stream,
//            BpStepSuperSample ss)
//        {
//            if (!accumulate) {
//                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
//                    (size_t)g.Nx * g.Ny * g.Nz * sizeof(float), stream));
//            }
//            detail::joseph_bp_dispatch<false>(
//                d_sino, 0, h_views, d_views_vox, d_vol,
//                g, Na, Nu, Nv, stream, ss);
//        }
//
//        void joseph_bp_launch(
//            cudaTextureObject_t                  sinoTex,
//            const std::vector<SConeProjGeomVec>& h_views,
//            const SConeProjGeomVec* d_views_vox,
//            float* d_vol,
//            const SVolGeom& g,
//            int Na, int Nu, int Nv,
//            bool accumulate,
//            cudaStream_t stream,
//            BpStepSuperSample ss)
//        {
//            if (!accumulate) {
//                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
//                    (size_t)g.Nx * g.Ny * g.Nz * sizeof(float), stream));
//            }
//            detail::joseph_bp_dispatch<true>(
//                nullptr, sinoTex, h_views, d_views_vox, d_vol,
//                g, Na, Nu, Nv, stream, ss);
//        }
//
//    } // namespace Bp
//} // namespace YK







// //YkBPJosephV2Kernel.cu
//
//#include "YkBPJosephLaunch.cuh"
//#include "../../global/YkMacro.hpp"
//#include "YkBPHelpers.cuh"
//#include <BP/YkBPCommon.cuh>
//
//namespace YK {
//    namespace Bp {
//        namespace detail {
// 
// //joseph 距离权重的反投影，消除attom_addmicAdd的原子操作，改为每个线程处理一个体素切片（主轴方向）
//            template<int ZSIZE>
//            __global__ void joseph_bp_v2_kernel(
//                cudaTextureObject_t                  sinoTex,
//                const SConeProjGeomVec* __restrict__ d_views,
//                float* __restrict__                  d_vol,
//                SVolGeom                             vg,
//                int                                  startAngle,
//                int                                  endAngle)
//            {
//                const int x = blockIdx.x * blockDim.x + threadIdx.x;
//                const int y = blockIdx.y * blockDim.y + threadIdx.y;
//                if (x >= vg.Nx || y >= vg.Ny) return;
//
//                const int startZ = blockIdx.z * ZSIZE;
//                if (startZ >= vg.Nz) return;
//
//                const float fX = vg.origin().x + x * vg.vox_x;
//                const float fY = vg.origin().y + y * vg.vox_y;
//
//                const float rcp_vox_x = __frcp_rn(vg.vox_x);
//                const float rcp_vox_y = __frcp_rn(vg.vox_y);
//                const float rcp_vox_z = __frcp_rn(vg.vox_z);
//
//                float Z[ZSIZE];
//#pragma unroll
//                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;
//
//                for (int angle = startAngle; angle < endAngle; ++angle)
//                {
//                    const SConeProjGeomVec& v = d_views[angle];
//                    const float ia_rel = (float)(angle - startAngle) + 0.5f;
//
//                    // 探测器法向量（Z 循环外）
//                    const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
//                    const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
//                    const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;
//
//                    const float SDD_plane =
//                        (v.detS.x - v.src.x) * nx +
//                        (v.detS.y - v.src.y) * ny +
//                        (v.detS.z - v.src.z) * nz;
//
//                    const float rcp_U2 = __frcp_rn(
//                        v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z);
//                    const float rcp_V2 = __frcp_rn(
//                        v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z);
//
//                    // 射线 XY 分量（Z 循环外）
//                    const float dx = fX - v.src.x;
//                    const float dy = fY - v.src.y;
//
//                    // 体素坐标系下的 XY 分量（Z 循环外）
//                    const float dx_vox = dx * rcp_vox_x;
//                    const float dy_vox = dy * rcp_vox_y;
//                    const float absDX_vox = fabsf(dx_vox);
//                    const float absDY_vox = fabsf(dy_vox);
//                    const float dx2_dy2 = dx * dx + dy * dy;
//
//#pragma unroll
//                    for (int iz = 0; iz < ZSIZE; ++iz)
//                    {
//                        const int zIdx = startZ + iz;
//                        if (zIdx >= vg.Nz) continue;
//
//                        const float worldZ = vg.origin().z + zIdx * vg.vox_z;
//                        const float dz = worldZ - v.src.z;
//
//                        // 射线与探测器平面的交点
//                        const float denom_n = dx * nx + dy * ny + dz * nz;
//                        if (fabsf(denom_n) < 1e-8f) continue;
//
//                        const float t = __fdividef(SDD_plane, denom_n);
//                        if (t <= 0.f) continue;
//
//                        const float px = v.src.x + t * dx - v.detS.x;
//                        const float py = v.src.y + t * dy - v.detS.y;
//                        const float pz = v.src.z + t * dz - v.detS.z;
//
//                        const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) * rcp_U2;
//                        const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) * rcp_V2;
//
//                        // 体素坐标系下的 Z 分量
//                        const float dz_vox = dz * rcp_vox_z;
//                        const float absDZ_vox = fabsf(dz_vox);
//
//                        // 主轴判断（体素坐标系下）
//                        float main_comp_vox, fMainAxisVox;
//                        if (absDX_vox >= absDY_vox && absDX_vox >= absDZ_vox) {
//                            main_comp_vox = absDX_vox;
//                            fMainAxisVox = vg.vox_x;
//                        }
//                        else if (absDY_vox >= absDZ_vox) {
//                            main_comp_vox = absDY_vox;
//                            fMainAxisVox = vg.vox_y;
//                        }
//                        else {
//                            main_comp_vox = absDZ_vox;
//                            fMainAxisVox = vg.vox_z;
//                        }
//
//                        // a1/a2（体素坐标系下的斜率，和 Joseph FP 完全对称）
//                        const float a1 = (absDX_vox >= absDY_vox && absDX_vox >= absDZ_vox)
//                            ? dy_vox / dx_vox
//                            : (absDY_vox >= absDZ_vox)
//                            ? dx_vox / dy_vox
//                            : dx_vox / dz_vox;
//                        const float a2 = (absDX_vox >= absDY_vox && absDX_vox >= absDZ_vox)
//                            ? dz_vox / dx_vox
//                            : (absDY_vox >= absDZ_vox)
//                            ? dz_vox / dy_vox
//                            : dy_vox / dz_vox;
//
//                        const float fDistCorr = __fsqrt_rn(a1 * a1 + a2 * a2 + 1.f);
//
//                        const float p = tex3D<float>(sinoTex, fu + 0.5f, fv + 0.5f, ia_rel);
//                        Z[iz] += p * fDistCorr * fMainAxisVox;
//                    }
//                }
//
//#pragma unroll
//                for (int iz = 0; iz < ZSIZE; ++iz)
//                {
//                    const int zIdx = startZ + iz;
//                    if (zIdx >= vg.Nz) continue;
//                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
//                        + (size_t)y * vg.Nx
//                        + (size_t)x;
//                    d_vol[idx] += Z[iz];
//                }
//            }
//
//
//            template<int ZSIZE>
//            static void joseph_bp_v2_launch_impl(
//                cudaTextureObject_t     sinoTex,
//                const SConeProjGeomVec* d_views_vox,
//                float* d_vol,
//                const SVolGeom& vg,
//                int startAngle, int endAngle,
//                cudaStream_t stream)
//            {
//                if (endAngle <= startAngle) return;
//
//                const dim3 block(16, 16, 1);
//                const dim3 grid(
//                    (vg.Nx + block.x - 1) / block.x,
//                    (vg.Ny + block.y - 1) / block.y,
//                    (vg.Nz + ZSIZE - 1) / ZSIZE);
//
//                joseph_bp_v2_kernel<ZSIZE> << <grid, block, 0, stream >> > (
//                    sinoTex, d_views_vox, d_vol, vg,
//                    startAngle, endAngle);
//            }
//
//        } // namespace detail
//
//        // ── 对外接口 ─────────────────────────────────────────────────────────────────
//
//        void joseph_bp_v2_launch(
//            cudaTextureObject_t                  sinoTex,
//            const std::vector<SConeProjGeomVec>& h_views,
//            const SConeProjGeomVec* d_views_vox,
//            float* d_vol,
//            const SVolGeom& vg,
//            int Na, int Nu, int Nv,
//            bool accumulate,
//            cudaStream_t stream
//           )
//        {
//            if (!accumulate) {
//                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
//                    (size_t)vg.Nx * vg.Ny * vg.Nz * sizeof(float), stream));
//            }
//            detail::joseph_bp_v2_launch_impl<4>(
//                sinoTex, d_views_vox, d_vol, vg,
//                0, Na, stream);
//        }
//
//    } // namespace Bp
//} // namespace YK



// YkBpJosephV2Kernel.cu
#include "YkBpJosephLaunch.cuh"
#include "../../global/YkMacro.hpp"
#include "YkBPHelpers.cuh"
#include <BP/YkBPCommon.cuh>

namespace YK {
    namespace Bp {
        namespace detail {
// fdk式的反投影，没有路径权重
//            template<int ZSIZE, int PROJ_PER_KERNEL>
//            __global__ void joseph_bp_v2_kernel(
//                cudaTextureObject_t                  sinoTex,
//                const SConeProjGeomVec* __restrict__ d_views,
//                float* __restrict__                  d_vol,
//                SVolGeom                             vg,
//                int                                  startAngle,
//                int                                  endAngle)
//            {
//                const int x = blockIdx.x * blockDim.x + threadIdx.x;
//                const int y = blockIdx.y * blockDim.y + threadIdx.y;
//                if (x >= vg.Nx || y >= vg.Ny) return;
//
//                const int startZ = blockIdx.z * ZSIZE;
//                if (startZ >= vg.Nz) return;
//
//                const float fX = vg.origin().x + x * vg.vox_x;
//                const float fY = vg.origin().y + y * vg.vox_y;
//
//                // 先读体素到寄存器
//                float voxelColumn[ZSIZE];
//#pragma unroll
//                for (int iz = 0; iz < ZSIZE; ++iz)
//                {
//                    const int zIdx = startZ + iz;
//                    if (zIdx >= vg.Nz) {
//                        voxelColumn[iz] = 0.f;
//                        continue;
//                    }
//                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
//                        + (size_t)y * vg.Nx
//                        + (size_t)x;
//                    voxelColumn[iz] = d_vol[idx];
//                }
//
//                const int Na = endAngle - startAngle;
//                const int nBatches = (Na + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;
//
//                for (int batch = 0; batch < nBatches; ++batch)
//                {
//                    const int batchStart = startAngle + batch * PROJ_PER_KERNEL;
//                    const int batchEnd = min(batchStart + PROJ_PER_KERNEL, endAngle);
//
//                    for (int angle = batchStart; angle < batchEnd; ++angle)
//                    {
//                        const SConeProjGeomVec& v = d_views[angle];
//                        const float ia_tex = (float)(angle - startAngle) + 0.5f;
//
//                        // 探测器法向量（角度循环外）
//                        const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
//                        const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
//                        const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;
//
//                        const float SDD_plane =
//                            (v.detS.x - v.src.x) * nx +
//                            (v.detS.y - v.src.y) * ny +
//                            (v.detS.z - v.src.z) * nz;
//
//                        const float rcp_U2 = __frcp_rn(
//                            v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z);
//                        const float rcp_V2 = __frcp_rn(
//                            v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z);
//
//                        // XY 射线分量（角度循环外）
//                        const float dx = fX - v.src.x;
//                        const float dy = fY - v.src.y;
//
//#pragma unroll
//                        for (int iz = 0; iz < ZSIZE; ++iz)
//                        {
//                            const int zIdx = startZ + iz;
//                            if (zIdx >= vg.Nz) continue;
//
//                            const float worldZ = vg.origin().z + zIdx * vg.vox_z;
//                            const float dz = worldZ - v.src.z;
//
//                            // 体素中心投影到探测器
//                            const float denom_n = dx * nx + dy * ny + dz * nz;
//                            if (fabsf(denom_n) < 1e-8f) continue;
//
//                            const float t = __fdividef(SDD_plane, denom_n);
//                            if (t <= 0.f) continue;
//
//                            const float px = v.src.x + t * dx - v.detS.x;
//                            const float py = v.src.y + t * dy - v.detS.y;
//                            const float pz = v.src.z + t * dz - v.detS.z;
//
//                            const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) * rcp_U2;
//                            const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) * rcp_V2;
//
//                            // 读正弦图，直接累加（和 TIGRE 一致，无路径长度权重）
//                            voxelColumn[iz] += tex3D<float>(sinoTex, fu + 0.5f, fv + 0.5f, ia_tex);
//                        }
//                    }
//                }
//
//                // 写回
//#pragma unroll
//                for (int iz = 0; iz < ZSIZE; ++iz)
//                {
//                    const int zIdx = startZ + iz;
//                    if (zIdx >= vg.Nz) continue;
//                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
//                        + (size_t)y * vg.Nx
//                        + (size_t)x;
//                    d_vol[idx] = voxelColumn[iz];
//                }
//            }

// fdk式的反投影，带fdk权重
            template<int ZSIZE, int PROJ_PER_KERNEL>
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

                // 先读体素到寄存器
                float voxelColumn[ZSIZE];
            #pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) { voxelColumn[iz] = 0.f; continue; }
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                                     + (size_t)y    * vg.Nx
                                     + (size_t)x;
                    voxelColumn[iz] = d_vol[idx];
                }

                const int Na       = endAngle - startAngle;
                const int nBatches = (Na + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;

                for (int batch = 0; batch < nBatches; ++batch)
                {
                    const int batchStart = startAngle + batch * PROJ_PER_KERNEL;
                    const int batchEnd   = min(batchStart + PROJ_PER_KERNEL, endAngle);

                    for (int angle = batchStart; angle < batchEnd; ++angle)
                    {
                        const SConeProjGeomVec& v = d_views[angle];
                        const float ia_tex = (float)(angle - startAngle) + 0.5f;

                        // 探测器法向量（角度循环外）
                        const float nx = v.detU.y*v.detV.z - v.detU.z*v.detV.y;
                        const float ny = v.detU.z*v.detV.x - v.detU.x*v.detV.z;
                        const float nz = v.detU.x*v.detV.y - v.detU.y*v.detV.x;

                        const float SDD_plane =
                            (v.detS.x - v.src.x)*nx +
                            (v.detS.y - v.src.y)*ny +
                            (v.detS.z - v.src.z)*nz;

                        const float rcp_U2 = __frcp_rn(
                            v.detU.x*v.detU.x + v.detU.y*v.detU.y + v.detU.z*v.detU.z);
                        const float rcp_V2 = __frcp_rn(
                            v.detV.x*v.detV.x + v.detV.y*v.detV.y + v.detV.z*v.detV.z);

                        // FDK 权重参数（角度循环外）
                        const float src_x = v.src.x, src_y = v.src.y, src_z = v.src.z;
                        const float SOD  = __fsqrt_rn(src_x*src_x + src_y*src_y + src_z*src_z);
                        const float SOD2 = SOD * SOD;
                        const float rcp_SOD = __frcp_rn(SOD);
                        // 主射线方向：源指向等中心（即 -src 归一化）
                        const float cr_x = -src_x * rcp_SOD;
                        const float cr_y = -src_y * rcp_SOD;
                        const float cr_z = -src_z * rcp_SOD;

                        // XY 射线分量（角度循环外）
                        const float dx = fX - src_x;
                        const float dy = fY - src_y;

            #pragma unroll
                        for (int iz = 0; iz < ZSIZE; ++iz)
                        {
                            const int zIdx = startZ + iz;
                            if (zIdx >= vg.Nz) continue;

                            const float worldZ = vg.origin().z + zIdx * vg.vox_z;
                            const float dz     = worldZ - src_z;

                            // 体素中心投影到探测器
                            const float denom_n = dx*nx + dy*ny + dz*nz;
                            if (fabsf(denom_n) < 1e-8f) continue;

                            const float t = __fdividef(SDD_plane, denom_n);
                            if (t <= 0.f) continue;

                            const float px = src_x + t*dx - v.detS.x;
                            const float py = src_y + t*dy - v.detS.y;
                            const float pz = src_z + t*dz - v.detS.z;

                            const float fu = (px*v.detU.x + py*v.detU.y + pz*v.detU.z) * rcp_U2;
                            const float fv = (px*v.detV.x + py*v.detV.y + pz*v.detV.z) * rcp_V2;

                            // FDK 权重：SOD^2 / denom_c^2
                            const float denom_c = dx*cr_x + dy*cr_y + dz*cr_z;
                            if (fabsf(denom_c) < 1e-8f) continue;
                            const float w = SOD2 / (denom_c * denom_c);

                            voxelColumn[iz] += tex3D<float>(sinoTex,
                                fu + 0.5f, fv + 0.5f, ia_tex) * w;
                        }
                    }
                }

                // 写回
            #pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) continue;
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                                     + (size_t)y    * vg.Nx
                                     + (size_t)x;
                    d_vol[idx] = voxelColumn[iz];
                }
            }


            template<int ZSIZE, int PROJ_PER_KERNEL>
            static void joseph_bp_v2_launch_impl(
                cudaTextureObject_t     sinoTex,
                const SConeProjGeomVec* d_views,
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

                joseph_bp_v2_kernel<ZSIZE, PROJ_PER_KERNEL> << <grid, block, 0, stream >> > (
                    sinoTex, d_views, d_vol, vg,
                    startAngle, endAngle);
            }

        } // namespace detail

        void joseph_bp_v2_launch(
            cudaTextureObject_t     sinoTex,
            const SConeProjGeomVec* d_views_world,
            float* d_vol,
            const SVolGeom& vg,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream)
        {
            if (!accumulate) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
                    (size_t)vg.Nx * vg.Ny * vg.Nz * sizeof(float), stream));
            }

            detail::joseph_bp_v2_launch_impl<4, 32>(
                sinoTex, d_views_world, d_vol, vg,
                0, Na, stream);
        }

    } // namespace Bp
} // namespace YK