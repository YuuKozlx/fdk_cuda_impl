#include "YkBpJosephLaunch.cuh"
#include "../../global/YkMacro.hpp"
#include "YkBPHelpers.cuh"
#include <BP/YkBPCommon.cuh>

namespace YK {
    namespace Bp {
        namespace detail {

            struct DirX {
                __host__ __device__ static float c0(float x, float y, float z) { return x; }
                __host__ __device__ static float c1(float x, float y, float z) { return y; }
                __host__ __device__ static float c2(float x, float y, float z) { return z; }
                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Nx; }
                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Ny; }
                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Nz; }
                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_x; }
                __device__ static float sampleVol(cudaTextureObject_t tex,
                    float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f0, f1, f2);
                }
                __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
                {
                    return make_int3(s, d1, d2);
                }
            };

            struct DirY {
                __host__ __device__ static float c0(float x, float y, float z) { return y; }
                __host__ __device__ static float c1(float x, float y, float z) { return x; }
                __host__ __device__ static float c2(float x, float y, float z) { return z; }
                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Ny; }
                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Nx; }
                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Nz; }
                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_y; }
                __device__ static float sampleVol(cudaTextureObject_t tex,
                    float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f1, f0, f2);
                }
                __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
                {
                    return make_int3(d1, s, d2);
                }
            };

            struct DirZ {
                __host__ __device__ static float c0(float x, float y, float z) { return z; }
                __host__ __device__ static float c1(float x, float y, float z) { return x; }
                __host__ __device__ static float c2(float x, float y, float z) { return y; }
                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Nz; }
                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Nx; }
                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Ny; }
                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_z; }
                __device__ static float sampleVol(cudaTextureObject_t tex,
                    float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f1, f2, f0);
                }
                __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
                {
                    return make_int3(d1, d2, s);
                }
            };

            // ============================================================
            // joseph_bp_kernel
            // d_views_vox：全局指针，不偏移
            // d_sino ：全局指针，不偏移
            // ============================================================
            template<typename DIR, int kStepDenom, bool UseTex>
            __global__ void joseph_bp_kernel(
                const float* d_sino,
                cudaTextureObject_t     sinoTex,
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

                // d_views_vox 用全局 angle 索引
                const SConeProjGeomVec& v = d_views_vox[angle];
                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
                const float fDetUX = v.detU.x, fDetUY = v.detU.y, fDetUZ = v.detU.z;
                const float fDetVX = v.detV.x, fDetVY = v.detV.y, fDetVZ = v.detV.z;
                const float fDetSX = v.detS.x + 0.5f * fDetUX + 0.5f * fDetVX;
                const float fDetSY = v.detS.y + 0.5f * fDetUY + 0.5f * fDetVY;
                const float fDetSZ = v.detS.z + 0.5f * fDetUZ + 0.5f * fDetVZ;

                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);
                const int nSlicesOrig = nSlices / kStepDenom;
                constexpr float step = 1.f / (float)kStepDenom;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int startV = (blockIdx.x / nUBlocks) * kDetBlockV;
                const int endV = min(startV + kDetBlockV, Nv);
                const int endSlice = min(startSlice + kBlockSlices, nSlices);

                for (int detV = startV; detV < endV; ++detV)
                {
                    const float fDetX = fDetSX + detU * fDetUX + detV * fDetVX;
                    const float fDetY = fDetSY + detU * fDetUY + detV * fDetVY;
                    const float fDetZ = fDetSZ + detU * fDetUZ + detV * fDetVZ;

                    const float den = DIR::c0(fSrcX, fSrcY, fSrcZ)
                        - DIR::c0(fDetX, fDetY, fDetZ);
                    if (fabsf(den) < 1e-8f) continue;

                    const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ)
                        - DIR::c1(fDetX, fDetY, fDetZ)) / den;
                    const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ)
                        - DIR::c2(fDetX, fDetY, fDetZ)) / den;
                    const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ)
                        - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                    const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ)
                        - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                    const float fDistCorr = sqrtf(a1 * a1 + a2 * a2 + 1.f);

                    // d_sino 用全局 angle 索引
                    float proj_val;
                    if constexpr (UseTex) {
                        proj_val = tex3D<float>(sinoTex,
                            detU + 0.5f, detV + 0.5f, angle + 0.5f);
                    }
                    else {
                        proj_val = d_sino[((size_t)angle * Nv + detV) * Nu + detU];
                    }
                    const float contrib_base = proj_val * fDistCorr * step * fMainAxisVox;

                    float f0 = startSlice * step + 0.5f * step;
                    float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                        + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
                    float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                        + b2 + 0.5f * nDim2 - 0.5f + 0.5f;

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
                                atomicAdd(&d_vol[idx], contrib_base * w);
                                };
                            add(i1, i2, (1.f - w1) * (1.f - w2));
                            add(i1 + 1, i2, w1 * (1.f - w2));
                            add(i1, i2 + 1, (1.f - w1) * w2);
                            add(i1 + 1, i2 + 1, w1 * w2);
                        }

                        f0 += step;
                        f1 += a1 * step;
                        f2 += a2 * step;
                    }
                }
            }

            // ── launch_group_impl：d_views_vox 全局指针，startAngle/endAngle ─
            template<typename DIR, int kStepDenom, bool UseTex>
            static void joseph_bp_launch_group_impl(
                const float* d_sino,
                cudaTextureObject_t     sinoTex,
                const SConeProjGeomVec* d_views_vox,   // 全局指针
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

                dim3 block(kDetBlockU, kAnglesPerBlock);
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                const float fVox = DIR::voxSize(g);

                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    joseph_bp_kernel<DIR, kStepDenom, UseTex> << <grid, block, 0, stream >> > (
                        d_sino, sinoTex,
                        d_views_vox,                    // 全局指针
                        d_vol, g, Nu, Nv,
                        s, startAngle, endAngle,    // 全局角度范围
                        fVox);
                }
            }

            // ── dispatch：d_views_vox 全局指针，不偏移 ──────────────────────
            template<bool UseTex>
            static void joseph_bp_dispatch(
                const float* d_sino,
                cudaTextureObject_t                  sinoTex,
                const std::vector<SConeProjGeomVec>& h_views,
                const SConeProjGeomVec* d_views_vox,   // 全局指针
                float* d_vol,
                const SVolGeom& g,
                int Na, int Nu, int Nv,
                cudaStream_t stream,
                BpStepSuperSample ss)
            {
                int i = 0;
                while (i < Na)
                {
                    const MainAxis ax = getMainAxis(h_views[i].src);
                    int j = i + 1;
                    while (j < Na && getMainAxis(h_views[j].src) == ax) ++j;

                    switch (ax) {
                    case MainAxis::X:
                        switch (ss) {
                        case BpStepSuperSample::x1:
                            joseph_bp_launch_group_impl<DirX, 1, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x2:
                            joseph_bp_launch_group_impl<DirX, 2, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x4:
                            joseph_bp_launch_group_impl<DirX, 4, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        }
                        break;
                    case MainAxis::Y:
                        switch (ss) {
                        case BpStepSuperSample::x1:
                            joseph_bp_launch_group_impl<DirY, 1, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x2:
                            joseph_bp_launch_group_impl<DirY, 2, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x4:
                            joseph_bp_launch_group_impl<DirY, 4, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        }
                        break;
                    case MainAxis::Z:
                        switch (ss) {
                        case BpStepSuperSample::x1:
                            joseph_bp_launch_group_impl<DirZ, 1, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x2:
                            joseph_bp_launch_group_impl<DirZ, 2, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        case BpStepSuperSample::x4:
                            joseph_bp_launch_group_impl<DirZ, 4, UseTex>(
                                d_sino, sinoTex, d_views_vox, d_vol, g, Nu, Nv,
                                i, j, stream); break;
                        }
                        break;
                    }
                    i = j;
                }
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
            if (!accumulate) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
                    (size_t)g.Nx * g.Ny * g.Nz * sizeof(float), stream));
            }
            detail::joseph_bp_dispatch<false>(
                d_sino, 0, h_views, d_views_vox, d_vol,
                g, Na, Nu, Nv, stream, ss);
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
            if (!accumulate) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
                    (size_t)g.Nx * g.Ny * g.Nz * sizeof(float), stream));
            }
            detail::joseph_bp_dispatch<true>(
                nullptr, sinoTex, h_views, d_views_vox, d_vol,
                g, Na, Nu, Nv, stream, ss);
        }

    } // namespace Bp
} // namespace YK




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
//            // ============================================================
//            // joseph_bp_kernel — 射线驱动，线程分工与 FP 完全一致
//            // ============================================================
//            template<typename DIR, int kStepDenom, bool UseTex>
//            __global__ void joseph_bp_kernel(
//                const float* d_sino,
//                cudaTextureObject_t     sinoTex,
//                const SConeProjGeomVec* d_views,   // 体素坐标系，与 FP 相同
//                float* d_vol,
//                SVolGeom                g,
//                int Nu, int Nv,
//                int startSlice,
//                int startAngle, int endAngle,
//                float fMainAxisVox)
//            {
//                // 线程分工与 FP 完全一致：(detU, angle)
//                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
//                const int angle = startAngle + localAngle;
//                if (angle >= endAngle) return;
//
//                const SConeProjGeomVec& v = d_views[localAngle];
//                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
//                const float fDetUX = v.detU.x, fDetUY = v.detU.y, fDetUZ = v.detU.z;
//                const float fDetVX = v.detV.x, fDetVY = v.detV.y, fDetVZ = v.detV.z;
//                const float fDetSX = v.detS.x + 0.5f * fDetUX + 0.5f * fDetVX;
//                const float fDetSY = v.detS.y + 0.5f * fDetUY + 0.5f * fDetVY;
//                const float fDetSZ = v.detS.z + 0.5f * fDetUZ + 0.5f * fDetVZ;
//
//                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
//                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
//                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);
//                const int nSlicesOrig = nSlices / kStepDenom;
//                constexpr float step = 1.f / (float)kStepDenom;
//
//                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
//                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
//                if (detU >= Nu) return;
//
//                const int startV = (blockIdx.x / nUBlocks) * kDetBlockV;
//                const int endV = min(startV + kDetBlockV, Nv);
//                const int endSlice = min(startSlice + kBlockSlices, nSlices);
//
//                for (int detV = startV; detV < endV; ++detV)
//                {
//                    const float fDetX = fDetSX + detU * fDetUX + detV * fDetVX;
//                    const float fDetY = fDetSY + detU * fDetUY + detV * fDetVY;
//                    const float fDetZ = fDetSZ + detU * fDetUZ + detV * fDetVZ;
//
//                    // 与 FP 完全相同的斜率公式
//                    const float den = DIR::c0(fSrcX, fSrcY, fSrcZ)
//                        - DIR::c0(fDetX, fDetY, fDetZ);
//                    if (fabsf(den) < 1e-8f) continue;
//
//                    const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ)
//                        - DIR::c1(fDetX, fDetY, fDetZ)) / den;
//                    const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ)
//                        - DIR::c2(fDetX, fDetY, fDetZ)) / den;
//                    const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ)
//                        - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
//                    const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ)
//                        - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);
//
//                    const float fDistCorr = sqrtf(a1 * a1 + a2 * a2 + 1.f);
//
//                    // 读正弦图（循环前一次读，与 FP 的写对称）
//                    float proj_val;
//                    if constexpr (UseTex) {
//                        proj_val = tex3D<float>(sinoTex,
//                            detU + 0.5f, detV + 0.5f, angle + 0.5f);
//                    }
//                    else {
//                        proj_val = d_sino[((size_t)angle * Nv + detV) * Nu + detU];
//                    }
//                    const float contrib_base = proj_val * fDistCorr * step * fMainAxisVox;
//
//                    // 与 FP 完全相同的步进起点
//                    float f0 = startSlice * step + 0.5f * step;
//                    float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
//                        + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
//                    float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
//                        + b2 + 0.5f * nDim2 - 0.5f + 0.5f;
//
//                    for (int s = startSlice; s < endSlice; ++s)
//                    {
//                        // FP: fVal += tex3D(volTex, f0, f1, f2)  自动双线性
//                        // BP: 手动双线性，atomicAdd 到周围 4 体素
//                        const int i0 = (int)floorf(f0 - 0.5f);
//                        const int i1 = (int)floorf(f1 - 0.5f);
//                        const int i2 = (int)floorf(f2 - 0.5f);
//
//                        const float w1 = (f1 - 0.5f) - i1;
//                        const float w2 = (f2 - 0.5f) - i2;
//                        if (i0 >= 0 && i0 < nSlicesOrig) {
//                            auto add = [&](int j1, int j2, float w) {
//                                if (j1 < 0 || j1 >= nDim1 || j2 < 0 || j2 >= nDim2) return;
//                                const int3 vox = DIR::toVoxel(i0, j1, j2);
//                                // ── 越界保护在 lambda 内部 ──────────────────────
//                                if (vox.x < 0 || vox.x >= g.Nx) return;
//                                if (vox.y < 0 || vox.y >= g.Ny) return;
//                                if (vox.z < 0 || vox.z >= g.Nz) return;
//                                // ────────────────────────────────────────────────
//                                const size_t idx = (size_t)vox.z * g.Ny * g.Nx
//                                    + (size_t)vox.y * g.Nx
//                                    + (size_t)vox.x;
//                                atomicAdd(&d_vol[idx], contrib_base * w);
//                                };
//                            add(i1, i2, (1.f - w1) * (1.f - w2));
//                            add(i1 + 1, i2, w1 * (1.f - w2));
//                            add(i1, i2 + 1, (1.f - w1) * w2);
//                            add(i1 + 1, i2 + 1, w1 * w2);
//                        }
//
//                        f0 += step;
//                        f1 += a1 * step;
//                        f2 += a2 * step;
//                    }
//                }
//            }
//
//            template<typename DIR, int kStepDenom, bool UseTex>
//            static void joseph_bp_launch_group_impl(
//                const float* d_sino,
//                cudaTextureObject_t     sinoTex,
//                const SConeProjGeomVec* d_views,
//                float* d_vol,
//                const SVolGeom& g,
//                int Na, int Nu, int Nv,
//                int angleOffset,
//                cudaStream_t stream)
//            {
//                if (Na <= 0) return;
//
//                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
//                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
//                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);
//
//                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
//                const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
//                const int nABlocks = (Na + kAnglesPerBlock - 1) / kAnglesPerBlock;
//
//                dim3 block(kDetBlockU, kAnglesPerBlock);
//                dim3 grid(nUBlocks * nVBlocks, nABlocks);
//
//                const float fVox = DIR::voxSize(g);
//
//                for (int s = 0; s < nSlices; s += kBlockSlices)
//                {
//                    YK_CUDA_CHECK(cudaGetLastError());
//
//                    if (s == 0) {
//                        YK_LOGD("[joseph_bp] g=%dx%dx%d Na=%d Nu=%d Nv=%d angleOffset=%d fVox=%f",
//                            g.Nx, g.Ny, g.Nz, Na, Nu, Nv, angleOffset, fVox);
//                        YK_LOGD("[joseph_bp] grid=(%d,%d) block=(%d,%d)",
//                            grid.x, grid.y, block.x, block.y);
//                    }
//
//                    joseph_bp_kernel<DIR, kStepDenom, UseTex> << <grid, block, 0, stream >> > (
//                        d_sino, sinoTex, d_views, d_vol,
//                        g, Nu, Nv,
//                        s, angleOffset, angleOffset + Na,
//                        fVox);
//
//                    YK_CUDA_CHECK(cudaGetLastError());
//                }
//            }
//
//            // ── dispatch：去掉 accumulate，调用方负责清零 ────────────
//            template<bool UseTex>
//            static void joseph_bp_dispatch(
//                const float* d_sino,
//                cudaTextureObject_t                  sinoTex,
//                const std::vector<SConeProjGeomVec>& h_views,
//                const SConeProjGeomVec* d_views,
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
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        case BpStepSuperSample::x2:
//                            joseph_bp_launch_group_impl<DirX, 2, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        case BpStepSuperSample::x4:
//                            joseph_bp_launch_group_impl<DirX, 4, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        }
//                        break;
//                    case MainAxis::Y:
//                        switch (ss) {
//                        case BpStepSuperSample::x1:
//                            joseph_bp_launch_group_impl<DirY, 1, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        case BpStepSuperSample::x2:
//                            joseph_bp_launch_group_impl<DirY, 2, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        case BpStepSuperSample::x4:
//                            joseph_bp_launch_group_impl<DirY, 4, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        }
//                        break;
//                    case MainAxis::Z:
//                        switch (ss) {
//                        case BpStepSuperSample::x1:
//                            joseph_bp_launch_group_impl<DirZ, 1, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        case BpStepSuperSample::x2:
//                            joseph_bp_launch_group_impl<DirZ, 2, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        case BpStepSuperSample::x4:
//                            joseph_bp_launch_group_impl<DirZ, 4, UseTex>(
//                                d_sino, sinoTex, d_views + i, d_vol, g,
//                                j - i, Nu, Nv, i, stream); break;
//                        }
//                        break;
//                    }
//                    i = j;
//                }
//            }
//
//        } // namespace detail
//
//        // ── 对外接口：accumulate 在外部处理清零 ─────────────────────
//        void joseph_bp_launch(
//            const float* d_sino,
//            const std::vector<SConeProjGeomVec>& h_views,
//            const SConeProjGeomVec* d_views,
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
//                d_sino, 0, h_views, d_views, d_vol,
//                g, Na, Nu, Nv, stream, ss);
//        }
//
//        void joseph_bp_launch(
//            cudaTextureObject_t                  sinoTex,
//            const std::vector<SConeProjGeomVec>& h_views,
//            const SConeProjGeomVec* d_views,
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
//                nullptr, sinoTex, h_views, d_views, d_vol,
//                g, Na, Nu, Nv, stream, ss);
//        }
//
//    } // namespace Bp
//} // namespace YK