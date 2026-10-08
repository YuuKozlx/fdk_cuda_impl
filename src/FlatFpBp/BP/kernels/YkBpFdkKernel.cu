// YkBpFdkKernel.cu
#include "YkFlatFdkBpLaunch.cuh"
#include "common/cuda/operators/YkOperatorKernelTypes.cuh"
#include "global/YkMacro.hpp"
#include "YkBPHelpers.cuh"
#include <FlatFpBp/BP/YkBPCommon.cuh>

namespace YK {
    namespace Bp {
        namespace detail {

            template<int ZSIZE, int PROJ_PER_KERNEL>
            __global__ void fdk_bp_kernel_no_weight(
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

                // 先读体素到寄存器
                float voxelColumn[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) { voxelColumn[iz] = 0.f; continue; }
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                        + (size_t)y * vg.Nx + (size_t)x;
                    voxelColumn[iz] = d_vol[idx];
                }

                const int Na = endAngle - startAngle;
                const int nBatches = (Na + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;

                for (int batch = 0; batch < nBatches; ++batch)
                {
                    const int batchStart = startAngle + batch * PROJ_PER_KERNEL;
                    const int batchEnd = min(batchStart + PROJ_PER_KERNEL, endAngle);

                    for (int angle = batchStart; angle < batchEnd; ++angle)
                    {
                        const FdkAffineCoeff& c = d_coeffs[angle];
                        const float ia_tex = (float)(angle - startAngle) + 0.5f;

                        // Z循环外：XY部分
                        const float denXY = c.Cd_w + c.Cd_x * fX + c.Cd_y * fY;
                        const float uNumXY = c.Cu_w + c.Cu_x * fX + c.Cu_y * fY;
                        const float vNumXY = c.Cv_w + c.Cv_x * fX + c.Cv_y * fY;

                        const float denStep = c.Cd_z * vg.vox_z;
                        const float uNumStep = c.Cu_z * vg.vox_z;
                        const float vNumStep = c.Cv_z * vg.vox_z;

                        const float fZ0 = vg.origin().z + (startZ - 1) * vg.vox_z;
                        float den = denXY + c.Cd_z * fZ0;
                        float uNum = uNumXY + c.Cu_z * fZ0;
                        float vNum = vNumXY + c.Cv_z * fZ0;

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

                            voxelColumn[iz] += tex3D<float>(sinoTex, fu + 0.5f, fv + 0.5f, ia_tex);
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
                        + (size_t)y * vg.Nx + (size_t)x;
                    d_vol[idx] = voxelColumn[iz];
                }
            }


            // fdk式的反投影，带fdk权重
            template<int ZSIZE, int PROJ_PER_KERNEL>
            __global__ void fdk_bp_kernel_fdk_weight(
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
                        + (size_t)y * vg.Nx
                        + (size_t)x;
                    voxelColumn[iz] = d_vol[idx];
                }

                const int Na = endAngle - startAngle;
                const int nBatches = (Na + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;

                for (int batch = 0; batch < nBatches; ++batch)
                {
                    const int batchStart = startAngle + batch * PROJ_PER_KERNEL;
                    const int batchEnd = min(batchStart + PROJ_PER_KERNEL, endAngle);

                    for (int angle = batchStart; angle < batchEnd; ++angle)
                    {
                        const SConeProjGeomVec& v = d_views[angle];
                        const float ia_tex = (float)(angle - startAngle) + 0.5f;

                        // 探测器法向量（角度循环外）
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

                        // FDK 权重参数（角度循环外）
                        const float src_x = v.src.x, src_y = v.src.y, src_z = v.src.z;
                        const float SOD = __fsqrt_rn(src_x * src_x + src_y * src_y + src_z * src_z);
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
                            const float dz = worldZ - src_z;

                            // 体素中心投影到探测器
                            const float denom_n = dx * nx + dy * ny + dz * nz;
                            if (fabsf(denom_n) < 1e-8f) continue;

                            const float t = __fdividef(SDD_plane, denom_n);
                            if (t <= 0.f) continue;

                            const float px = src_x + t * dx - v.detS.x;
                            const float py = src_y + t * dy - v.detS.y;
                            const float pz = src_z + t * dz - v.detS.z;

                            const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) * rcp_U2;
                            const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) * rcp_V2;

                            // FDK 权重：SOD^2 / denom_c^2
                            const float denom_c = dx * cr_x + dy * cr_y + dz * cr_z;
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
                        + (size_t)y * vg.Nx
                        + (size_t)x;
                    d_vol[idx] = voxelColumn[iz];
                }
            }


            template<int ZSIZE, int PROJ_PER_KERNEL>
            __global__ void fdk_bp_kernel_fdk_weight_v2(
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

                float voxelColumn[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) { voxelColumn[iz] = 0.f; continue; }
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                        + (size_t)y * vg.Nx
                        + (size_t)x;
                    voxelColumn[iz] = d_vol[idx];
                }

                const int Na = endAngle - startAngle;
                const int nBatches = (Na + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;

                for (int batch = 0; batch < nBatches; ++batch)
                {
                    const int batchStart = startAngle + batch * PROJ_PER_KERNEL;
                    const int batchEnd = min(batchStart + PROJ_PER_KERNEL, endAngle);

                    for (int angle = batchStart; angle < batchEnd; ++angle)
                    {
                        const FdkAffineCoeff& c = d_coeffs[angle];
                        const SConeProjGeomVec& v = d_views[angle];
                        const float ia_tex = (float)(angle - startAngle) + 0.5f;

                        // fu/fv仿射系数XY部分
                        const float denXY = c.Cd_w + c.Cd_x * fX + c.Cd_y * fY;
                        const float uNumXY = c.Cu_w + c.Cu_x * fX + c.Cu_y * fY;
                        const float vNumXY = c.Cv_w + c.Cv_x * fX + c.Cv_y * fY;

                        const float denStep = c.Cd_z * vg.vox_z;
                        const float uNumStep = c.Cu_z * vg.vox_z;
                        const float vNumStep = c.Cv_z * vg.vox_z;

                        // denom_c系数：Cc_w/x/y/z
                        const float rcp_SOD = __frsqrt_rn(c.source_to_axis_sq);
                        const float Cc_x = -v.src.x * rcp_SOD;
                        const float Cc_y = -v.src.y * rcp_SOD;
                        const float Cc_z = -v.src.z * rcp_SOD;
                        const float Cc_w = __fsqrt_rn(c.source_to_axis_sq);

                        const float cNumXY = Cc_w + Cc_x * fX + Cc_y * fY;
                        const float cNumStep = Cc_z * vg.vox_z;

                        // FDK权重基础值
                        const float w_base = c.source_to_axis_sq * c.dtheta * c.fScaleDTheta;

                        // 初始值退一步
                        const float fZ0 = vg.origin().z + (startZ - 1) * vg.vox_z;
                        float den = denXY + c.Cd_z * fZ0;
                        float uNum = uNumXY + c.Cu_z * fZ0;
                        float vNum = vNumXY + c.Cv_z * fZ0;
                        float cNum = cNumXY + Cc_z * fZ0;

#pragma unroll
                        for (int iz = 0; iz < ZSIZE; ++iz)
                        {
                            den += denStep;
                            uNum += uNumStep;
                            vNum += vNumStep;
                            cNum += cNumStep;

                            const int zIdx = startZ + iz;
                            if (zIdx >= vg.Nz) continue;
                            if (den < 1e-8f) continue;
                            if (fabsf(cNum) < 1e-8f) continue;

                            const float fr = __frcp_rn(den);
                            const float fu = uNum * fr;
                            const float fv = vNum * fr;
                            const float w = w_base * __frcp_rn(cNum * cNum);

                            voxelColumn[iz] += tex3D<float>(sinoTex,
                                fu + 0.5f, fv + 0.5f, ia_tex) * w;
                        }
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
                    d_vol[idx] = voxelColumn[iz];
                }
            }


            template<int ZSIZE, int PROJ_PER_KERNEL>
            __global__ void fdk_bp_kernel_matched_weight(
                cudaTextureObject_t                  sinoTex,
                const SConeProjGeomVec* __restrict__ d_views,
                const FdkAffineCoeff* __restrict__   d_coeffs,
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

                const float scale = vg.vox_x * vg.vox_y * vg.vox_z / d_coeffs[0].du_mm / d_coeffs[0].dv_mm;

                const float fX = vg.origin().x + x * vg.vox_x;
                const float fY = vg.origin().y + y * vg.vox_y;

                float voxelColumn[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= vg.Nz) { voxelColumn[iz] = 0.f; continue; }
                    const size_t idx = (size_t)zIdx * vg.Ny * vg.Nx
                        + (size_t)y * vg.Nx
                        + (size_t)x;
                    voxelColumn[iz] = d_vol[idx];
                }

                const int Na = endAngle - startAngle;
                const int nBatches = (Na + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;

                for (int batch = 0; batch < nBatches; ++batch)
                {
                    const int batchStart = startAngle + batch * PROJ_PER_KERNEL;
                    const int batchEnd = min(batchStart + PROJ_PER_KERNEL, endAngle);

                    for (int angle = batchStart; angle < batchEnd; ++angle)
                    {
                        const FdkAffineCoeff& c = d_coeffs[angle];
                        const SConeProjGeomVec& v = d_views[angle];
                        const float ia_tex = (float)(angle - startAngle) + 0.5f;

                        // ── 仿射系数 XY 部分 ──────────────────────────────────
                        const float denXY = c.Cd_w + c.Cd_x * fX + c.Cd_y * fY;
                        const float uNumXY = c.Cu_w + c.Cu_x * fX + c.Cu_y * fY;
                        const float vNumXY = c.Cv_w + c.Cv_x * fX + c.Cv_y * fY;

                        const float denStep = c.Cd_z * vg.vox_z;
                        const float uNumStep = c.Cu_z * vg.vox_z;
                        const float vNumStep = c.Cv_z * vg.vox_z;

                        // ── 源点坐标（Z循环外）───────────────────────────────
                        const float src_x = v.src.x;
                        const float src_y = v.src.y;
                        const float src_z = v.src.z;

                        // ── 体素XY到源的方向分量（Z循环外）──────────────────
                        const float dx_xy = fX - src_x;
                        const float dy_xy = fY - src_y;

                        // 初始值退一步
                        const float fZ0 = vg.origin().z + (startZ - 1) * vg.vox_z;
                        float den = denXY + c.Cd_z * fZ0;
                        float uNum = uNumXY + c.Cu_z * fZ0;
                        float vNum = vNumXY + c.Cv_z * fZ0;

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

                            // ── L² = |src - det(u,v)|² ────────────────────────
                            // Precompute expands the exact old expression:
                            // det(u,v) = detS + u*detU + v*detV.
                            const float L2 = fmaf(fv, fmaf(fv, c.L2_vv,
                                               c.L2_v + fu * c.L2_uv),
                                             fmaf(fu, fmaf(fu, c.L2_uu, c.L2_u),
                                                  c.L2_0));
                            if (L2 < 1e-8f) continue;
                            const float L = __fsqrt_rn(L2);

                            // ── lsq = |src - voxel|²（源到体素距离平方）─────
                            const float worldZ = vg.origin().z + zIdx * vg.vox_z;
                            const float dz = worldZ - src_z;
                            const float lsq = dx_xy * dx_xy + dy_xy * dy_xy + dz * dz;
                            if (lsq < 1e-8f) continue;

                            // ── matched weight = L³ / (SDD_plane · lsq) ──────
                            // inv_SDD_plane 与派生的实际探测器平面距离使用同一约定；
                            // 相比旧实现只改变运算顺序，
                            // and repeated geometry work have changed.
                            const float w_matched =
                                __fdividef(L2 * L, lsq) * c.inv_SDD_plane * scale;

                            voxelColumn[iz] += tex3D<float>(sinoTex,
                                fu + 0.5f, fv + 0.5f, ia_tex) * w_matched;
                        }
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
                    d_vol[idx] = voxelColumn[iz];
                }
            }


            template<int ZSIZE, int PROJ_PER_KERNEL>
            static void fdk_bp_launch_impl(
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



                fdk_bp_kernel_fdk_weight_v2<ZSIZE, PROJ_PER_KERNEL> << <grid, block, 0, stream >> > (
                    sinoTex, d_views, d_coeffs, d_vol, vg,
                    startAngle, endAngle);
            }


            template<int ZSIZE, int PROJ_PER_KERNEL>
            static void fdk_matched_bp_launch_impl(
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



                fdk_bp_kernel_matched_weight<ZSIZE, PROJ_PER_KERNEL> << <grid, block, 0, stream >> > (
                    sinoTex, d_views, d_coeffs, d_vol, vg,
                    startAngle, endAngle);
            }

        } // namespace detail

        void fdk_bp_launch(
            cudaTextureObject_t     sinoTex,
            const SConeProjGeomVec* d_views_world,
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
            detail::fdk_bp_launch_impl<4, 32>(
                sinoTex, d_views_world, d_coeffs, d_vol, vg,
                0, Na, stream);
            YK_CUDA_KERNEL_CHECK();
        }


        void fdk_matched_bp_launch(
            cudaTextureObject_t     sinoTex,
            const SConeProjGeomVec* d_views_world,
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
            detail::fdk_matched_bp_launch_impl<4, 32>(
                sinoTex, d_views_world, d_coeffs, d_vol, vg,
                0, Na, stream);
            YK_CUDA_KERNEL_CHECK();
        }

    } // namespace Bp
} // namespace YK
