#pragma once
#include <cuda_runtime.h>
#include "YkGlobals.h"
#include "YkVecOperation.hpp"

namespace YK {

    // ============================================================
// Derived-based projector (device)
//   - u/v: by intersecting ray (src->P) with detector plane
//   - denom_c: (P-S)·ray0hat  (projection onto central ray direction)
// Uses exported per-view info (gv):
//   gv.nhat, gv.DSD_n, gv.ray0hat,
//   gv.basis_valid, gv.UU, gv.VV, gv.UV, gv.invDetUV
//
// NOTE
//   - We do a LOCAL flip (nhat,DSD_n) only for intersection to enforce t>0.
//     This does NOT modify stored gv.nhat / gv.DSD_n.
// ============================================================
    template<int ZSIZE>
    __global__ void fdk_bp_kernel(
        const cudaTextureObject_t* __restrict__ tex_views, // [K]
        float* __restrict__ vol,
        int Nx, int Ny, int Nz, float vox,
        int K)
        // base_a 不再需要，gC_coeffs 已是当前 chunk 的偏移
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= Nx || y >= Ny) return;

        const int startZ = blockIdx.z * ZSIZE;
        if (startZ >= Nz) return;

        const float fX = (y - (Ny - 1) * 0.5f) * vox;
        const float fY = (x - (Nx - 1) * 0.5f) * vox;
        const float fZ = (startZ - (Nz - 1) * 0.5f) * vox;

        // Z 方向累加器，全部住在寄存器
        float Z[ZSIZE];
#pragma unroll
        for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

        for (int i = 0; i < K; ++i) {
            const FdkAffineCoeff& c = gC_coeffs[i];

            // (X, Y) 的基值，只算一次
            float uNum = c.Cu.w + fX * c.Cu.x + fY * c.Cu.y + fZ * c.Cu.z;
            float vNum = c.Cv.w + fX * c.Cv.x + fY * c.Cv.y + fZ * c.Cv.z;
            float den = c.Cd.w + fX * c.Cd.x + fY * c.Cd.y + fZ * c.Cd.z;

            // Z 步长
            const float uStep = c.Cu.z * vox;
            const float vStep = c.Cv.z * vox;
            const float dStep = c.Cd.z * vox;

            const float w_base = c.SID2 * c.dtheta;

#pragma unroll
            for (int iz = 0; iz < ZSIZE; ++iz) {
                float fr = __fdividef(1.f, den);
                float u = uNum * fr;
                float v = vNum * fr;
                float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                Z[iz] += p * (w_base * fr * fr);

                uNum += uStep;
                vNum += vStep;
                den += dStep;
            }
        }

        // 写回，边界检查
        const int endZ = min(startZ + ZSIZE, Nz);
#pragma unroll
        for (int iz = 0; iz < ZSIZE; ++iz) {
            if (startZ + iz < endZ) {
                size_t idx = (size_t)(startZ + iz) * Ny * Nx + (size_t)y * Nx + x;
                vol[idx] += Z[iz];
            }
        }
    }



    __device__ __forceinline__ bool project_uv_and_terms_derived(
        const SConeProjectionVec& g,
        const SFDKGeoParamPerView& gv,
        float3 P,
        float& u_pix,
        float& v_pix,
        float& denom_c)
    {
        const float3 dir = f3_sub(P, g.src);

        denom_c = f3_dot(dir, gv.ray_center);

        const float denom_n = f3_dot(dir, gv.det_n);
        if (fabsf(denom_n) < 1e-8f) return false;

        const float t = __fdividef(gv.SDD_mm, denom_n);
        if (t <= 0.f) return false;

        // D·detU = t*(dir·detU) - (detS-src)·detU
        const float DU = t * f3_dot(dir, g.detU) - gv.detS_sub_src_dot_dU;
        const float DV = t * f3_dot(dir, g.detV) - gv.detS_sub_src_dot_dV;

        u_pix = (DU * gv.VV - DV * gv.UV) * gv.invDetUV;
        v_pix = (-DU * gv.UV + DV * gv.UU) * gv.invDetUV;
        return true;
    }


    template<int ZSIZE>
    __global__ void fdk_bp_kernel(
        const cudaTextureObject_t* __restrict__ tex_views,
        const SConeProjectionVec* __restrict__ d_geo,    // 已是 chunk 起始：d_geo + base
        const SFDKGeoParamPerView* __restrict__ d_gv,     // 已是 chunk 起始：d_gv + base
        float* __restrict__ vol,
        int Nx, int Ny, int Nz, float vox,
        int K)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= Nx || y >= Ny) return;

        const int startZ = blockIdx.z * ZSIZE;
        if (startZ >= Nz) return;

        float Z[ZSIZE];
#pragma unroll
        for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

        for (int i = 0; i < K; ++i) {
            const SConeProjectionVec& g = d_geo[i];   // 直接用 i，不加 base_a
            const SFDKGeoParamPerView& gv = d_gv[i];

            const float fX = (y - (Ny - 1) * 0.5f) * vox;
            const float fY = (x - (Nx - 1) * 0.5f) * vox;

#pragma unroll
            for (int iz = 0; iz < ZSIZE; ++iz) {
                const int zIdx = startZ + iz;
                if (zIdx >= Nz) continue;

                float3 P = make_float3(
                    fX, fY,
                    (zIdx - (Nz - 1) * 0.5f) * vox);

                float u, v, denom_c;
                if (!project_uv_and_terms_derived(g, gv, P, u, v, denom_c))
                    continue;

                float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                float w = (gv.SOD_mm * gv.SOD_mm) / (denom_c * denom_c);
                Z[iz] += p * w * gv.dtheta;
            }
        }

        const int endZ = min(startZ + ZSIZE, Nz);
#pragma unroll
        for (int iz = 0; iz < ZSIZE; ++iz) {
            if (startZ + iz < endZ) {
                size_t idx = (size_t)(startZ + iz) * Ny * Nx + (size_t)y * Nx + x;
                vol[idx] += Z[iz];
            }
        }
    }


    // 根据 Nz 选合适的 ZSIZE
    inline void launchBpKernel(
        const cudaTextureObject_t* d_texObjs,
        float* d_vol,
        int Nx, int Ny, int Nz, float vox,
        int K, cudaStream_t stream)
    {
        constexpr int ZSIZE = 4; // 可以根据 Nz 调整，比如 Nz 很大时可以用 8 或 16，但要注意寄存器压力和 occupancy
        dim3 block(16, 16, 1);
        dim3 grid(
            (Nx + block.x - 1) / block.x,
            (Ny + block.y - 1) / block.y,
            (Nz + ZSIZE - 1) / ZSIZE);

        fdk_bp_kernel<ZSIZE> << <grid, block, 0, stream >> > (
            d_texObjs, d_vol,
            Nx, Ny, Nz, vox, K);
        YK_CUDA_KERNEL_CHECK();
    }


    // 非预计算版本，直接在 kernel 内计算 u/v/denom_c，避免了全局内存访问，但计算量增加
    inline void launchBpKernel(
        const cudaTextureObject_t* d_texObjs,
        const SConeProjectionVec* d_geo,   // 传 ctx.d_geo + base
        const SFDKGeoParamPerView* d_gv,    // 传 ctx.d_gv  + base
        float* d_vol,
        int Nx, int Ny, int Nz, float vox,
        int K, cudaStream_t stream)
    {

        constexpr int ZSIZE = 4; // 可以根据 Nz 调整，比如 Nz 很大时可以用 8 或 16，但要注意寄存器压力和 occupancy
        dim3 block(16, 16, 1);
        dim3 grid(
            (Nx + block.x - 1) / block.x,
            (Ny + block.y - 1) / block.y,
            (Nz + ZSIZE - 1) / ZSIZE);

        fdk_bp_kernel<ZSIZE> << <grid, block, 0, stream >> > (
            d_texObjs,
            d_geo,
            d_gv,
            d_vol,
            Nx, Ny, Nz, vox, K);
        YK_CUDA_KERNEL_CHECK();
    }


} // namespace YK