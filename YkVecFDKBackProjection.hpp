#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>

#include "YkGlobals.h"
#include "YkFDKFilter.hpp"
#include "YkFDKPreWeight.hpp"
#include "YkVecGeo.hpp"
#include "YKtestconv.hpp"

namespace YK {

    // ===============================
    // float3 helpers
    // ===============================
    __device__ __forceinline__ float3 f3_add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ __forceinline__ float3 f3_sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __device__ __forceinline__ float3 f3_mul(float3 a, float t) { return make_float3(a.x * t, a.y * t, a.z * t); }
    __device__ __forceinline__ float  f3_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ __forceinline__ float3 f3_cross(float3 a, float3 b) {
        return make_float3(a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
    }

    // ===============================
    // bilinear sample for one view (Nv x Nu), u-fastest
    // ===============================
    __device__ __forceinline__ float bilinear_sample_2d(const float* img, int Nu, int Nv, float u, float v)
    {
        if (u < 0.0f || u >(float)(Nu - 1) || v < 0.0f || v >(float)(Nv - 1)) return 0.0f;

        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = (u0 + 1 < Nu) ? (u0 + 1) : u0;
        int v1 = (v0 + 1 < Nv) ? (v0 + 1) : v0;

        float fu = u - (float)u0;
        float fv = v - (float)v0;

        float p00 = img[v0 * Nu + u0];
        float p10 = img[v0 * Nu + u1];
        float p01 = img[v1 * Nu + u0];
        float p11 = img[v1 * Nu + u1];

        float p0 = p00 + fu * (p10 - p00);
        float p1 = p01 + fu * (p11 - p01);
        return p0 + fv * (p1 - p0);
    }

    // ===============================
    // crop padded -> Nu (per view)
    // ===============================
    __global__ void kernel_crop_u_2d(const float* src, float* dst, int Nu, int Nv, int paddedN, int start_u)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int v = blockIdx.y * blockDim.y + threadIdx.y;
        if (u >= Nu || v >= Nv) return;

        int su = start_u + u;
        float val = 0.0f;
        if (su >= 0 && su < paddedN) val = src[v * paddedN + su];
        dst[v * Nu + u] = val;
    }

    // ============================================================
    // Vector projection + terms for FDK weight:
    // denom = (P-S)·n_hat
    // DSD_n = (detS-S)·n_hat  (signed distance along normal)
    // enforce t = DSD_n/denom > 0 by flipping normal if necessary
    // ============================================================
    __device__ __forceinline__ bool project_uv_and_terms(
        const SConeProjectionVec& g,
        float3 P,
        float& u_pix,
        float& v_pix,
        float& denom,
        float& DSD_n,
        float& SOD)
    {
        float3 U = g.detU;
        float3 V = g.detV;

        float3 n = f3_cross(U, V);
        float nlen2 = f3_dot(n, n);
        if (nlen2 < 1e-20f) return false;

        float invn = rsqrtf(nlen2);
        float3 nh = make_float3(n.x * invn, n.y * invn, n.z * invn);

        float3 dir = f3_sub(P, g.src);

        denom = f3_dot(dir, nh);
        if (fabsf(denom) < 1e-8f) return false;

        float SOD_square = f3_dot(g.src, g.src);
        SOD = sqrtf(SOD_square);

        DSD_n = f3_dot(f3_sub(g.detS, g.src), nh);
        if (fabsf(DSD_n) < 1e-8f) return false;

        // CRITICAL: force t > 0 => DSD_n and denom same sign
        if (DSD_n * denom < 0.0f) {
            nh = make_float3(-nh.x, -nh.y, -nh.z);
            denom = -denom;
            DSD_n = -DSD_n;
        }

        float t = DSD_n / denom;
        if (t <= 0.0f) return false;

        float3 Q = f3_add(g.src, f3_mul(dir, t));
        float3 D = f3_sub(Q, g.detS);

        // Solve D = u*U + v*V  (u,v are pixel coordinates)
        float UU = f3_dot(U, U);
        float VV = f3_dot(V, V);
        float UV = f3_dot(U, V);
        float DU = f3_dot(D, U);
        float DV = f3_dot(D, V);

        float det = UU * VV - UV * UV;
        if (fabsf(det) < 1e-12f) return false;

        u_pix = (DU * VV - DV * UV) / det;
        v_pix = (-DU * UV + DV * UU) / det;
        return true;
    }

    // ============================================================
    // Vec backprojection kernel (chunk)
    // - views_chunk layout: [K][Nv][Nu], u-fastest
    // - IMPORTANT: integrate with per-view dtheta[a]
    // ============================================================
    __global__ void fdk_vec_backproject_chunk_kernel(
        const float* __restrict__ views_chunk,            // [K*Nv*Nu]
        const SConeProjectionVec* __restrict__ d_geo,     // [Ang]
        const float* __restrict__ d_dtheta,               // [Ang]  <-- per-view Δθ
        float* __restrict__ vol,                          // [Nz*Ny*Nx]
        int Nx, int Ny, int Nz, float vox,
        int Nu, int Nv,
        int K, int base_a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= Nx || y >= Ny || z >= Nz) return;

        // NOTE: you intentionally use (y,x,z) mapping here; keep as-is
        float3 P = make_float3(
            (y - (Ny - 1) * 0.5f) * vox,
            (x - (Nx - 1) * 0.5f) * vox,
            (z - (Nz - 1) * 0.5f) * vox
        );

        float acc = 0.0f;

        for (int i = 0; i < K; ++i) {
            int a = base_a + i;
            const SConeProjectionVec& g = d_geo[a];

            float u, v, denom, DSD_n, SOD;
            if (!project_uv_and_terms(g, P, u, v, denom, DSD_n, SOD)) continue;

            const float* view_i = views_chunk + (size_t)i * Nv * Nu;
            float p = bilinear_sample_2d(view_i, Nu, Nv, u, v);

            // FDK Jacobian-like factor (your chosen form)
            float w = (SOD * SOD) / (denom * denom);

            // integrate with per-view Δθ
            float dt = d_dtheta[a];

            acc += p * w * dt;
        }

        size_t vidx = (size_t)z * Ny * Nx + (size_t)y * Nx + x;
        vol[vidx] += acc; // <-- no extra global dtheta
    }

    // ============================================================
    // Build per-view dtheta from vector-geometry (Z-axis rotation)
    // src parameterization consistent with your circular geometry:
    //   src = ( SOD*sinθ, -SOD*cosθ, 0 )
    // => θ = atan2(src.x, -src.y)
    // ============================================================
    inline void build_dtheta_from_geo_zaxis(
        const std::vector<SConeProjectionVec>& h_geo,
        std::vector<float>& h_dtheta)
    {
        const int Ang = (int)h_geo.size();
        h_dtheta.assign(Ang, 0.0f);
        if (Ang <= 1) return;

        std::vector<float> theta(Ang);

        // 1) theta from src
        for (int a = 0; a < Ang; ++a) {
            float sx = h_geo[a].src.x;
            float sy = h_geo[a].src.y;
            theta[a] = atan2f(sx, -sy); // atan2(sinθ, cosθ)
        }

        // 2) unwrap to avoid 2π jumps
        for (int a = 1; a < Ang; ++a) {
            float t = theta[a];
            float p = theta[a - 1];
            while (t - p > M_PI) t -= 2.0f * M_PI;
            while (t - p < -M_PI) t += 2.0f * M_PI;
            theta[a] = t;
        }

        // 3) center difference (short-scan safe: no wrap)
        for (int a = 0; a < Ang; ++a) {
            float dt;
            if (a == 0)          dt = theta[1] - theta[0];
            else if (a == Ang - 1) dt = theta[Ang - 1] - theta[Ang - 2];
            else                 dt = 0.5f * (theta[a + 1] - theta[a - 1]);

            if (dt < 0.0f) dt = -dt;
            if (dt < 1e-8f) dt = 1e-8f;
            h_dtheta[a] = dt;
        }
    }

    // ============================================================
    // Streaming vec-FDK (offset-aware)
    // Input projection layout on host: [Ang][Nv][Nu] (A-V-U), u-fastest
    // ============================================================
    inline void fdk_vec_recon_streaming(
        const float* h_proj,                               // host [Ang*Nv*Nu]
        float* d_vol,                                      // device [Nz*Ny*Nx]
        const std::vector<SConeProjectionVec>& h_geo,       // host [Ang]
        int Nu, int Nv, int Ang,
        int Nx, int Ny, int Nz, float vox,
        float SID, float SDD, float du, float dv,
        int Kchunk,
        float offsetU_pix,
        float offsetV_pix,
        cudaStream_t stream)
    {
        (void)SID;

        // upload geometry
        SConeProjectionVec* d_geo = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_geo, (size_t)Ang * sizeof(SConeProjectionVec)));
        YK_CUDA_CHECK(cudaMemcpyAsync(d_geo, h_geo.data(),
            (size_t)Ang * sizeof(SConeProjectionVec),
            cudaMemcpyHostToDevice, stream));

        // build per-view dtheta on host from geo, upload to device
        std::vector<float> h_dtheta;
        build_dtheta_from_geo_zaxis(h_geo, h_dtheta);

        float* d_dtheta = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_dtheta, (size_t)Ang * sizeof(float)));
        YK_CUDA_CHECK(cudaMemcpyAsync(d_dtheta, h_dtheta.data(),
            (size_t)Ang * sizeof(float),
            cudaMemcpyHostToDevice, stream));

        const size_t view_elems = (size_t)Nu * Nv;

        // ---- Preweight (cos) ----
        PreweightManager pw;
        pw.init(Nu, Nv, 1, du, dv, /*DSD=*/SDD,
            /*offsetU=*/offsetU_pix,
            /*offsetV=*/offsetV_pix,
            /*power=*/1, stream);

        // ---- Filter (1D along U, batch=Nv) ----
        FilterManager fm;
        fm.init(Nu, du, /*batch=*/Nv, stream);
        const int paddedN = fm.getPaddedN();

        // ---- Offset-aware padding alignment ----
        const float offsetX = offsetU_pix; // U-direction axis offset
        const float axis_idx = (Nu - 1) * 0.5f + offsetX;
        const int start_u = (int)lrintf(paddedN * 0.5f - axis_idx);

        // buffers
        float* d_view_in = nullptr; // [Nv*Nu]
        float* d_view_pw = nullptr; // [Nv*Nu]
        float* d_view_flt = nullptr; // [Nv*Nu]
        float* d_padded = nullptr; // [Nv*paddedN]
        float* d_chunk = nullptr; // [Kchunk*Nv*Nu]

        YK_CUDA_CHECK(cudaMalloc(&d_view_in, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_pw, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_flt, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_padded, (size_t)Nv * paddedN * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_chunk, (size_t)Kchunk * view_elems * sizeof(float)));

        // clear volume
        YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0, (size_t)Nx * Ny * Nz * sizeof(float), stream));

        for (int base = 0; base < Ang; base += Kchunk) {
            int K = (base + Kchunk <= Ang) ? Kchunk : (Ang - base);

            // build filtered views into d_chunk
            for (int i = 0; i < K; ++i) {
                int a = base + i;
                const float* h_view = h_proj + (size_t)a * view_elems;

                // H2D one view
                YK_CUDA_CHECK(cudaMemcpyAsync(d_view_in, h_view, view_elems * sizeof(float),
                    cudaMemcpyHostToDevice, stream));

                // preweight
                pw.setStream(stream);
                pw.apply(d_view_in, d_view_pw);

                // pad + filter
                fm.setStream(stream);

                dim3 b1(256, 1);
                dim3 g1((paddedN + 255) / 256, Nv);
                YK::_kernel_pad_with_offset << <g1, b1, 0, stream >> > (
                    d_view_pw, d_padded, Nu, Nv, paddedN, offsetX);
                YK_CUDA_KERNEL_CHECK();

                fm.apply(d_padded);

                // crop -> Nu
                dim3 b2(16, 16);
                dim3 g2((Nu + b2.x - 1) / b2.x, (Nv + b2.y - 1) / b2.y);
                kernel_crop_u_2d << <g2, b2, 0, stream >> > (
                    d_padded, d_view_flt, Nu, Nv, paddedN, start_u);
                YK_CUDA_KERNEL_CHECK();

                // copy into chunk slot (device-to-device)
                float* d_slot = d_chunk + (size_t)i * view_elems;
                YK_CUDA_CHECK(cudaMemcpyAsync(d_slot, d_view_flt, view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }

            // backproject chunk
            dim3 block(8, 8, 4);
            dim3 grid((Nx + block.x - 1) / block.x,
                (Ny + block.y - 1) / block.y,
                (Nz + block.z - 1) / block.z);

            fdk_vec_backproject_chunk_kernel << <grid, block, 0, stream >> > (
                d_chunk, d_geo, d_dtheta, d_vol,
                Nx, Ny, Nz, vox,
                Nu, Nv,
                K, base);
            YK_CUDA_KERNEL_CHECK();
        }

        // cleanup
        YK_CUDA_CHECK(cudaFree(d_geo));
        YK_CUDA_CHECK(cudaFree(d_dtheta));
        YK_CUDA_CHECK(cudaFree(d_view_in));
        YK_CUDA_CHECK(cudaFree(d_view_pw));
        YK_CUDA_CHECK(cudaFree(d_view_flt));
        YK_CUDA_CHECK(cudaFree(d_padded));
        YK_CUDA_CHECK(cudaFree(d_chunk));
    }

    // ------------------------------------------------------------
    // Backward-compatible overload: defaults offsets = 0
    // ------------------------------------------------------------
    inline void fdk_vec_recon_streaming(
        const float* h_proj,
        float* d_vol,
        const std::vector<SConeProjectionVec>& h_geo,
        int Nu, int Nv, int Ang,
        int Nx, int Ny, int Nz, float vox,
        float SID, float SDD, float du, float dv,
        int Kchunk,
        cudaStream_t stream)
    {
        fdk_vec_recon_streaming(h_proj, d_vol, h_geo,
            Nu, Nv, Ang,
            Nx, Ny, Nz, vox,
            SID, SDD, du, dv,
            Kchunk,
            /*offsetU_pix=*/0.0f,
            /*offsetV_pix=*/0.0f, stream);
    }

} // namespace YK
