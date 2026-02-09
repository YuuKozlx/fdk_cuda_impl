#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime_api.h>
#include "YkFDKFilter.hpp"
#include "YkGlobals.h"
#include "YkSampling2D.hpp"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"
#include "YkUtil.hpp"

namespace YK {

    // ============================================================
    // Constants
    // ============================================================
    static constexpr float PI_F = 3.14159265358979323846f;

    

    // ============================================================
    // Per-view vector-geometry preweight (power = 1)
    // w_pre(u,v) = |(detS-src)·n_hat| / |(Q-src)|
    // Q = detS + u*detU + v*detV
    // ============================================================
    __global__ void kernel_preweight_vec_view_power1(
        const float* __restrict__ src_view,   // [Nv*Nu]
        float* __restrict__ dst_view,         // [Nv*Nu]
        SConeProjectionVec g,
        int Nu, int Nv)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int v = blockIdx.y * blockDim.y + threadIdx.y;
        if (u >= Nu || v >= Nv) return;

        float3 U = g.detU;
        float3 V = g.detV;

        float3 n = f3_cross(U, V);
        float nlen2 = f3_dot(n, n);
        if (nlen2 < 1e-20f) { dst_view[v * Nu + u] = 0.0f; return; }

        float invn = rsqrtf(nlen2);
        float3 nh = make_float3(n.x * invn, n.y * invn, n.z * invn);

        float DSD_n = f3_dot(f3_sub(g.detS, g.src), nh);
        if (DSD_n < 0.0f) DSD_n = -DSD_n;

        float3 Q = f3_add(g.detS, f3_add(f3_mul(U, (float)u), f3_mul(V, (float)v)));
        float3 QS = f3_sub(Q, g.src);

        float r2 = f3_dot(QS, QS);
        float r = sqrtf(fmaxf(r2, 1e-20f));

        float w = DSD_n / r;
        dst_view[v * Nu + u] = src_view[v * Nu + u] * w;
    }

    // ============================================================
    // project_uv + denom for BP (vector geometry)
    // denom = (P-S)·n_hat, enforce t>0 by flipping normal
    // ============================================================
    __device__ __forceinline__ bool project_uv_and_terms(
        const SConeProjectionVec& g,
        float3 P,
        float& u_pix,
        float& v_pix,
        float& denom,
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

        SOD = sqrtf(f3_dot(g.src, g.src));

        float DSD_n = f3_dot(f3_sub(g.detS, g.src), nh);
        if (fabsf(DSD_n) < 1e-8f) return false;

        if (DSD_n * denom < 0.0f) {
            nh = make_float3(-nh.x, -nh.y, -nh.z);
            denom = -denom;
            DSD_n = -DSD_n;
        }

        float t = DSD_n / denom;
        if (t <= 0.0f) return false;

        float3 Q = f3_add(g.src, f3_mul(dir, t));
        float3 D = f3_sub(Q, g.detS);

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
    // BP chunk kernel with per-view dtheta[a]
    // ============================================================
    __global__ void fdk_vec_backproject_chunk_kernel(
        const float* __restrict__ views_chunk,            // [K*Nv*Nu]
        const SConeProjectionVec* __restrict__ d_geo,     // [Ang]
        const float* __restrict__ d_dtheta,               // [Ang]
        float* __restrict__ vol,                          // [Nz*Ny*Nx]
        int Nx, int Ny, int Nz, float vox,
        int Nu, int Nv,
        int K, int base_a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= Nx || y >= Ny || z >= Nz) return;

        // keep your mapping (y,x,z)
        float3 P = make_float3(
            (y - (Ny - 1) * 0.5f) * vox,
            (x - (Nx - 1) * 0.5f) * vox,
            (z - (Nz - 1) * 0.5f) * vox
        );

        float acc = 0.0f;

        for (int i = 0; i < K; ++i) {
            int a = base_a + i;
            const SConeProjectionVec& g = d_geo[a];

            float u, v, denom, SOD;
            if (!project_uv_and_terms(g, P, u, v, denom, SOD)) continue;

            const float* view_i = views_chunk + (size_t)i * Nv * Nu;
            float p = YK::Interp::sample2d(view_i, Nu, Nv, u, v);

            float w = (SOD * SOD) / (denom * denom);
            acc += p * w * d_dtheta[a];
        }

        size_t vidx = (size_t)z * Ny * Nx + (size_t)y * Nx + x;
        vol[vidx] += acc;
    }

    // ============================================================
    // per-view dtheta from geo (Z rotation)
    // theta = atan2(src.x, -src.y)
    // ============================================================
    inline void build_dtheta_from_geo_zaxis(
        const std::vector<SConeProjectionVec>& h_geo,
        std::vector<float>& h_dtheta)
    {
        const int Ang = (int)h_geo.size();
        h_dtheta.assign(Ang, 0.0f);
        if (Ang <= 1) return;

        std::vector<float> theta(Ang);

        for (int a = 0; a < Ang; ++a) {
            float sx = h_geo[a].src.x;
            float sy = h_geo[a].src.y;
            theta[a] = std::atan2(sx, -sy);
        }

        for (int a = 1; a < Ang; ++a) {
            float t = theta[a];
            float p = theta[a - 1];
            while (t - p > PI_F) t -= 2.0f * PI_F;
            while (t - p < -PI_F) t += 2.0f * PI_F;
            theta[a] = t;
        }

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
    // Compute du/dv arrays from geo (for validate/record),
    // but return du0 (const) for filter.
    // ============================================================
    inline void compute_du_dv_from_geo(
        const std::vector<SConeProjectionVec>& h_geo,
        std::vector<float>& h_du,
        std::vector<float>& h_dv,
        float& du0,
        float& dv0)
    {
        const int Ang = (int)h_geo.size();
        h_du.resize(Ang);
        h_dv.resize(Ang);
        for (int a = 0; a < Ang; ++a) {
            h_du[a] = f3_len(h_geo[a].detU);
            h_dv[a] = f3_len(h_geo[a].detV);
        }
        du0 = (Ang > 0) ? h_du[0] : 1.0f;
        dv0 = (Ang > 0) ? h_dv[0] : 1.0f;
        if (!(du0 > 0.0f)) du0 = 1.0f;
        if (!(dv0 > 0.0f)) dv0 = 1.0f;
    }

    // ============================================================
    // AUTO offset from geo (isocenter at origin):
    // - central ray src -> O=(0,0,0)
    // - intersect detector plane -> (cu,cv) in det basis
    // - offset = (cu,cv) - center_index
    // Use median across views for robustness.
    // ============================================================
    inline bool compute_offset_uv_from_geo_isocenter0_one(
        const SConeProjectionVec& g,
        int Nu, int Nv,
        float& offsetU_pix,
        float& offsetV_pix)
    {
        const float3 O = make_float3(0.0f, 0.0f, 0.0f);

        float3 n = f3_cross(g.detU, g.detV);
        float nlen = f3_len(n);
        if (nlen < 1e-12f) return false;
        float3 nh = make_float3(n.x / nlen, n.y / nlen, n.z / nlen);

        float3 dir = f3_sub(O, g.src); // = -src
        float denom = f3_dot(dir, nh);
        if (std::fabs(denom) < 1e-12f) return false;

        float t = f3_dot(f3_sub(g.detS, g.src), nh) / denom;
        if (t <= 0.0f) return false;

        float3 C = make_float3(
            g.src.x + t * dir.x,
            g.src.y + t * dir.y,
            g.src.z + t * dir.z
        );

        float3 D = f3_sub(C, g.detS);

        float UU = f3_dot(g.detU, g.detU);
        float VV = f3_dot(g.detV, g.detV);
        float UV = f3_dot(g.detU, g.detV);
        float DU = f3_dot(D, g.detU);
        float DV = f3_dot(D, g.detV);

        float det = UU * VV - UV * UV;
        if (std::fabs(det) < 1e-20f) return false;

        float cu = (DU * VV - DV * UV) / det;
        float cv = (-DU * UV + DV * UU) / det;

        offsetU_pix = cu - (Nu - 1) * 0.5f;
        offsetV_pix = cv - (Nv - 1) * 0.5f;
        return std::isfinite(offsetU_pix) && std::isfinite(offsetV_pix);
    }

    inline bool compute_offset_uv_from_geo_isocenter0_median(
        const std::vector<SConeProjectionVec>& h_geo,
        int Nu, int Nv,
        float& offsetU_pix,
        float& offsetV_pix)
    {
        std::vector<float> ou, ov;
        ou.reserve(h_geo.size());
        ov.reserve(h_geo.size());

        for (size_t a = 0; a < h_geo.size(); ++a) {
            float u = 0.0f, v = 0.0f;
            if (compute_offset_uv_from_geo_isocenter0_one(h_geo[a], Nu, Nv, u, v)) {
                ou.push_back(u);
                ov.push_back(v);
            }
        }
        if (ou.empty()) return false;

        auto median = [](std::vector<float>& x) -> float {
            const size_t n = x.size();
            const size_t mid = n / 2;
            std::nth_element(x.begin(), x.begin() + mid, x.end());
            float m = x[mid];
            if ((n & 1u) == 0u) {
                std::nth_element(x.begin(), x.begin() + (mid - 1), x.end());
                m = 0.5f * (m + x[mid - 1]);
            }
            return m;
            };

        offsetU_pix = median(ou);
        offsetV_pix = median(ov);
        return true;
    }

    // ============================================================
    // ✅ Single public entry:
    // Streaming vec-FDK (per-view):
    //  - auto offsetU/V from geo (median)
    //  - preweight per-view (power=1)
    //  - filter uses du0 as constant (computed from geo[0])
    //  - BP uses per-view dtheta[a]
    //
    // Input layout on host: h_proj[a*(Nv*Nu) + v*Nu + u]  (A-V-U)
    // ============================================================
    inline void fdk_vec_recon_streaming(
        const float* h_proj,                               // host [Ang*Nv*Nu]
        float* d_vol,                                      // device [Nz*Ny*Nx]
        const std::vector<SConeProjectionVec>& h_geo,       // host [Ang]
        int Nu, int Nv, int Ang,
        int Nx, int Ny, int Nz, float vox,
        int Kchunk,
        cudaStream_t stream)
    {
        // -------- auto offsets from geo --------
        float offsetU_pix = 0.0f, offsetV_pix = 0.0f;
        if (!compute_offset_uv_from_geo_isocenter0_median(h_geo, Nu, Nv, offsetU_pix, offsetV_pix)) {
            offsetU_pix = 0.0f;
            offsetV_pix = 0.0f;
        }

        // -------- upload geometry --------
        SConeProjectionVec* d_geo = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_geo, (size_t)Ang * sizeof(SConeProjectionVec)));
        YK_CUDA_CHECK(cudaMemcpyAsync(d_geo, h_geo.data(),
            (size_t)Ang * sizeof(SConeProjectionVec),
            cudaMemcpyHostToDevice, stream));

        // -------- per-view dtheta --------
        std::vector<float> h_dtheta;
        build_dtheta_from_geo_zaxis(h_geo, h_dtheta);

        float* d_dtheta = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_dtheta, (size_t)Ang * sizeof(float)));
        YK_CUDA_CHECK(cudaMemcpyAsync(d_dtheta, h_dtheta.data(),
            (size_t)Ang * sizeof(float),
            cudaMemcpyHostToDevice, stream));

        // -------- du/dv arrays (computed), but filter uses du0 constant --------
        std::vector<float> h_du, h_dv;
        float du0 = 1.0f, dv0 = 1.0f;
        compute_du_dv_from_geo(h_geo, h_du, h_dv, du0, dv0);
        (void)dv0;

        //-------- PreweightManager --------

        // -------- filter manager (constant du0) --------
        FilterManager fm;
        fm.init(Nu, du0, /*batch=*/Nv, stream);
        const int paddedN = fm.getPaddedN();

        // offset-aware padding alignment (U only)
        const float offsetX = offsetU_pix;
        const float axis_idx = (Nu - 1) * 0.5f + offsetX;
        const int start_u = (int)lrintf(paddedN * 0.5f - axis_idx);

        const size_t view_elems = (size_t)Nu * Nv;

        // -------- buffers --------
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

        // launch configs
        dim3 pre_b(16, 16);
        dim3 pre_g((Nu + pre_b.x - 1) / pre_b.x, (Nv + pre_b.y - 1) / pre_b.y);

        dim3 b1(256, 1);
        dim3 g1((paddedN + 255) / 256, Nv);



        // -------- main loop --------
        for (int base = 0; base < Ang; base += Kchunk) {
            int K = (base + Kchunk <= Ang) ? Kchunk : (Ang - base);

            // build filtered views into d_chunk
            for (int i = 0; i < K; ++i) {
                int a = base + i;
                const float* h_view = h_proj + (size_t)a * view_elems;

                YK_CUDA_CHECK(cudaMemcpyAsync(d_view_in, h_view, view_elems * sizeof(float),
                    cudaMemcpyHostToDevice, stream));

                // per-view preweight (power=1)
                kernel_preweight_vec_view_power1 << <pre_g, pre_b, 0, stream >> > (
                    d_view_in, d_view_pw, h_geo[a], Nu, Nv);
                YK_CUDA_KERNEL_CHECK();

                // pad rows
                fm.setStream(stream);
                YK::_kernel_pad_with_offset << <g1, b1, 0, stream >> > (
                    d_view_pw, d_padded, Nu, Nv, paddedN, offsetX);
                YK_CUDA_KERNEL_CHECK();

                // filter in-place
                fm.apply(d_padded);

                // crop
                YK::Util::crop_u_2d(d_padded, d_view_flt, Nu, Nv, paddedN, start_u, stream);
                YK_CUDA_KERNEL_CHECK();

                // copy into chunk
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

} // namespace YK
