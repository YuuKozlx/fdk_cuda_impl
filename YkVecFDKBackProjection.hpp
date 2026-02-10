#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime_api.h>
#include <sstream>
#include "YkFDKFilter.hpp"
#include "YkFDKVecAlignPadCrop.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFDKVecPreWeight.hpp"
#include "YkGlobals.h"
#include "YkIoDump.hpp"
#include "YkSampling2D.hpp"
#include "YkUtil.hpp"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"

namespace YK { 
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

    inline void fdk_vec_recon_streaming(
        const float* h_proj,                               // host [Ang*Nv*Nu]
        float* d_vol,                                      // device [Nz*Ny*Nx]
        const std::vector<SConeProjectionVec>& h_geo,       // host [Ang]
        int Nu, int Nv, int Ang,
        int Nx, int Ny, int Nz, float vox,
        int Kchunk,
        cudaStream_t stream)
    {
        // ------------------------------------------------------------
        // Debug: dump only one view (preweight/pad/fltpad/crop)
        // ------------------------------------------------------------
        YK::IO::DumpManager dump;
        dump.init("./dbg/", /*enabled=*/true);
        const int  DUMP_A = 0;      // 只保存这一张 view
        bool dumped = false;

        // ============================================================
        // 0) CPU derived params (offset/dtheta/du0 etc.)
        // ============================================================
        GeoDerivedManagerVec derived;
        {
            GeoDerivedManagerVec::GeoDerivedOptions opt;
            opt.offset_mode = GeoDerivedManagerVec::EOffsetMode::PerView;
            opt.isocenter = make_float3(0.0f, 0.0f, 0.0f);
            opt.dtheta_eps = 1e-8f;
            (void)derived.init(Nu, Nv, opt);
            (void)derived.build_geo_params(h_geo);
        }
        const auto& gv = derived.views();
        const float du0 = derived.du0_mm();

        // ============================================================
        // 1) Upload geo
        // ============================================================
        SConeProjectionVec* d_geo = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_geo, (size_t)Ang * sizeof(SConeProjectionVec)));
        YK_CUDA_CHECK(cudaMemcpyAsync(
            d_geo, h_geo.data(),
            (size_t)Ang * sizeof(SConeProjectionVec),
            cudaMemcpyHostToDevice, stream));

        // ============================================================
        // 2) Upload dtheta
        // ============================================================
        float* d_dtheta = nullptr;
        {
            std::vector<float> h_dtheta(Ang, 1e-8f);
            if ((int)gv.size() == Ang) {
                for (int a = 0; a < Ang; ++a) h_dtheta[a] = gv[a].dtheta;
            }
            YK_CUDA_CHECK(cudaMalloc(&d_dtheta, (size_t)Ang * sizeof(float)));
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_dtheta, h_dtheta.data(),
                (size_t)Ang * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        }

        // ============================================================
        // 3) Managers
        // ============================================================
        PreweightManagerVec pw;
        pw.init(Nu, Nv, /*blockThreads=*/256, stream);

        AlignPadCropManagerVec align;
        align.init(Nu, Nv, stream);
        const int paddedN = align.paddedN();

        FilterManager fm;
        // batch = Nv (每行做1D FFT)
        if (!fm.init(Nu, paddedN, du0, /*batch=*/Nv, stream)) {
            // hard fail: your paddedN/params mismatch
            YK_ASSERT(false && "FilterManager init failed");
        }

        // ============================================================
        // 4) Buffers
        // ============================================================
        const size_t view_elems = (size_t)Nu * (size_t)Nv;

        float* d_view_in = nullptr; // [Nv*Nu]
        float* d_view_pw = nullptr; // [Nv*Nu]
        float* d_view_flt = nullptr; // [Nv*Nu]
        float* d_padded = nullptr; // [Nv*paddedN]
        float* d_chunk = nullptr; // [Kchunk*Nv*Nu]

        YK_CUDA_CHECK(cudaMalloc(&d_view_in, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_pw, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_flt, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_padded, (size_t)Nv * (size_t)paddedN * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_chunk, (size_t)Kchunk * view_elems * sizeof(float)));

        YK_CUDA_CHECK(cudaMemsetAsync(
            d_vol, 0,
            (size_t)Nx * (size_t)Ny * (size_t)Nz * sizeof(float),
            stream));

        // ============================================================
        // 5) Main loop
        // ============================================================
        for (int base = 0; base < Ang; base += Kchunk) {
            const int K = (base + Kchunk <= Ang) ? Kchunk : (Ang - base);

            for (int i = 0; i < K; ++i) {
                const int a = base + i;
                const float* h_view = h_proj + (size_t)a * view_elems;

                // H2D
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_view_in, h_view,
                    view_elems * sizeof(float),
                    cudaMemcpyHostToDevice, stream));

                // (1) preweight (K=1)
                pw.setStream(stream);
                pw.applyChunk(d_view_in, d_view_pw, d_geo, /*K=*/1, /*base_a=*/a);

                // dump preweight once
                if (!dumped && a == DUMP_A) {
                    dump.dumpDeviceF32_A("pw", a, d_view_pw, view_elems, stream);
                }

                // (2) pad with per-view offsetU
                float offsetU = 0.0f;
                if ((int)gv.size() == Ang && gv[a].offset_valid) offsetU = gv[a].offsetU_pix;

                align.setStream(stream);
                align.pad(d_view_pw, d_padded, offsetU);

                // dump padded once + meta
                if (!dumped && a == DUMP_A) {
                    dump.dumpDeviceF32_A("pad", a, d_padded, (size_t)Nv * (size_t)paddedN, stream);

                    std::ostringstream ss;
                    ss << "a=" << a << "\n";
                    ss << "Nu=" << Nu << " Nv=" << Nv << " paddedN=" << paddedN << "\n";
                    ss << "offsetU_pix=" << offsetU << "\n";
                    ss << "start_u=" << align.lastStartU() << "\n";
                    ss << "du0_mm=" << du0 << "\n";
                    if ((int)gv.size() == Ang) ss << "dtheta=" << gv[a].dtheta << "\n";
                    dump.dumpText_A("meta", a, ss.str());
                }

                // (3) filter in-place on padded
                fm.setStream(stream);
                fm.apply(d_padded);

                // dump filtered padded once
                if (!dumped && a == DUMP_A) {
                    dump.dumpDeviceF32_A("fltpad", a, d_padded, (size_t)Nv * (size_t)paddedN, stream);
                }

                // (4) crop back
                align.crop(d_padded, d_view_flt);

                // dump crop once -> then disable
                if (!dumped && a == DUMP_A) {
                    dump.dumpDeviceF32_A("crop", a, d_view_flt, view_elems, stream);
                    dumped = true;
                    dump.setEnabled(false);
                }

                // pack into chunk
                float* d_slot = d_chunk + (size_t)i * view_elems;
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_slot, d_view_flt,
                    view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }

            // BP chunk
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

        // ============================================================
        // 6) Cleanup
        // ============================================================
        YK_CUDA_CHECK(cudaFree(d_geo));
        YK_CUDA_CHECK(cudaFree(d_dtheta));
        YK_CUDA_CHECK(cudaFree(d_view_in));
        YK_CUDA_CHECK(cudaFree(d_view_pw));
        YK_CUDA_CHECK(cudaFree(d_view_flt));
        YK_CUDA_CHECK(cudaFree(d_padded));
        YK_CUDA_CHECK(cudaFree(d_chunk));
    }




} // namespace YK
