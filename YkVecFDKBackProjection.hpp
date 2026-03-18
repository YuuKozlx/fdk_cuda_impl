#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cstdio>
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
    __device__ __forceinline__ bool project_uv_and_terms_derived(
        const SConeProjectionVec& g,
        const SFDKGeoParamPerView& gv,
        float3 P,
        float& u_pix,
        float& v_pix,
        float& denom_c)
    {
        // ray from source to voxel
        const float3 dir = f3_sub(P, g.src);

        // denom_c = (P-S) · ray0hat  (projection on central ray direction you defined)
        denom_c = f3_dot(dir, gv.ray0hat);

        // plane intersection path uses exported detector normal & DSD_n
        float3 nh_local = gv.nhat;
        float  DSD_n_local = gv.DSD_n;

        float denom_n = f3_dot(dir, nh_local);
        if (fabsf(denom_n) < 1e-8f) return false;
        if (fabsf(DSD_n_local) < 1e-8f) return false;

        // local flip only for intersection to enforce t>0
        if (DSD_n_local * denom_n < 0.0f) {
            nh_local = make_float3(-nh_local.x, -nh_local.y, -nh_local.z);
            denom_n = -denom_n;
            DSD_n_local = -DSD_n_local;
        }

        const float t = DSD_n_local / denom_n;
        if (t <= 0.0f) return false;

        // hit point on detector plane
        const float3 Q = f3_add(g.src, f3_mul(dir, t));
        const float3 D = f3_sub(Q, g.detS);

        // detector basis solve using exported cache:
        //   D = u*detU + v*detV  (detU,detV can be non-orthogonal)
        if (!gv.basis_valid || gv.invDetUV == 0.0f) return false;

        const float DU = f3_dot(D, g.detU);
        const float DV = f3_dot(D, g.detV);

        u_pix = (DU * gv.VV - DV * gv.UV) * gv.invDetUV;
        v_pix = (-DU * gv.UV + DV * gv.UU) * gv.invDetUV;
        return true;
    }

    // ============================================================
    // BP chunk kernel (uses derived table)
    //   weight: w = (SID^2) / (denom_c^2)
    //   where SID is stored in gv.SOD_mm by your chosen definition
    // ============================================================
    __global__ void fdk_vec_backproject_chunk_kernel(
        const float* __restrict__ views_chunk,            // [K*Nv*Nu]
        const SConeProjectionVec* __restrict__ d_geo,     // [Ang]
        const SFDKGeoParamPerView* __restrict__ d_gv,     // [Ang]
        float* __restrict__ vol,                          // [Nz*Ny*Nx]
        int Nx, int Ny, int Nz, float vox,
        int Nu, int Nv,
        int K, int base_a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= Nx || y >= Ny || z >= Nz) return;

        // NOTE: your coordinate convention kept as-is
        float3 P = make_float3(
            (y - (Ny - 1) * 0.5f) * vox,
            (x - (Nx - 1) * 0.5f) * vox,
            (z - (Nz - 1) * 0.5f) * vox
        );

        float acc = 0.0f;

        for (int i = 0; i < K; ++i) {
            int a = base_a + i;

            const SConeProjectionVec& g = d_geo[a];
            const SFDKGeoParamPerView& gv = d_gv[a];

            float u = 0.0f, v = 0.0f, denom_c = 0.0f;
            if (!project_uv_and_terms_derived(g, gv, P, u, v, denom_c)) continue;

            const float* view_i = views_chunk + (size_t)i * (size_t)Nv * (size_t)Nu;
            float p = YK::Interp::sample2d(view_i, Nu, Nv, u, v);

            // IMPORTANT:
            //   gv.SOD_mm stores SID by your definition (distance to plane through Z-axis with normal || central ray)
            const float SID = gv.SOD_mm;

            const float denom2 = denom_c * denom_c;
            if (denom2 < 1e-20f) continue;

            const float w = (SID * SID) / denom2;
            acc += p * w * gv.dtheta;
        }

        size_t vidx = (size_t)z * (size_t)Ny * (size_t)Nx + (size_t)y * (size_t)Nx + (size_t)x;
        vol[vidx] += acc;
    }

    // ============================================================
    // Streaming recon (minimal fixes + use derived offsets correctly)
    // ============================================================
    inline void fdk_vec_recon_streaming(
        const float* h_proj,                               // host [Ang*Nv*Nu]
        float* d_vol,                                      // device [Nz*Ny*Nx]
        const std::vector<SConeProjectionVec>& h_geo,       // host [Ang]
        SDimensions3D dims, float vox,
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
        // 0) CPU derived params (theta/dtheta/du/dv/SID/SDD/offset/ray0hat/nhat/basis-cache...)
        // ============================================================
        std::vector<SFDKGeoParamPerView> h_gv(dims.iPAng, SFDKGeoParamPerView{});
        GeoDerivedManagerVec derived;
        {
            GeoDerivedManagerVec::GeoDerivedOptions opt;
            opt.dtheta_eps = 1e-8f;
            (void)derived.build_geo_params(dims.iPU, dims.iPV, h_geo, h_gv);
        }

        // ============================================================
        // 1) Upload geo
        // ============================================================
        SConeProjectionVec* d_geo = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_geo, (size_t)dims.iPAng * sizeof(SConeProjectionVec)));
        YK_CUDA_CHECK(cudaMemcpyAsync(
            d_geo, h_geo.data(),
            (size_t)dims.iPAng * sizeof(SConeProjectionVec),
            cudaMemcpyHostToDevice, stream));

        // ============================================================
        // 2) Upload derived per-view table (gv)
        // ============================================================
        SFDKGeoParamPerView* d_gv = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_gv, (size_t)dims.iPAng * sizeof(SFDKGeoParamPerView)));
        YK_CUDA_CHECK(cudaMemcpyAsync(
            d_gv, h_gv.data(),
            (size_t)dims.iPAng * sizeof(SFDKGeoParamPerView),
            cudaMemcpyHostToDevice, stream));

        // ============================================================
        // 3) Managers
        // ============================================================
        PreweightManagerVec pw;

        AlignPadCropManagerVec align;
        const int host_chunk = 1;
        align.init(dims.iPU, dims.iPV, host_chunk, stream);
        const int paddedN = align.paddedN();

        FilterManager fm;
        // batch = Nv (每行做1D FFT)
        if (!fm.init(dims.iPU, paddedN, h_gv[0].du_mm, /*batch=*/dims.iPV, stream)) {
            YK_ASSERT(false && "FilterManager init failed");
        }

        // ============================================================
        // 4) Buffers
        // ============================================================
        const size_t view_elems = (size_t)dims.iPU * (size_t)dims.iPV;

        float* d_view_in = nullptr; // [Nv*Nu]
        float* d_view_pw = nullptr; // [Nv*Nu]
        float* d_view_flt = nullptr; // [Nv*Nu]
        float* d_padded = nullptr; // [Nv*paddedN]
        float* d_chunk = nullptr; // [Kchunk*Nv*Nu]

        YK_CUDA_CHECK(cudaMalloc(&d_view_in, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_pw, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_flt, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_padded, (size_t)dims.iPV * (size_t)paddedN * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_chunk, (size_t)Kchunk * view_elems * sizeof(float)));

        YK_CUDA_CHECK(cudaMemsetAsync(
            d_vol, 0,
            (size_t)dims.iVX * (size_t)dims.iVY * (size_t)dims.iVZ * sizeof(float),
            stream));

        // ============================================================
        // 5) Main loop
        // ============================================================
        for (int base = 0; base < dims.iPAng; base += Kchunk) {
            const int K = (base + Kchunk <= dims.iPAng) ? Kchunk : (dims.iPAng - base);

            for (int i = 0; i < K; ++i) {
                const int a = base + i;
                const float* h_view = h_proj + (size_t)a * view_elems;

                // H2D
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_view_in, h_view,
                    view_elems * sizeof(float),
                    cudaMemcpyHostToDevice, stream));

                // Single view: K=1, base_a=a
                pw.applyChunk(dims, d_view_in, d_view_pw, d_geo, d_gv, /*K=*/1, /*base_a=*/a, stream);

                // dump preweight once
                if (!dumped && a == DUMP_A) {
                    dump.dumpDeviceF32_A("pw", a, d_view_pw, view_elems, stream);
                }

                // (2) pad with per-view offsetU (pixel)
                std::vector<float> offsetU(1, 0.0f);
                offsetU[0] = h_gv[a].offsetU_pix;

                align.padChunk(d_view_pw, d_padded, offsetU);

                // (3) filter in-place on padded
                fm.setStream(stream);
                fm.apply(d_padded);

                // dump filtered padded once
                if (!dumped && a == DUMP_A) {
                    dump.dumpDeviceF32_A("fltpad", a, d_padded, (size_t)dims.iPV * (size_t)paddedN, stream);
                }

                // (4) crop back
                align.cropChunk(d_padded, d_view_flt);

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
            dim3 grid(
                (dims.iVX + block.x - 1) / block.x,
                (dims.iVY + block.y - 1) / block.y,
                (dims.iVZ + block.z - 1) / block.z);

            fdk_vec_backproject_chunk_kernel << <grid, block, 0, stream >> > (
                d_chunk, d_geo, d_gv, d_vol,
                dims.iVX, dims.iVY, dims.iVZ, vox,
                dims.iPU, dims.iPV,
                K, base);
            YK_CUDA_KERNEL_CHECK();
        }

        // ============================================================
        // 6) Cleanup
        // ============================================================
        YK_CUDA_CHECK(cudaFree(d_geo));
        YK_CUDA_CHECK(cudaFree(d_view_in));
        YK_CUDA_CHECK(cudaFree(d_view_pw));
        YK_CUDA_CHECK(cudaFree(d_view_flt));
        YK_CUDA_CHECK(cudaFree(d_padded));
        YK_CUDA_CHECK(cudaFree(d_chunk));
        YK_CUDA_CHECK(cudaFree(d_gv));
    }

} // namespace YK
