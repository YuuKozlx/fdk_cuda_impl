#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#include "YkFDKPreWeight.hpp"
#include "YkGlobals.h"
#include "YkSampling2D.hpp"

namespace YK {

    // ============================================================
    // bilinear sample for one view (Nv x Nu), u-fastest
    // ============================================================
//__device__ __forceinline__ float bilinear_sample_2d(
//        const float* __restrict__ img, int Nu, int Nv, float u, float v)
//    {
//        if (u < 0.0f || u >(float)(Nu - 1) || v < 0.0f || v >(float)(Nv - 1)) return 0.0f;
//
//        int u0 = (int)floorf(u);
//        int v0 = (int)floorf(v);
//        int u1 = (u0 + 1 < Nu) ? (u0 + 1) : u0;
//        int v1 = (v0 + 1 < Nv) ? (v0 + 1) : v0;
//
//        float fu = u - (float)u0;
//        float fv = v - (float)v0;
//
//        float p00 = img[v0 * Nu + u0];
//        float p10 = img[v0 * Nu + u1];
//        float p01 = img[v1 * Nu + u0];
//        float p11 = img[v1 * Nu + u1];
//
//        float p0 = p00 + fu * (p10 - p00);
//        float p1 = p01 + fu * (p11 - p01);
//        return p0 + fv * (p1 - p0);
//    }

    // ============================================================
    // crop padded [Nv*paddedN] -> [Nv*Nu] into dst (u-fastest)
    // ============================================================
    //__global__ void kernel_crop_u_2d(
    //    const float* __restrict__ src, // [Nv*paddedN]
    //    float* __restrict__ dst,       // [Nv*Nu]
    //    int Nu, int Nv, int paddedN, int start_u)
    //{
    //    int u = blockIdx.x * blockDim.x + threadIdx.x;
    //    int v = blockIdx.y * blockDim.y + threadIdx.y;
    //    if (u >= Nu || v >= Nv) return;

    //    int su = start_u + u;
    //    float val = 0.0f;
    //    if (su >= 0 && su < paddedN) val = src[v * paddedN + su];
    //    dst[v * Nu + u] = val;
    //}

    // ============================================================
    // Backproject a CHUNK of filtered views into volume (accumulate)
    // - views_chunk layout: [K][Nv][Nu] (contiguous), u-fastest
    // - angle indices: base_a + i (i in [0,K))
    // - cos/sin tables on device: d_cos[a], d_sin[a]
    // ============================================================
    __global__ void fdk_backproject_chunk_kernel(
        const float* __restrict__ views_chunk, // [K*Nv*Nu]
        float* __restrict__ vol,               // [Nz*Ny*Nx]
        int Nx, int Ny, int Nz, float vox,
        int Nu, int Nv, float du, float dv,
        int K, int base_a,
        const float* __restrict__ d_cos,
        const float* __restrict__ d_sin,
        float SID, float SDD,
        float offsetU, float offsetV,
        float dtheta)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= Nx || y >= Ny || z >= Nz) return;

        float X = (x - (Nx - 1) * 0.5f) * vox;
        float Y = (y - (Ny - 1) * 0.5f) * vox;
        float Z = (z - (Nz - 1) * 0.5f) * vox;

        // geometry in angle-local frame
        const float x_det = -(SDD - SID);
        const float u0 = (Nu - 1) * 0.5f + offsetU;
        const float v0 = (Nv - 1) * 0.5f + offsetV;

        float acc = 0.0f;

        for (int i = 0; i < K; ++i) {
            int a = base_a + i;
            float c = d_cos[a];
            float s = d_sin[a];

            float xr = c * X + s * Y;
            float yr = -s * X + c * Y;
            float zr = Z;

            float dx = xr - SID;
            float dy = yr;
            float dz = zr;

            if (fabsf(dx) < 1e-6f) continue;

            float t = (x_det - SID) / dx;
            if (t <= 0.0f) continue;

            float y_det = t * dy;
            float z_det = t * dz;

            float u = y_det / du + u0;
            float v = z_det / dv + v0;

            const float* view_i = views_chunk + (size_t)i * Nv * Nu;
            float p = YK::Interp::sample2d(view_i, Nu, Nv, u, v);

            float denom = (SID - xr);
            if (fabsf(denom) < 1e-6f) continue;
            float w = (SID * SID) / (denom * denom);

            acc += p * w;
        }

        // accumulate into volume (no atomic needed: one thread writes one voxel)
        size_t vidx = (size_t)z * Ny * Nx + (size_t)y * Nx + x;
        vol[vidx] += acc * dtheta;
    }

    // ============================================================
    // Helper: create device cos/sin table once
    // ============================================================
    inline void create_angle_table(int Ang, float** d_cos, float** d_sin, cudaStream_t stream)
    {
        std::vector<float> hcos(Ang), hsin(Ang);
        const float two_pi = 2.0f * 3.14159265358979323846f;
        for (int a = 0; a < Ang; ++a) {
            float theta = two_pi * (float)a / (float)Ang;
            hcos[a] = cosf(theta);
            hsin[a] = sinf(theta);
        }
        YK_CUDA_CHECK(cudaMalloc(d_cos, Ang * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(d_sin, Ang * sizeof(float)));
        YK_CUDA_CHECK(cudaMemcpyAsync(*d_cos, hcos.data(), Ang * sizeof(float), cudaMemcpyHostToDevice, stream));
        YK_CUDA_CHECK(cudaMemcpyAsync(*d_sin, hsin.data(), Ang * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    // ============================================================
    // Core streaming FDK: preweight + filter + BP in chunks (no full proj store)
    // Input projection layout assumed A-V-U contiguous on host:
    //   h_proj[ a*(Nv*Nu) + v*Nu + u ]
    //
    // Output: d_vol on device (caller alloc) or host copy after sync
    // ============================================================
    inline void fdk_recon_streaming(
        const float* h_proj,                   // host projections [Ang*Nv*Nu]
        float* d_vol,                          // device volume [Nz*Ny*Nx]
        int Nu, int Nv, int Ang,
        int Nx, int Ny, int Nz, float vox,
        float SID, float SDD, float du, float dv,
        float offsetU, float offsetV,           // pixel
        int Kchunk,                             // e.g. 8 or 16
        cudaStream_t stream)
    {
        // 1) angle table (device persistent for this call)
        float* d_cos = nullptr;
        float* d_sin = nullptr;
        create_angle_table(Ang, &d_cos, &d_sin, stream);

        const float dtheta = (2.0f * 3.14159265358979323846f) / (float)Ang;

        // 2) init preweight manager (batch=1: one view each time)
        PreweightManager pw;
        pw.init(Nu, Nv, 1, du, dv, /*DSD=*/SDD, offsetU, offsetV, /*power=*/1, stream);

        // 3) init filter manager for 1D row filtering (batch = Nv, length = Nu)
        FilterManager fm;
        fm.init(Nu, du, /*batch=*/Nv, stream);
        int paddedN = fm.getPaddedN();

        // crop alignment (offsetX = 0 for filter center; adapt if you have axis offset)
        const float offsetX = 0.0f;
        const float axis_idx = (Nu - 1) * 0.5f + offsetX;
        const int start_u = (int)lrintf(paddedN * 0.5f - axis_idx);

        // 4) allocate per-view buffers (reused)
        const size_t view_elems = (size_t)Nu * Nv;

        float* d_view_in = nullptr; // [Nv*Nu]
        float* d_view_pw = nullptr; // [Nv*Nu]
        float* d_view_flt = nullptr; // [Nv*Nu] (cropped)
        float* d_padded = nullptr; // [Nv*paddedN]
        YK_CUDA_CHECK(cudaMalloc(&d_view_in, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_pw, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_view_flt, view_elems * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_padded, (size_t)Nv * paddedN * sizeof(float)));

        // 5) allocate chunk buffer: [Kchunk][Nv][Nu]
        float* d_chunk = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_chunk, (size_t)Kchunk * view_elems * sizeof(float)));

        // 6) clear volume once
        YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0, (size_t)Nx * Ny * Nz * sizeof(float), stream));

        // 7) process angles in chunks
        for (int base = 0; base < Ang; base += Kchunk) {
            int K = (base + Kchunk <= Ang) ? Kchunk : (Ang - base);

            // --- build K filtered views into d_chunk ---
            for (int i = 0; i < K; ++i) {
                int a = base + i;
                const float* h_view = h_proj + (size_t)a * view_elems;

                // H2D one view
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_view_in, h_view, view_elems * sizeof(float),
                    cudaMemcpyHostToDevice, stream));

                // preweight cos
                pw.setStream(stream);
                pw.apply(d_view_in, d_view_pw);

                // pad rows (batch=Nv) -> d_padded
                fm.setStream(stream);
                dim3 b1(256, 1);
                dim3 g1((paddedN + 255) / 256, Nv);
                YK::_kernel_pad_with_offset << <g1, b1, 0, stream >> > (
                    d_view_pw, d_padded, Nu, Nv, paddedN, offsetX);
                YK_CUDA_KERNEL_CHECK();

                // filter in-place on padded
                fm.apply(d_padded);

                YK::Util::crop_u_2d(d_padded, d_view_flt, Nu, Nv, paddedN, start_u, stream);
                YK_CUDA_KERNEL_CHECK();

                // copy into chunk slot (device-to-device)
                float* d_slot = d_chunk + (size_t)i * view_elems;
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_slot, d_view_flt, view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }

            // --- backproject this chunk (accumulate into d_vol) ---
            dim3 block(8, 8, 4);
            dim3 grid((Nx + block.x - 1) / block.x,
                (Ny + block.y - 1) / block.y,
                (Nz + block.z - 1) / block.z);

            fdk_backproject_chunk_kernel << <grid, block, 0, stream >> > (
                d_chunk, d_vol,
                Nx, Ny, Nz, vox,
                Nu, Nv, du, dv,
                K, base,
                d_cos, d_sin,
                SID, SDD,
                offsetU, offsetV,
                dtheta);
            YK_CUDA_KERNEL_CHECK();
        }

        // cleanup (for this call)
        YK_CUDA_CHECK(cudaFree(d_cos));
        YK_CUDA_CHECK(cudaFree(d_sin));
        YK_CUDA_CHECK(cudaFree(d_view_in));
        YK_CUDA_CHECK(cudaFree(d_view_pw));
        YK_CUDA_CHECK(cudaFree(d_view_flt));
        YK_CUDA_CHECK(cudaFree(d_padded));
        YK_CUDA_CHECK(cudaFree(d_chunk));
    }

} // namespace YK
