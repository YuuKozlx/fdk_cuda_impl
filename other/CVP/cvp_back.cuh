#pragma once
#include "cvp_geometry.cuh"
#include "global/YkMacro.hpp"
namespace cvp
{
    // ============================================================
    //  CVP Back Projector
    //  Strategy: pixel-driven (ray-driven), 1 thread = 1 pixel
    //
    //  Per pixel (m,n):
    //    1. Construct the ray from src through the pixel center
    //    2. Walk through the volume using Siddon stepping
    //    3. For each intersected voxel: accumulate
    //         (seg_len / r^2) * weight_scale * sinogram_value
    //       into the voxel via atomicAdd
    //
    //  Back-projection formula (in the FDK framework):
    //    delta_mu^{i,j,k} += E^{(m,n)} * weight(i,j,k,m,n)
    //  where weight = seg_len / r^2 / norm_factor,
    //  which is the approximate transpose of the forward projector.
    //
    //  Two weight modes are provided:
    //    fdk_weight = true  : standard FDK back-projection weight SDD^2 / r_pix^2
    //    fdk_weight = false : unweighted (identity weight = 1), for adjoint use
    // ============================================================

    // ============================================================
    //  Siddon ray traversal (device function)
    //
    //  Inputs:
    //    orig         - ray origin (typically the source position)
    //    dir          - ray direction (need not be normalized)
    //    vol_origin   - world position of voxel (0,0,0) center
    //    a1, a2, a3   - voxel sizes [mm]
    //    Nx, Ny, Nz   - volume dimensions
    //    d_volume     - device pointer to volume (accumulation target)
    //    sino_val     - sinogram value for this pixel
    //    weight_scale - common weight factor precomputed per pixel
    // ============================================================
    __device__ __forceinline__
        void siddon_trace(
            float3 orig,
            float3 dir,
            float3 vol_origin,
            float  a1, float a2, float a3,
            int    Nx, int Ny, int Nz,
            float* d_volume,
            float  sino_val,
            float  weight_scale)
    {
        // Voxel grid boundaries (faces of the bounding box)
        const float x_min = vol_origin.x - a1 * 0.5f;
        const float y_min = vol_origin.y - a2 * 0.5f;
        const float z_min = vol_origin.z - a3 * 0.5f;
        const float x_max = x_min + a1 * Nx;
        const float y_max = y_min + a2 * Ny;
        const float z_max = z_min + a3 * Nz;

        // Slab intersection of the ray with the bounding box
        float inv_dx = (fabsf(dir.x) > 1e-9f) ? 1.f / dir.x : 1e30f;
        float inv_dy = (fabsf(dir.y) > 1e-9f) ? 1.f / dir.y : 1e30f;
        float inv_dz = (fabsf(dir.z) > 1e-9f) ? 1.f / dir.z : 1e30f;

        float tx0 = (x_min - orig.x) * inv_dx;
        float tx1 = (x_max - orig.x) * inv_dx;
        float ty0 = (y_min - orig.y) * inv_dy;
        float ty1 = (y_max - orig.y) * inv_dy;
        float tz0 = (z_min - orig.z) * inv_dz;
        float tz1 = (z_max - orig.z) * inv_dz;

        if (tx0 > tx1) { float tmp = tx0; tx0 = tx1; tx1 = tmp; }
        if (ty0 > ty1) { float tmp = ty0; ty0 = ty1; ty1 = tmp; }
        if (tz0 > tz1) { float tmp = tz0; tz0 = tz1; tz1 = tmp; }

        float t_enter = fmaxf(fmaxf(tx0, ty0), tz0);
        float t_exit = fminf(fminf(tx1, ty1), tz1);

        if (t_enter >= t_exit || t_exit <= 0.f) return;  // no intersection
        t_enter = fmaxf(t_enter, 0.f);

        // Entry point into the volume
        const float3 p_enter = make_float3(
            orig.x + t_enter * dir.x,
            orig.y + t_enter * dir.y,
            orig.z + t_enter * dir.z);

        // Starting voxel index
        int ix = (int)floorf((p_enter.x - x_min) / a1);
        int iy = (int)floorf((p_enter.y - y_min) / a2);
        int iz = (int)floorf((p_enter.z - z_min) / a3);
        ix = max(0, min(Nx - 1, ix));
        iy = max(0, min(Ny - 1, iy));
        iz = max(0, min(Nz - 1, iz));

        // Step direction along each axis
        const int sx = (dir.x >= 0.f) ? 1 : -1;
        const int sy = (dir.y >= 0.f) ? 1 : -1;
        const int sz = (dir.z >= 0.f) ? 1 : -1;

        // Parameter t at the next voxel boundary along each axis
        const float next_x = x_min + (ix + (sx > 0 ? 1 : 0)) * a1;
        const float next_y = y_min + (iy + (sy > 0 ? 1 : 0)) * a2;
        const float next_z = z_min + (iz + (sz > 0 ? 1 : 0)) * a3;

        float t_next_x = (next_x - orig.x) * inv_dx;
        float t_next_y = (next_y - orig.y) * inv_dy;
        float t_next_z = (next_z - orig.z) * inv_dz;

        const float dt_x = fabsf(a1 * inv_dx);
        const float dt_y = fabsf(a2 * inv_dy);
        const float dt_z = fabsf(a3 * inv_dz);

        float t_cur = t_enter;

        // Upper bound on the number of voxels a ray can traverse
        const int max_steps = Nx + Ny + Nz;
        for (int step = 0; step < max_steps; ++step) {
            if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz) break;

            float t_boundary = fminf(fminf(t_next_x, t_next_y), t_next_z);
            t_boundary = fminf(t_boundary, t_exit);

            const float seg_len = t_boundary - t_cur;  // path length through this voxel
            if (seg_len > 0.f) {
                // Voxel center (used to compute r^2 from source)
                const float vcx = vol_origin.x + ix * a1;
                const float vcy = vol_origin.y + iy * a2;
                const float vcz = vol_origin.z + iz * a3;
                const float dx = vcx - orig.x;
                const float dy = vcy - orig.y;
                const float dz = vcz - orig.z;
                const float r2 = dx * dx + dy * dy + dz * dz;

                // Back-projection contribution: seg_len / r^2 * weight_scale * sino_val
                // This is the approximate transpose of the forward cut_vol / r^2 term.
                const float contrib = (seg_len / r2) * weight_scale * sino_val;

                atomicAdd(&d_volume[iz * Ny * Nx + iy * Nx + ix], contrib);
            }

            if (t_boundary >= t_exit) break;

            // Advance to the next voxel
            if (t_next_x <= t_next_y && t_next_x <= t_next_z) {
                t_cur = t_next_x; t_next_x += dt_x; ix += sx;
            }
            else if (t_next_y <= t_next_z) {
                t_cur = t_next_y; t_next_y += dt_y; iy += sy;
            }
            else {
                t_cur = t_next_z; t_next_z += dt_z; iz += sz;
            }
        }
    }

    // ============================================================
    //  Back-projection kernel (pixel-driven)
    //  1 thread = 1 pixel (m,n)
    // ============================================================
    __global__ void cvp_back_kernel(
        const float* __restrict__ sinogram,  // [N][M]
        float* volume,    // [Nz][Ny][Nx], written via atomicAdd
        SCVPViewCache             c,
        bool                      fdk_weight) // true = FDK weight, false = unweighted
    {
        const int m = blockIdx.x * blockDim.x + threadIdx.x;
        const int n = blockIdx.y * blockDim.y + threadIdx.y;
        if (m >= c.M || n >= c.N) return;

        const float sino_val = sinogram[n * c.M + m];
        if (sino_val == 0.f) return;

        // ----------------------------------------------------------
        // World position of pixel (m,n) center:
        //   pixel_center = detS + (m+0.5)*detU + (n+0.5)*detV
        // ----------------------------------------------------------
        const float3 pix_center = make_float3(
            c.detS.x + ((float)m + 0.5f) * c.detU.x + ((float)n + 0.5f) * c.detV.x,
            c.detS.y + ((float)m + 0.5f) * c.detU.y + ((float)n + 0.5f) * c.detV.y,
            c.detS.z + ((float)m + 0.5f) * c.detU.z + ((float)n + 0.5f) * c.detV.z);

        // Ray direction from source to pixel center
        const float3 ray_dir = d_sub(pix_center, c.src);

        // ----------------------------------------------------------
        // FDK back-projection weight: SDD^2 / r_pix^2
        // where r_pix is the distance from source to pixel center.
        // This is the standard Feldkamp/Parker cone-beam correction factor.
        // ----------------------------------------------------------
        float weight_scale = 1.f;
        if (fdk_weight) {
            const float off_u = ((float)m + 0.5f - (float)c.M * 0.5f) * c.du;
            const float off_v = ((float)n + 0.5f - (float)c.N * 0.5f) * c.dv;
            const float r2_pix = c.SDD * c.SDD + off_u * off_u + off_v * off_v;
            weight_scale = (c.SDD * c.SDD) / r2_pix;
        }

        // Siddon traversal: scatter back into volume
        siddon_trace(
            c.src, ray_dir,
            c.vol_origin, c.a1, c.a2, c.a3,
            c.Nx, c.Ny, c.Nz,
            volume, sino_val, weight_scale);
    }

    // ============================================================
    //  Launch wrapper
    // ============================================================
    YK_INLINE void launch_cvp_back(
        const float* d_sino,    // device [N][M] (filtered/weighted sinogram)
        float* d_volume,  // device [Nz][Ny][Nx], must be zeroed before call
        const SCVPViewCache& cache,
        bool                fdk_weight = true,
        cudaStream_t        stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((cache.M + 15) / 16, (cache.N + 15) / 16);
        cvp_back_kernel << <grid, block, 0, stream >> > (
            d_sino, d_volume, cache, fdk_weight);
    }
};