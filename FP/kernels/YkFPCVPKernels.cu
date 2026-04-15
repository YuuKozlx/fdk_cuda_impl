#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../../global/YkMacro.hpp"
#include "YkFPCVPHelpers.cuh"


namespace YK
{
    namespace Fp {
        namespace detail {
            // ============================================================
       //  CVP Forward Projector
       //  Strategy: voxel-driven, 1 thread = 1 voxel
       //
       //  Per voxel:
       //    1. Project voxel center onto the detector -> continuous pixel coords (pu, pv)
       //    2. Estimate AABB footprint on detector using magnified half-voxel size
       //    3. For each covered pixel compute 2D overlap fraction * voxel volume = cut volume
       //    4. Accumulate mu/r^2 * cut_volume via atomicAdd into sinogram
       //
       //  The normalization factor f^2 / (a^{mn} * cos^3(theta)) is applied in a
       //  separate pass (see cvp_normalize_kernel).
       //
       //  Sinogram layout: [N][M]  (row = V direction, col = U direction)
       // ============================================================

            __global__ void cvp_forward_kernel(
                const float* __restrict__ volume,   // [Nz][Ny][Nx], z-major (slowest z)
                float* sinogram, // [N][M], written via atomicAdd
                SCVPViewCache             c)
            {
                const int ix = blockIdx.x * blockDim.x + threadIdx.x;
                const int iy = blockIdx.y * blockDim.y + threadIdx.y;
                const int iz = blockIdx.z * blockDim.z + threadIdx.z;

                if (ix >= c.Nx || iy >= c.Ny || iz >= c.Nz) return;

                const float mu = volume[iz * c.Ny * c.Nx + iy * c.Nx + ix];
                if (mu == 0.f) return;  // early exit for sparse volumes

                // ----------------------------------------------------------
                // 1. Voxel center in world coordinates
                // ----------------------------------------------------------
                const float3 vc = make_float3(
                    c.vol_origin.x + ix * c.a1,
                    c.vol_origin.y + iy * c.a2,
                    c.vol_origin.z + iz * c.a3);

                // ----------------------------------------------------------
                // 2. Perspective projection onto the detector
                //    Ray direction: d = vc - src
                //    Parameter t such that src + t*d lies on the detector plane:
                //      dot(t*d, srcCR) = SDD  =>  t = SDD / dot(d, srcCR)
                // ----------------------------------------------------------
                const float3 d = d_sub(vc, c.src);
                const float  d_dot_n = d_dot(d, c.srcCR);

                // Skip voxels behind the source or parallel to the detector normal
                if (d_dot_n <= 0.f) return;

                const float  t = c.SDD / d_dot_n;       // magnification factor
                const float3 hit = d_fma(t, d, c.src);    // hit point on detector plane
                const float3 dh = d_sub(hit, c.detS);    // offset from pixel (0,0) origin

                // Continuous pixel coordinates (origin = pixel (0,0) corner)
                const float pu = d_dot(dh, c.detU_n) * c.inv_du + 0.5f;
                const float pv = d_dot(dh, c.detV_n) * c.inv_dv + 0.5f;

                // ----------------------------------------------------------
                // 3. Voxel footprint AABB on detector
                //    Approximation: project voxel center +/- magnified half-voxel size.
                //    half_u = (a1/2) * t / du  (world -> detector magnification, then -> pixels)
                //    Note: t already encodes SDD/dist_along_ray, so dividing by du converts
                //    the world-space half-width to pixel units.
                //    This is an axis-aligned approximation; error is small when the voxel
                //    footprint is close to isotropic on the detector.
                // ----------------------------------------------------------
                const float half_u = 0.5f * c.a1 * t * c.inv_du;
                const float half_v = 0.5f * c.a2 * t * c.inv_dv;

                // Integer pixel range covered by the footprint
                const int m_lo = max(0, (int)floorf(pu - half_u));
                const int m_hi = min(c.M - 1, (int)floorf(pu + half_u));
                const int n_lo = max(0, (int)floorf(pv - half_v));
                const int n_hi = min(c.N - 1, (int)floorf(pv + half_v));

                if (m_lo > m_hi || n_lo > n_hi) return;  // footprint entirely outside detector

                // ----------------------------------------------------------
                // 4. 1/r^2 weight  (paper eq. vox: contribution = mu/r^2 * cut_volume)
                // ----------------------------------------------------------
                const float r2 = d_dot(d, d);
                const float w_mu_r2 = mu / r2;

                // Full voxel volume [mm^3]
                const float vox_vol = c.a1 * c.a2 * c.a3;

                // ----------------------------------------------------------
                // 5. Iterate over covered pixels, compute 2D overlap, scatter-add
                //
                //    cut_volume = vox_vol * overlap_fraction
                //    overlap_fraction = (ou_pix / footprint_u) * (ov_pix / footprint_v)
                //    footprint area in pixel units = 2*half_u * 2*half_v
                //
                //    Precompute: scale = vox_vol / footprint_area  (pixel units denominator)
                //    contrib    = mu/r^2 * scale * ou_pix * ov_pix
                //
                //    Pixel (m,n) occupies the interval [m, m+1) x [n, n+1) in pixel coords.
                // ----------------------------------------------------------
                const float footprint_area = (2.f * half_u) * (2.f * half_v);
                const float scale = vox_vol / fmaxf(footprint_area, 1e-10f);

                for (int n = n_lo; n <= n_hi; ++n) {
                    const float ov_pix = fminf((float)(n + 1), pv + half_v)
                        - fmaxf((float)n, pv - half_v);
                    if (ov_pix <= 0.f) continue;

                    for (int m = m_lo; m <= m_hi; ++m) {
                        const float ou_pix = fminf((float)(m + 1), pu + half_u)
                            - fmaxf((float)m, pu - half_u);
                        if (ou_pix <= 0.f) continue;

                        // cut_volume = vox_vol * (ou_pix * ov_pix) / footprint_area
                        const float contrib = w_mu_r2 * scale * ou_pix * ov_pix;

                        atomicAdd(&sinogram[n * c.M + m], contrib);
                    }
                }
            }

            // ============================================================
            //  Normalization pass  (paper eq. cvp: factor f^2 / (a^{mn} * cos^3(theta)))
            //  One thread per pixel.
            //
            //  cos_theta[n*M+m] : cosine of the cone angle for each pixel center,
            //                     precomputed by cvp_precompute_costheta.
            //  pixel_area       : detector pixel area = du * dv [mm^2]
            // ============================================================
            __global__ void cvp_normalize_kernel(
                float* sinogram,    // [N][M] in-place
                const float* cos_theta,   // [N][M] per-pixel cos(theta)
                float        SDD,
                float        pixel_area,  // du * dv [mm^2]
                int          M, int N)
            {
                const int m = blockIdx.x * blockDim.x + threadIdx.x;
                const int n = blockIdx.y * blockDim.y + threadIdx.y;
                if (m >= M || n >= N) return;

                const int   idx = n * M + m;
                const float cos_t = cos_theta[idx];
                // factor = f^2 / (pixel_area * cos^3(theta))
                const float norm = (SDD * SDD) / (pixel_area * cos_t * cos_t * cos_t);
                sinogram[idx] *= norm;
            }

            // ============================================================
            //  Precompute cos_theta map (call once per view before the main kernel)
            //
            //  cos_theta(m,n) = dot(ray_direction(m,n), srcCR)
            //                 = SDD / sqrt(SDD^2 + off_u^2 + off_v^2)
            //  where off_u, off_v are the physical offsets of pixel center
            //  from the detector geometric center [mm].
            // ============================================================
            __global__ void cvp_precompute_costheta(
                float* cos_theta,  // [N][M] output
                SCVPViewCache c)
            {
                const int m = blockIdx.x * blockDim.x + threadIdx.x;
                const int n = blockIdx.y * blockDim.y + threadIdx.y;
                if (m >= c.M || n >= c.N) return;

                // Physical offset of pixel (m,n) center from detector geometric center
                const float off_u = ((float)m + 0.5f - (float)c.M * 0.5f) * c.du;
                const float off_v = ((float)n + 0.5f - (float)c.N * 0.5f) * c.dv;

                // Distance from source to pixel center:
                //   r = sqrt(SDD^2 + off_u^2 + off_v^2)
                //   cos(theta) = SDD / r
                const float r = sqrtf(c.SDD * c.SDD + off_u * off_u + off_v * off_v);
                cos_theta[n * c.M + m] = c.SDD / r;
            }


        }; // namespace detail
        // ============================================================
        //  Launch wrapper  (3-step pipeline per view)
        // ============================================================
        void fp_cvp_launch(
            const float* d_vol,
            float* d_sino,
            const std::vector<YK::SConeProjGeomVec>& h_views,
            const YK::SVolGeom& g,
            int Na, int Nu, int Nv,
            cudaStream_t stream)
        {
            float* d_cos_theta = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_cos_theta, (size_t)Nv * Nu * sizeof(float)));

            for (int a = 0; a < Na; ++a)
            {
                const auto cache = make_view_cache(h_views[a], Nu, Nv, g);
                //printf("vol_origin=(%.3f, %.3f, %.3f)\n",
                //    cache.vol_origin.x, cache.vol_origin.y, cache.vol_origin.z);
                //printf("src=(%.3f, %.3f, %.3f)\n",
                //    cache.src.x, cache.src.y, cache.src.z);
                float* d_s = d_sino + (size_t)a * Nv * Nu;

                // Step 1: cos_theta
                {
                    dim3 block(16, 16);
                    dim3 grid((Nu + 15) / 16, (Nv + 15) / 16);
                    detail::cvp_precompute_costheta << <grid, block, 0, stream >> > (d_cos_theta, cache);
                }
                // Step 2: scatter
                {
                    dim3 block(8, 8, 8);
                    dim3 grid((cache.Nx + 7) / 8, (cache.Ny + 7) / 8, (cache.Nz + 7) / 8);
                    detail::cvp_forward_kernel << <grid, block, 0, stream >> > (d_vol, d_s, cache);
                }
                // Step 3: normalize
                {
                    dim3 block(16, 16);
                    dim3 grid((Nu + 15) / 16, (Nv + 15) / 16);
                    float pixel_area = cache.du * cache.dv;
                    detail::cvp_normalize_kernel << <grid, block, 0, stream >> > (
                        d_s, d_cos_theta, cache.SDD, pixel_area, Nu, Nv);
                }
            }

            YK_CUDA_CHECK(cudaFree(d_cos_theta));
        }
    };

};
