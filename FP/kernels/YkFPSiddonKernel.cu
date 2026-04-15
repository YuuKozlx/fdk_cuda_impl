#include "YkFPSiddonLaunch.cuh"
#include <algorithm>

namespace YK {
    namespace Fp {
        namespace detail {

            __global__ void siddon_fp_kernel(
                const float* __restrict__ d_vol,
                float* d_sino,
                const SConeProjGeomVec* d_views,
                float3 vol_origin,
                float  vox_x, float vox_y, float vox_z,
                int    Nx, int Ny, int Nz,
                int    Nu, int Nv, int K,
                bool   accumulate)
            {
                const int iu = blockIdx.x * blockDim.x + threadIdx.x;
                const int iv = blockIdx.y * blockDim.y + threadIdx.y;
                const int ia = blockIdx.z;
                if (iu >= Nu || iv >= Nv || ia >= K) return;

                const SConeProjGeomVec& v = d_views[ia];

                // 探测器像素中心世界坐标
                const float3 detC = {
                    v.detS.x + (iu + 0.5f) * v.detU.x + (iv + 0.5f) * v.detV.x,
                    v.detS.y + (iu + 0.5f) * v.detU.y + (iv + 0.5f) * v.detV.y,
                    v.detS.z + (iu + 0.5f) * v.detU.z + (iv + 0.5f) * v.detV.z
                };

                // ray 方向向量（世界坐标，src → det）
                const float3 ray = {
                    detC.x - v.src.x,
                    detC.y - v.src.y,
                    detC.z - v.src.z
                };

                // 体积 AABB（世界坐标）
                const float x0 = vol_origin.x, x1 = x0 + Nx * vox_x;
                const float y0 = vol_origin.y, y1 = y0 + Ny * vox_y;
                const float z0 = vol_origin.z, z1 = z0 + Nz * vox_z;

                // ray 与 AABB 的参数化交点 [tmin, tmax]
                float tmin = 0.f, tmax = 1.f;

#define SLAB(r, s, b0, b1)                          \
    if (fabsf(r) > 1e-8f) {                         \
        float ta = (b0 - s) / r;                    \
        float tb = (b1 - s) / r;                    \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; } \
        tmin = fmaxf(tmin, ta);                     \
        tmax = fminf(tmax, tb);                     \
    } else if (s < b0 || s > b1) return;

                SLAB(ray.x, v.src.x, x0, x1)
                    SLAB(ray.y, v.src.y, y0, y1)
                    SLAB(ray.z, v.src.z, z0, z1)
#undef SLAB

                    if (tmin >= tmax) return;

                // 入射点体素坐标
                const float ex = v.src.x + tmin * ray.x;
                const float ey = v.src.y + tmin * ray.y;
                const float ez = v.src.z + tmin * ray.z;

                int ix = (int)floorf((ex - vol_origin.x) / vox_x);
                int iy = (int)floorf((ey - vol_origin.y) / vox_y);
                int iz = (int)floorf((ez - vol_origin.z) / vox_z);
                ix = max(0, min(Nx - 1, ix));
                iy = max(0, min(Ny - 1, iy));
                iz = max(0, min(Nz - 1, iz));

                const int stepX = (ray.x >= 0.f) ? 1 : -1;
                const int stepY = (ray.y >= 0.f) ? 1 : -1;
                const int stepZ = (ray.z >= 0.f) ? 1 : -1;

                // 到下一条体素边界的 t 值
                const float bx = vol_origin.x + (ix + (stepX > 0 ? 1 : 0)) * vox_x;
                const float by = vol_origin.y + (iy + (stepY > 0 ? 1 : 0)) * vox_y;
                const float bz = vol_origin.z + (iz + (stepZ > 0 ? 1 : 0)) * vox_z;

                float tX = (fabsf(ray.x) > 1e-8f) ? (bx - v.src.x) / ray.x : 1e30f;
                float tY = (fabsf(ray.y) > 1e-8f) ? (by - v.src.y) / ray.y : 1e30f;
                float tZ = (fabsf(ray.z) > 1e-8f) ? (bz - v.src.z) / ray.z : 1e30f;

                const float dtX = (fabsf(ray.x) > 1e-8f) ? fabsf(vox_x / ray.x) : 1e30f;
                const float dtY = (fabsf(ray.y) > 1e-8f) ? fabsf(vox_y / ray.y) : 1e30f;
                const float dtZ = (fabsf(ray.z) > 1e-8f) ? fabsf(vox_z / ray.z) : 1e30f;

                const float ray_len = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

                float fVal = 0.f;
                float t_cur = tmin;

                while (t_cur < tmax)
                {
                    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
                        break;

                    const float t_next = fminf(fminf(tX, tY), fminf(tZ, tmax));
                    fVal += d_vol[(size_t)iz * Ny * Nx + iy * Nx + ix]
                        * (t_next - t_cur) * ray_len;

                    t_cur = t_next;

                    if (tX <= tY && tX <= tZ) { ix += stepX; tX += dtX; }
                    else if (tY <= tZ) { iy += stepY; tY += dtY; }
                    else { iz += stepZ; tZ += dtZ; }
                }

                const size_t idx = ((size_t)ia * Nv + iv) * Nu + iu;
                if (accumulate) d_sino[idx] += fVal;
                else            d_sino[idx] = fVal;
            }



        } // namespace detail

        void siddon_launchGroup(
            const float* d_vol,
            float* d_sino,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            bool accumulate,
            cudaStream_t stream)
        {
            if (K <= 0) return;

            // blockIdx.z 最大值受 GPU 限制（通常 65535），K=360 完全没问题
            dim3 block(16, 16, 1);
            dim3 grid((Nu + 15) / 16, (Nv + 15) / 16, K);

            detail::siddon_fp_kernel << <grid, block, 0, stream >> > (
                d_vol, d_sino, d_views,
                g.origin(),
                g.vox_x, g.vox_y, g.vox_z,
                g.Nx, g.Ny, g.Nz,
                Nu, Nv, K,
                accumulate);
        }
    } // namespace Fp
} // namespace YK