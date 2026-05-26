// YkBPSiddonKernel.cu
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "BP/kernels/YkBPSiddonLaunch.cuh"

namespace YK {
    namespace Bp {
        namespace detail {

            __global__ void siddon_bp_kernel(
                const float* __restrict__ d_sino,
                float* __restrict__ d_vol,
                const SConeProjGeomVec* d_views,
                float3 vol_origin,
                float  vox_x, float vox_y, float vox_z,
                int    Nx, int Ny, int Nz,
                int    Nu, int Nv, int K)
            {
                const int iu = blockIdx.x * blockDim.x + threadIdx.x;
                const int iv = blockIdx.y * blockDim.y + threadIdx.y;
                const int ia = blockIdx.z;
                if (iu >= Nu || iv >= Nv || ia >= K) return;

                const SConeProjGeomVec& v = d_views[ia];

                const float3 detC = {
                    v.detS.x + (iu + 0.5f) * v.detU.x + (iv + 0.5f) * v.detV.x,
                    v.detS.y + (iu + 0.5f) * v.detU.y + (iv + 0.5f) * v.detV.y,
                    v.detS.z + (iu + 0.5f) * v.detU.z + (iv + 0.5f) * v.detV.z
                };

                const float3 ray = {
                    detC.x - v.src.x,
                    detC.y - v.src.y,
                    detC.z - v.src.z
                };
                const float ray_len = sqrtf(
                    ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

                // AABB 裁剪
                float x0 = vol_origin.x, x1 = x0 + Nx * vox_x;
                float y0 = vol_origin.y, y1 = y0 + Ny * vox_y;
                float z0 = vol_origin.z, z1 = z0 + Nz * vox_z;
                float tmin = 0.f, tmax = 1.f;

#define SLAB(r, s, b0, b1) \
    if (fabsf(r) > 1e-8f) { \
        float ta = (b0 - s) / r; \
        float tb = (b1 - s) / r; \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; } \
        tmin = fmaxf(tmin, ta); \
        tmax = fminf(tmax, tb); \
    } else if (s < b0 || s > b1) return;

                SLAB(ray.x, v.src.x, x0, x1)
                    SLAB(ray.y, v.src.y, y0, y1)
                    SLAB(ray.z, v.src.z, z0, z1)
#undef SLAB

                    if (tmin >= tmax) return;

                // 入射点体素索引
                float ex = v.src.x + tmin * ray.x;
                float ey = v.src.y + tmin * ray.y;
                float ez = v.src.z + tmin * ray.z;
                int ix = (int)floorf((ex - vol_origin.x) / vox_x);
                int iy = (int)floorf((ey - vol_origin.y) / vox_y);
                int iz = (int)floorf((ez - vol_origin.z) / vox_z);
                ix = max(0, min(Nx - 1, ix));
                iy = max(0, min(Ny - 1, iy));
                iz = max(0, min(Nz - 1, iz));

                const int stepX = (ray.x >= 0.f) ? 1 : -1;
                const int stepY = (ray.y >= 0.f) ? 1 : -1;
                const int stepZ = (ray.z >= 0.f) ? 1 : -1;

                float bx = vol_origin.x + (ix + (stepX > 0 ? 1 : 0)) * vox_x;
                float by = vol_origin.y + (iy + (stepY > 0 ? 1 : 0)) * vox_y;
                float bz = vol_origin.z + (iz + (stepZ > 0 ? 1 : 0)) * vox_z;

                float tX = (fabsf(ray.x) > 1e-8f) ? (bx - v.src.x) / ray.x : 1e30f;
                float tY = (fabsf(ray.y) > 1e-8f) ? (by - v.src.y) / ray.y : 1e30f;
                float tZ = (fabsf(ray.z) > 1e-8f) ? (bz - v.src.z) / ray.z : 1e30f;

                const float dtX = (fabsf(ray.x) > 1e-8f) ? fabsf(vox_x / ray.x) : 1e30f;
                const float dtY = (fabsf(ray.y) > 1e-8f) ? fabsf(vox_y / ray.y) : 1e30f;
                const float dtZ = (fabsf(ray.z) > 1e-8f) ? fabsf(vox_z / ray.z) : 1e30f;

                const size_t sino_idx = ((size_t)ia * Nv + iv) * Nu + iu;
                const float proj_val = d_sino[sino_idx];

                float t_cur = tmin;
                while (t_cur < tmax)
                {
                    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
                        break;

                    const float t_next = fminf(fminf(tX, tY), fminf(tZ, tmax));
                    const float seg_len = (t_next - t_cur) * ray_len;

                    atomicAdd(&d_vol[(size_t)iz * Ny * Nx + (size_t)iy * Nx + ix],
                        proj_val * seg_len);

                    t_cur = t_next;

                    if (tX <= tY && tX <= tZ) { ix += stepX; tX += dtX; }
                    else if (tY <= tZ) { iy += stepY; tY += dtY; }
                    else { iz += stepZ; tZ += dtZ; }
                }
            }

        } // namespace detail

        void bp_siddon_launch(
            const float* d_sino,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            cudaStream_t            stream)
        {
            if (K <= 0) return;

            dim3 block(16, 16, 1);
            dim3 grid((Nu + 15) / 16, (Nv + 15) / 16, K);

            detail::siddon_bp_kernel << <grid, block, 0, stream >> > (
                d_sino, d_vol, d_views,
                g.origin(),
                g.vox_x, g.vox_y, g.vox_z,
                g.Nx, g.Ny, g.Nz,
                Nu, Nv, K);
        }

    } // namespace Bp
} // namespace YK


namespace YK {
    namespace Bp {
        namespace detail {

            __global__ void siddon_bp_voxel_kernel(
                const float* __restrict__ d_sino,
                float* __restrict__       d_vol,
                const SConeProjGeomVec* d_views,
                float3 vol_origin,
                float  vox_x, float vox_y, float vox_z,
                int    Nx, int Ny, int Nz,
                int    Nu, int Nv, int K)
            {
                const int ix = blockIdx.x * blockDim.x + threadIdx.x;
                const int iy = blockIdx.y * blockDim.y + threadIdx.y;
                const int iz = blockIdx.z * blockDim.z + threadIdx.z;
                if (ix >= Nx || iy >= Ny || iz >= Nz) return;

                // 体素中心世界坐标
                const float cx = vol_origin.x + (ix + 0.5f) * vox_x;
                const float cy = vol_origin.y + (iy + 0.5f) * vox_y;
                const float cz = vol_origin.z + (iz + 0.5f) * vox_z;

                // 体素 AABB
                const float vx0 = vol_origin.x + ix * vox_x;
                const float vx1 = vol_origin.x + (ix + 1) * vox_x;
                const float vy0 = vol_origin.y + iy * vox_y;
                const float vy1 = vol_origin.y + (iy + 1) * vox_y;
                const float vz0 = vol_origin.z + iz * vox_z;
                const float vz1 = vol_origin.z + (iz + 1) * vox_z;

                float acc = 0.f;

                for (int ia = 0; ia < K; ++ia)
                {
                    const SConeProjGeomVec& v = d_views[ia];

                    // ── Step 1：体素中心投影到探测器，找 (iu, iv) ──────────────
                    // 探测器平面法向量 n = detU × detV
                    const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
                    const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
                    const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;

                    // src → 体素中心方向
                    const float dx = cx - v.src.x;
                    const float dy = cy - v.src.y;
                    const float dz = cz - v.src.z;

                    const float denom = dx * nx + dy * ny + dz * nz;
                    if (fabsf(denom) < 1e-8f) continue;

                    // src + t_det * (cx-src) 落在探测器平面
                    const float ex = v.detS.x - v.src.x;
                    const float ey = v.detS.y - v.src.y;
                    const float ez = v.detS.z - v.src.z;
                    const float t_det = (ex * nx + ey * ny + ez * nz) / denom;
                    if (t_det <= 0.f) continue;

                    // 探测器平面上的交点，相对于 detS
                    const float px = v.src.x + t_det * dx - v.detS.x;
                    const float py = v.src.y + t_det * dy - v.detS.y;
                    const float pz = v.src.z + t_det * dz - v.detS.z;

                    // 投影到 U/V 轴
                    const float detU_len2 = v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z;
                    const float detV_len2 = v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z;
                    const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) / detU_len2;
                    const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) / detV_len2;

                    const int iu = (int)floorf(fu);
                    const int iv = (int)floorf(fv);
                    if (iu < 0 || iu >= Nu || iv < 0 || iv >= Nv) continue;

                    // ── Step 2：用探测器像素中心重建射线，与 FP 完全一致 ────────
                    const float3 detC = {
                        v.detS.x + (iu + 0.5f) * v.detU.x + (iv + 0.5f) * v.detV.x,
                        v.detS.y + (iu + 0.5f) * v.detU.y + (iv + 0.5f) * v.detV.y,
                        v.detS.z + (iu + 0.5f) * v.detU.z + (iv + 0.5f) * v.detV.z
                    };
                    const float3 ray = {
                        detC.x - v.src.x,
                        detC.y - v.src.y,
                        detC.z - v.src.z
                    };
                    const float ray_len = sqrtf(
                        ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
                    if (ray_len < 1e-8f) continue;

                    // ── Step 3：射线与体素 AABB 求交，得弦长 ────────────────────
                    float tmin = 0.f, tmax = 1.f;

#define VSLAB(r, s, b0, b1)                               \
    if (fabsf(r) > 1e-8f) {                               \
        float ta = (b0 - s) / r;                          \
        float tb = (b1 - s) / r;                          \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; } \
        tmin = fmaxf(tmin, ta);                           \
        tmax = fminf(tmax, tb);                           \
    } else if (s < b0 || s > b1) { tmin = 1.f; tmax = 0.f; }

                    VSLAB(ray.x, v.src.x, vx0, vx1)
                        VSLAB(ray.y, v.src.y, vy0, vy1)
                        VSLAB(ray.z, v.src.z, vz0, vz1)
#undef VSLAB

                        if (tmin >= tmax) continue;

                    const float seg_len = (tmax - tmin) * ray_len;

                    // ── Step 4：读正弦图，按弦长加权累加 ────────────────────────
                    acc += d_sino[((size_t)ia * Nv + iv) * Nu + iu] * seg_len;
                }

                d_vol[(size_t)iz * Ny * Nx + (size_t)iy * Nx + ix] = acc;
            }

//            __global__ void siddon_bp_voxel_kernel(
//                const float* __restrict__ d_sino,   // [K, Nv, Nu]
//                float* __restrict__       d_vol,    // [Nz, Ny, Nx]
//                const SConeProjGeomVec* d_views,
//                float3 vol_origin,
//                float  vox_x, float vox_y, float vox_z,
//                int    Nx, int Ny, int Nz,
//                int    Nu, int Nv, int K)
//            {
//                const int ix = blockIdx.x * blockDim.x + threadIdx.x;
//                const int iy = blockIdx.y * blockDim.y + threadIdx.y;
//                const int iz = blockIdx.z * blockDim.z + threadIdx.z;
//                if (ix >= Nx || iy >= Ny || iz >= Nz) return;
//
//                // 体素中心世界坐标
//                const float cx = vol_origin.x + (ix + 0.5f) * vox_x;
//                const float cy = vol_origin.y + (iy + 0.5f) * vox_y;
//                const float cz = vol_origin.z + (iz + 0.5f) * vox_z;
//
//                // 体素 AABB
//                const float vx0 = vol_origin.x + ix * vox_x;
//                const float vx1 = vol_origin.x + (ix + 1) * vox_x;
//                const float vy0 = vol_origin.y + iy * vox_y;
//                const float vy1 = vol_origin.y + (iy + 1) * vox_y;
//                const float vz0 = vol_origin.z + iz * vox_z;
//                const float vz1 = vol_origin.z + (iz + 1) * vox_z;
//
//                float acc = 0.f;
//
//                for (int ia = 0; ia < K; ++ia)
//                {
//                    const SConeProjGeomVec& v = d_views[ia];
//
//                    // 射线方向：src → 体素中心
//                    const float3 ray = {
//                        cx - v.src.x,
//                        cy - v.src.y,
//                        cz - v.src.z
//                    };
//                    const float ray_len = sqrtf(
//                        ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
//                    if (ray_len < 1e-8f) continue;
//
//                    // 射线与体素 AABB 求交，得到弦长
//                    float tmin = 0.f, tmax = 1.f;
//
//#define VSLAB(r, s, b0, b1) \
//    if (fabsf(r) > 1e-8f) { \
//        float ta = (b0 - s) / r; \
//        float tb = (b1 - s) / r; \
//        if (ta > tb) { float _t = ta; ta = tb; tb = _t; } \
//        tmin = fmaxf(tmin, ta); \
//        tmax = fminf(tmax, tb); \
//    } else if (s < b0 || s > b1) { tmin = tmax = 0.f; }
//
//                    VSLAB(ray.x, v.src.x, vx0, vx1)
//                        VSLAB(ray.y, v.src.y, vy0, vy1)
//                        VSLAB(ray.z, v.src.z, vz0, vz1)
//#undef VSLAB
//
//                        if (tmin >= tmax) continue;
//
//                    const float seg_len = (tmax - tmin) * ray_len;
//
//                    // 体素中心投影到探测器，计算 (iu, iv)
//                    // 用 src→体素中心 方向投影到探测器平面
//                    // 探测器平面法向量：src→detCenter 方向
//                    // 用参数化：找 t 使得 src + t*ray 落在探测器平面上
//                    // 探测器平面方程：(P - detS) · n = 0，n = detU × detV
//
//                    // 法向量
//                    const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
//                    const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
//                    const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;
//
//                    const float denom = ray.x * nx + ray.y * ny + ray.z * nz;
//                    if (fabsf(denom) < 1e-8f) continue;
//
//                    // src + t_det * ray 在探测器平面上
//                    const float dx = v.detS.x - v.src.x;
//                    const float dy = v.detS.y - v.src.y;
//                    const float dz = v.detS.z - v.src.z;
//                    const float t_det = (dx * nx + dy * ny + dz * nz) / denom;
//                    if (t_det <= 0.f) continue;
//
//                    // 交点
//                    const float px = v.src.x + t_det * ray.x - v.detS.x;
//                    const float py = v.src.y + t_det * ray.y - v.detS.y;
//                    const float pz = v.src.z + t_det * ray.z - v.detS.z;
//
//                    // 投影到 U/V 轴
//                    const float detU_len2 = v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z;
//                    const float detV_len2 = v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z;
//                    const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) / detU_len2;
//                    const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) / detV_len2;
//
//                    // 像素坐标（0-based，对应像素中心在 +0.5 处）
//                    const int iu = (int)floorf(fu);
//                    const int iv = (int)floorf(fv);
//                    if (iu < 0 || iu >= Nu || iv < 0 || iv >= Nv) continue;
//
//                    const float sino_val =
//                        d_sino[((size_t)ia * Nv + iv) * Nu + iu];
//
//                    acc += sino_val * seg_len;
//                }
//
//                d_vol[(size_t)iz * Ny * Nx + (size_t)iy * Nx + ix] = acc;
//            }

        } // namespace detail

        void bp_siddon_voxel_launch(
            const float* d_sino,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            cudaStream_t            stream)
        {
            if (K <= 0) return;

            // 三维 block，每个线程对应一个体素
            dim3 block(8, 8, 4);
            dim3 grid(
                (g.Nx + block.x - 1) / block.x,
                (g.Ny + block.y - 1) / block.y,
                (g.Nz + block.z - 1) / block.z);

            detail::siddon_bp_voxel_kernel << <grid, block, 0, stream >> > (
                d_sino, d_vol, d_views,
                g.origin(),
                g.vox_x, g.vox_y, g.vox_z,
                g.Nx, g.Ny, g.Nz,
                Nu, Nv, K);
        }

    } // namespace Bp
} // namespace YK