#include "YkFPSiddonLaunch.cuh"
#include <algorithm>

#define AABB_LO(o, v)      ((o) - 0.5f * (v))
#define AABB_HI(o, N, v)   ((o) + ((N) - 0.5f) * (v))
#define VOX_LO(o, i, v)    ((o) + ((i) - 0.5f) * (v))
#define VOX_HI(o, i, v)    ((o) + ((i) + 0.5f) * (v))
#define IDX_FROM_WORLD(w, o, v) ((int)floorf(((w) - (o)) / (v) + 0.5f))

namespace YK {
    namespace Fp {
        namespace detail {
            // =============================================================================
// ray-driven Siddon FP（scalar 版）
// =============================================================================
            __global__ void siddon_fp_kernel(
                const float* __restrict__ d_vol,
                float* d_sino,
                const SConeProjGeomVec* __restrict__ d_views,
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

                const float3 detC = {
                    v.detS.x + iu * v.detU.x + iv * v.detV.x,
                    v.detS.y + iu * v.detU.y + iv * v.detV.y,
                    v.detS.z + iu * v.detU.z + iv * v.detV.z
                };
                const float3 ray = {
                    detC.x - v.src.x,
                    detC.y - v.src.y,
                    detC.z - v.src.z
                };

                // [F5] volume AABB
                const float x0 = AABB_LO(vol_origin.x, vox_x);
                const float x1 = AABB_HI(vol_origin.x, Nx, vox_x);
                const float y0 = AABB_LO(vol_origin.y, vox_y);
                const float y1 = AABB_HI(vol_origin.y, Ny, vox_y);
                const float z0 = AABB_LO(vol_origin.z, vox_z);
                const float z1 = AABB_HI(vol_origin.z, Nz, vox_z);

                // [F2]
                float tmin = 0.f, tmax = FLT_MAX;

#define SLAB(r, s, b0, b1)                                  \
    if (fabsf(r) > 1e-8f) {                                 \
        float ta = (b0 - s) / r;                            \
        float tb = (b1 - s) / r;                            \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; }   \
        tmin = fmaxf(tmin, ta);                             \
        tmax = fminf(tmax, tb);                             \
    } else if (s < b0 || s > b1) return;

                SLAB(ray.x, v.src.x, x0, x1)
                    SLAB(ray.y, v.src.y, y0, y1)
                    SLAB(ray.z, v.src.z, z0, z1)
#undef SLAB

                    tmax = fminf(tmax, 1.f);  // [F2]
                if (tmin >= tmax) return;

                // [F5] 入射点 → 最近体素索引
                const float ex = v.src.x + tmin * ray.x;
                const float ey = v.src.y + tmin * ray.y;
                const float ez = v.src.z + tmin * ray.z;
                int ix = max(0, min(Nx - 1, IDX_FROM_WORLD(ex, vol_origin.x, vox_x)));
                int iy = max(0, min(Ny - 1, IDX_FROM_WORLD(ey, vol_origin.y, vox_y)));
                int iz = max(0, min(Nz - 1, IDX_FROM_WORLD(ez, vol_origin.z, vox_z)));

                const int stepX = (ray.x >= 0.f) ? 1 : -1;
                const int stepY = (ray.y >= 0.f) ? 1 : -1;
                const int stepZ = (ray.z >= 0.f) ? 1 : -1;

                // [F5] 下一体素边界
                float bx = VOX_HI(vol_origin.x, ix, vox_x);  if (stepX < 0) bx = VOX_LO(vol_origin.x, ix, vox_x);
                float by = VOX_HI(vol_origin.y, iy, vox_y);  if (stepY < 0) by = VOX_LO(vol_origin.y, iy, vox_y);
                float bz = VOX_HI(vol_origin.z, iz, vox_z);  if (stepZ < 0) bz = VOX_LO(vol_origin.z, iz, vox_z);

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
                    // [F6] size_t 防溢出
                    fVal += d_vol[(size_t)iz * Ny * Nx + (size_t)iy * Nx + ix]
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

            // =============================================================================
            // ray-driven Siddon FP（texture 版）
            // =============================================================================
            __global__ void siddon_fp_tex_kernel(
                cudaTextureObject_t                  tex,
                float* d_sino,
                const SConeProjGeomVec* __restrict__ d_views,
                float3 vol_origin,
                float  vox_x, float  vox_y, float  vox_z,
                float  rcp_vox_x, float rcp_vox_y, float rcp_vox_z,
                int    Nx, int Ny, int Nz,
                int    Nu, int Nv, int K,
                bool   accumulate)
            {
                const int iu = blockIdx.x * blockDim.x + threadIdx.x;
                const int iv = blockIdx.y * blockDim.y + threadIdx.y;
                const int ia = blockIdx.z;
                if (iu >= Nu || iv >= Nv || ia >= K) return;

                const SConeProjGeomVec& v = d_views[ia];

                const float3 detC = {
                    v.detS.x + iu * v.detU.x + iv * v.detV.x,
                    v.detS.y + iu * v.detU.y + iv * v.detV.y,
                    v.detS.z + iu * v.detU.z + iv * v.detV.z
                };
                const float3 ray = {
                    detC.x - v.src.x,
                    detC.y - v.src.y,
                    detC.z - v.src.z
                };

                const bool  ax_valid = fabsf(ray.x) > 1e-8f;
                const bool  ay_valid = fabsf(ray.y) > 1e-8f;
                const bool  az_valid = fabsf(ray.z) > 1e-8f;
                const float rcp_rx = ax_valid ? __frcp_rn(ray.x) : 0.f;
                const float rcp_ry = ay_valid ? __frcp_rn(ray.y) : 0.f;
                const float rcp_rz = az_valid ? __frcp_rn(ray.z) : 0.f;

                // [F5] volume AABB
                const float x0 = AABB_LO(vol_origin.x, vox_x);
                const float x1 = AABB_HI(vol_origin.x, Nx, vox_x);
                const float y0 = AABB_LO(vol_origin.y, vox_y);
                const float y1 = AABB_HI(vol_origin.y, Ny, vox_y);
                const float z0 = AABB_LO(vol_origin.z, vox_z);
                const float z1 = AABB_HI(vol_origin.z, Nz, vox_z);

                // [F2]
                float tmin = 0.f, tmax = FLT_MAX;

#define SLAB(valid, rcp_r, s, b0, b1)                           \
    if (valid) {                                                 \
        float ta = (b0 - s) * rcp_r;                            \
        float tb = (b1 - s) * rcp_r;                            \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; }       \
        tmin = fmaxf(tmin, ta);                                 \
        tmax = fminf(tmax, tb);                                 \
    } else if (s < b0 || s > b1) return;

                SLAB(ax_valid, rcp_rx, v.src.x, x0, x1)
                    SLAB(ay_valid, rcp_ry, v.src.y, y0, y1)
                    SLAB(az_valid, rcp_rz, v.src.z, z0, z1)
#undef SLAB

                    tmax = fminf(tmax, 1.f);  // [F2]
                if (tmin >= tmax) return;

                // [F5] 入射点 → 最近体素索引
                const float ex = v.src.x + tmin * ray.x;
                const float ey = v.src.y + tmin * ray.y;
                const float ez = v.src.z + tmin * ray.z;
                int ix = max(0, min(Nx - 1, (int)floorf((ex - vol_origin.x) * rcp_vox_x + 0.5f)));
                int iy = max(0, min(Ny - 1, (int)floorf((ey - vol_origin.y) * rcp_vox_y + 0.5f)));
                int iz = max(0, min(Nz - 1, (int)floorf((ez - vol_origin.z) * rcp_vox_z + 0.5f)));

                const int stepX = (ray.x >= 0.f) ? 1 : -1;
                const int stepY = (ray.y >= 0.f) ? 1 : -1;
                const int stepZ = (ray.z >= 0.f) ? 1 : -1;

                // [F5] 下一体素边界
                float bx = VOX_HI(vol_origin.x, ix, vox_x);  if (stepX < 0) bx = VOX_LO(vol_origin.x, ix, vox_x);
                float by = VOX_HI(vol_origin.y, iy, vox_y);  if (stepY < 0) by = VOX_LO(vol_origin.y, iy, vox_y);
                float bz = VOX_HI(vol_origin.z, iz, vox_z);  if (stepZ < 0) bz = VOX_LO(vol_origin.z, iz, vox_z);

                float tX = ax_valid ? (bx - v.src.x) * rcp_rx : 1e30f;
                float tY = ay_valid ? (by - v.src.y) * rcp_ry : 1e30f;
                float tZ = az_valid ? (bz - v.src.z) * rcp_rz : 1e30f;

                const float dtX = ax_valid ? fabsf(vox_x * rcp_rx) : 1e30f;
                const float dtY = ay_valid ? fabsf(vox_y * rcp_ry) : 1e30f;
                const float dtZ = az_valid ? fabsf(vox_z * rcp_rz) : 1e30f;

                const float ray_len = __fsqrt_rn(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

                // [F5] tex3D unnormalized 坐标：体素 ix 的中心对应坐标 ix+0.5
                //      但 vol_origin 是第0体素中心，tex 坐标 0.5 = 第0体素中心
                //      → tex 坐标 = ix + 0.5  （不变，tex 坐标系以左边界为0）
                float fVal = 0.f;
                float t_cur = tmin;
                while (t_cur < tmax)
                {
                    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
                        break;

                    const float t_next = fminf(fminf(tX, tY), fminf(tZ, tmax));
                    fVal += tex3D<float>(tex, ix + 0.5f, iy + 0.5f, iz + 0.5f)
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


//            __global__ void siddon_fp_tex_kernel(
//                cudaTextureObject_t     tex,
//                float* d_sino,
//                const SConeProjGeomVec* d_views,
//                float3 vol_origin,
//                float  vox_x, float  vox_y, float  vox_z,
//                float  rcp_vox_x, float  rcp_vox_y, float  rcp_vox_z,
//                int    Nx, int Ny, int Nz,
//                int    Nu, int Nv, int K,
//                bool   accumulate)
//            {
//                const int iu = blockIdx.x * blockDim.x + threadIdx.x;
//                const int iv = blockIdx.y * blockDim.y + threadIdx.y;
//                const int ia = blockIdx.z;
//                if (iu >= Nu || iv >= Nv || ia >= K) return;
//
//                const SConeProjGeomVec& v = d_views[ia];
//
//                const float3 detC = {
//                    v.detS.x + (iu + 0.5f) * v.detU.x + (iv + 0.5f) * v.detV.x,
//                    v.detS.y + (iu + 0.5f) * v.detU.y + (iv + 0.5f) * v.detV.y,
//                    v.detS.z + (iu + 0.5f) * v.detU.z + (iv + 0.5f) * v.detV.z
//                };
//
//                const float3 ray = {
//                    detC.x - v.src.x,
//                    detC.y - v.src.y,
//                    detC.z - v.src.z
//                };
//
//                // 射线方向倒数
//                const bool  ax_valid = fabsf(ray.x) > 1e-8f;
//                const bool  ay_valid = fabsf(ray.y) > 1e-8f;
//                const bool  az_valid = fabsf(ray.z) > 1e-8f;
//                const float rcp_rx = ax_valid ? __frcp_rn(ray.x) : 0.f;
//                const float rcp_ry = ay_valid ? __frcp_rn(ray.y) : 0.f;
//                const float rcp_rz = az_valid ? __frcp_rn(ray.z) : 0.f;
//
//                const float x0 = vol_origin.x, x1 = x0 + Nx * vox_x;
//                const float y0 = vol_origin.y, y1 = y0 + Ny * vox_y;
//                const float z0 = vol_origin.z, z1 = z0 + Nz * vox_z;
//
//                float tmin = 0.f, tmax = 1.f;
//
//#define SLAB(valid, rcp_r, s, b0, b1)                           \
//    if (valid) {                                                 \
//        float ta = (b0 - s) * rcp_r;                            \
//        float tb = (b1 - s) * rcp_r;                            \
//        if (ta > tb) { float _t = ta; ta = tb; tb = _t; }      \
//        tmin = fmaxf(tmin, ta);                                 \
//        tmax = fminf(tmax, tb);                                 \
//    } else if (s < b0 || s > b1) return;
//
//                SLAB(ax_valid, rcp_rx, v.src.x, x0, x1)
//                    SLAB(ay_valid, rcp_ry, v.src.y, y0, y1)
//                    SLAB(az_valid, rcp_rz, v.src.z, z0, z1)
//#undef SLAB
//
//                    if (tmin >= tmax) return;
//
//                const float ex = v.src.x + tmin * ray.x;
//                const float ey = v.src.y + tmin * ray.y;
//                const float ez = v.src.z + tmin * ray.z;
//
//                int ix = max(0, min(Nx - 1, (int)floorf((ex - vol_origin.x) * rcp_vox_x)));
//                int iy = max(0, min(Ny - 1, (int)floorf((ey - vol_origin.y) * rcp_vox_y)));
//                int iz = max(0, min(Nz - 1, (int)floorf((ez - vol_origin.z) * rcp_vox_z)));
//
//                const int stepX = (ray.x >= 0.f) ? 1 : -1;
//                const int stepY = (ray.y >= 0.f) ? 1 : -1;
//                const int stepZ = (ray.z >= 0.f) ? 1 : -1;
//
//                const float bx = vol_origin.x + (ix + (stepX > 0 ? 1 : 0)) * vox_x;
//                const float by = vol_origin.y + (iy + (stepY > 0 ? 1 : 0)) * vox_y;
//                const float bz = vol_origin.z + (iz + (stepZ > 0 ? 1 : 0)) * vox_z;
//
//                float tX = ax_valid ? (bx - v.src.x) * rcp_rx : 1e30f;
//                float tY = ay_valid ? (by - v.src.y) * rcp_ry : 1e30f;
//                float tZ = az_valid ? (bz - v.src.z) * rcp_rz : 1e30f;
//
//                const float dtX = ax_valid ? fabsf(vox_x * rcp_rx) : 1e30f;
//                const float dtY = ay_valid ? fabsf(vox_y * rcp_ry) : 1e30f;
//                const float dtZ = az_valid ? fabsf(vox_z * rcp_rz) : 1e30f;
//
//                const float ray_len = __fsqrt_rn(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
//
//                float fVal = 0.f;
//                float t_cur = tmin;
//
//                while (t_cur < tmax)
//                {
//                    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
//                        break;
//
//                    const float t_next = fminf(fminf(tX, tY), fminf(tZ, tmax));
//
//                    fVal += tex3D<float>(tex, ix + 0.5f, iy + 0.5f, iz + 0.5f)
//                        * (t_next - t_cur) * ray_len;
//
//                    t_cur = t_next;
//
//                    if (tX <= tY && tX <= tZ) { ix += stepX; tX += dtX; }
//                    else if (tY <= tZ) { iy += stepY; tY += dtY; }
//                    else { iz += stepZ; tZ += dtZ; }
//                }
//
//                const size_t idx = ((size_t)ia * Nv + iv) * Nu + iu;
//                if (accumulate) d_sino[idx] += fVal;
//                else            d_sino[idx] = fVal;
//            }

//__global__ void siddon_fp_tex_kernel(
//    cudaTextureObject_t     tex,
//    float* d_sino,
//    const SConeProjGeomVec* d_views,
//    float3 vol_origin,
//    float  vox_x, float  vox_y, float  vox_z,
//    float  rcp_vox_x, float  rcp_vox_y, float  rcp_vox_z,
//    int    Nx, int Ny, int Nz,
//    int    Nu, int Nv, int K,
//    bool   accumulate)
//{
//    const int iu = blockIdx.x * blockDim.x + threadIdx.x;
//    const int iv = blockIdx.y * blockDim.y + threadIdx.y;
//    const int ia = blockIdx.z;
//    if (iu >= Nu || iv >= Nv || ia >= K) return;
//
//    const SConeProjGeomVec& v = d_views[ia];
//
//    const float3 detC = {
//        v.detS.x + (iu + 0.5f) * v.detU.x + (iv + 0.5f) * v.detV.x,
//        v.detS.y + (iu + 0.5f) * v.detU.y + (iv + 0.5f) * v.detV.y,
//        v.detS.z + (iu + 0.5f) * v.detU.z + (iv + 0.5f) * v.detV.z
//    };
//
//    const float3 ray = {
//        detC.x - v.src.x,
//        detC.y - v.src.y,
//        detC.z - v.src.z
//    };
//
//    // 射线方向倒数
//    const bool  ax_valid = fabsf(ray.x) > 1e-8f;
//    const bool  ay_valid = fabsf(ray.y) > 1e-8f;
//    const bool  az_valid = fabsf(ray.z) > 1e-8f;
//    const float rcp_rx = ax_valid ? __frcp_rn(ray.x) : 0.f;
//    const float rcp_ry = ay_valid ? __frcp_rn(ray.y) : 0.f;
//    const float rcp_rz = az_valid ? __frcp_rn(ray.z) : 0.f;
//
//    const float x0 = vol_origin.x, x1 = x0 + Nx * vox_x;
//    const float y0 = vol_origin.y, y1 = y0 + Ny * vox_y;
//    const float z0 = vol_origin.z, z1 = z0 + Nz * vox_z;
//
//    float tmin = 0.f, tmax = 1.f;
//
//#define SLAB(valid, rcp_r, s, b0, b1)                           \
//    if (valid) {                                                 \
//        float ta = (b0 - s) * rcp_r;                            \
//        float tb = (b1 - s) * rcp_r;                            \
//        if (ta > tb) { float _t = ta; ta = tb; tb = _t; }      \
//        tmin = fmaxf(tmin, ta);                                 \
//        tmax = fminf(tmax, tb);                                 \
//    } else if (s < b0 || s > b1) return;
//
//    SLAB(ax_valid, rcp_rx, v.src.x, x0, x1)
//        SLAB(ay_valid, rcp_ry, v.src.y, y0, y1)
//        SLAB(az_valid, rcp_rz, v.src.z, z0, z1)
//#undef SLAB
//
//        if (tmin >= tmax) return;
//
//    const float ex = v.src.x + tmin * ray.x;
//    const float ey = v.src.y + tmin * ray.y;
//    const float ez = v.src.z + tmin * ray.z;
//
//    int ix = max(0, min(Nx - 1, (int)floorf((ex - vol_origin.x) * rcp_vox_x)));
//    int iy = max(0, min(Ny - 1, (int)floorf((ey - vol_origin.y) * rcp_vox_y)));
//    int iz = max(0, min(Nz - 1, (int)floorf((ez - vol_origin.z) * rcp_vox_z)));
//
//    const int stepX = (ray.x >= 0.f) ? 1 : -1;
//    const int stepY = (ray.y >= 0.f) ? 1 : -1;
//    const int stepZ = (ray.z >= 0.f) ? 1 : -1;
//
//    const float bx = vol_origin.x + (ix + (stepX > 0 ? 1 : 0)) * vox_x;
//    const float by = vol_origin.y + (iy + (stepY > 0 ? 1 : 0)) * vox_y;
//    const float bz = vol_origin.z + (iz + (stepZ > 0 ? 1 : 0)) * vox_z;
//
//    float tX = ax_valid ? (bx - v.src.x) * rcp_rx : 1e30f;
//    float tY = ay_valid ? (by - v.src.y) * rcp_ry : 1e30f;
//    float tZ = az_valid ? (bz - v.src.z) * rcp_rz : 1e30f;
//
//    const float dtX = ax_valid ? fabsf(vox_x * rcp_rx) : 1e30f;
//    const float dtY = ay_valid ? fabsf(vox_y * rcp_ry) : 1e30f;
//    const float dtZ = az_valid ? fabsf(vox_z * rcp_rz) : 1e30f;
//
//    const float ray_len = __fsqrt_rn(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
//
//    float fVal = 0.f;
//    float t_cur = tmin;
//
//    while (t_cur < tmax)
//    {
//        if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
//            break;
//
//        const float t_next = fminf(fminf(tX, tY), fminf(tZ, tmax));
//
//        fVal += tex3D<float>(tex, ix + 0.5f, iy + 0.5f, iz + 0.5f)
//            * (t_next - t_cur) * ray_len;
//
//        t_cur = t_next;
//
//        if (tX <= tY && tX <= tZ) { ix += stepX; tX += dtX; }
//        else if (tY <= tZ) { iy += stepY; tY += dtY; }
//        else { iz += stepZ; tZ += dtZ; }
//    }
//
//    const size_t idx = ((size_t)ia * Nv + iv) * Nu + iu;
//    if (accumulate) d_sino[idx] += fVal;
//    else            d_sino[idx] = fVal;
//}


        } // namespace detail

        void fp_siddon_launch(
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


        void fp_siddon_launch(
            cudaTextureObject_t     tex,
            float* d_sino,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            bool accumulate,
            cudaStream_t stream)
        {
            if (K <= 0) return;

            const float3 origin = g.origin();
            const float rcp_vox_x = 1.f / g.vox_x;
            const float rcp_vox_y = 1.f / g.vox_y;
            const float rcp_vox_z = 1.f / g.vox_z;

            dim3 block(16, 16, 1);
            dim3 grid((Nu + 15) / 16, (Nv + 15) / 16, K);

            detail::siddon_fp_tex_kernel << <grid, block, 0, stream >> > (
                tex, d_sino, d_views,
                origin,
                g.vox_x, g.vox_y, g.vox_z,
                rcp_vox_x, rcp_vox_y, rcp_vox_z,
                g.Nx, g.Ny, g.Nz,
                Nu, Nv, K,
                accumulate);
        }
    } // namespace Fp
} // namespace YK