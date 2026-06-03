// YkBPSiddonKernel.cu
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "BP/kernels/YkBPSiddonLaunch.cuh"

// ─── 公用宏：vol_origin = 第0体素中心 ────────────────────────────────────────
//
// AABB_LO(origin, vox)   = 第0体素左边界
// AABB_HI(origin, N, vox)= 第N-1体素右边界
// VOX_LO(origin, i, vox) = 第i体素左边界
// VOX_HI(origin, i, vox) = 第i体素右边界
// VOX_CTR(origin, i, vox)= 第i体素中心
// IDX_FROM_WORLD(w, origin, vox) = 最近体素索引（四舍五入）
//
#define AABB_LO(o, v)      ((o) - 0.5f * (v))
#define AABB_HI(o, N, v)   ((o) + ((N) - 0.5f) * (v))
#define VOX_LO(o, i, v)    ((o) + ((i) - 0.5f) * (v))
#define VOX_HI(o, i, v)    ((o) + ((i) + 0.5f) * (v))
#define VOX_CTR(o, i, v)   ((o) + (float)(i) * (v))
#define IDX_FROM_WORLD(w, o, v) ((int)floorf(((w) - (o)) / (v) + 0.5f))

namespace YK {
    namespace Bp {
        namespace detail {
            __global__ void siddon_bp_kernel(
                const float* __restrict__ d_sino,
                float* __restrict__ d_vol,
                const SConeProjGeomVec* __restrict__ d_views,
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
                    v.detS.x + iu * v.detU.x + iv * v.detV.x,
                    v.detS.y + iu * v.detU.y + iv * v.detV.y,
                    v.detS.z + iu * v.detU.z + iv * v.detV.z
                };
                const float3 ray = {
                    detC.x - v.src.x,
                    detC.y - v.src.y,
                    detC.z - v.src.z
                };
                const float ray_len = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

                // [F5] volume AABB：第0体素左边界 ~ 第N-1体素右边界
                const float x0 = AABB_LO(vol_origin.x, vox_x);
                const float x1 = AABB_HI(vol_origin.x, Nx, vox_x);
                const float y0 = AABB_LO(vol_origin.y, vox_y);
                const float y1 = AABB_HI(vol_origin.y, Ny, vox_y);
                const float z0 = AABB_LO(vol_origin.z, vox_z);
                const float z1 = AABB_HI(vol_origin.z, Nz, vox_z);

                // [F2] tmax = FLT_MAX，裁剪后 clamp 到探测器端 t=1
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

                    tmax = fminf(tmax, 1.f);
                if (tmin >= tmax) return;

                // [F5] 入射点 → 最近体素索引（四舍五入）
                const float ex = v.src.x + tmin * ray.x;
                const float ey = v.src.y + tmin * ray.y;
                const float ez = v.src.z + tmin * ray.z;
                int ix = max(0, min(Nx - 1, IDX_FROM_WORLD(ex, vol_origin.x, vox_x)));
                int iy = max(0, min(Ny - 1, IDX_FROM_WORLD(ey, vol_origin.y, vox_y)));
                int iz = max(0, min(Nz - 1, IDX_FROM_WORLD(ez, vol_origin.z, vox_z)));

                const int stepX = (ray.x >= 0.f) ? 1 : -1;
                const int stepY = (ray.y >= 0.f) ? 1 : -1;
                const int stepZ = (ray.z >= 0.f) ? 1 : -1;

                // [F5] 下一体素边界 = 当前体素的右（或左）边界
                float bx = VOX_HI(vol_origin.x, ix, vox_x);  if (stepX < 0) bx = VOX_LO(vol_origin.x, ix, vox_x);
                float by = VOX_HI(vol_origin.y, iy, vox_y);  if (stepY < 0) by = VOX_LO(vol_origin.y, iy, vox_y);
                float bz = VOX_HI(vol_origin.z, iz, vox_z);  if (stepZ < 0) bz = VOX_LO(vol_origin.z, iz, vox_z);

                float tX = (fabsf(ray.x) > 1e-8f) ? (bx - v.src.x) / ray.x : 1e30f;
                float tY = (fabsf(ray.y) > 1e-8f) ? (by - v.src.y) / ray.y : 1e30f;
                float tZ = (fabsf(ray.z) > 1e-8f) ? (bz - v.src.z) / ray.z : 1e30f;

                const float dtX = (fabsf(ray.x) > 1e-8f) ? fabsf(vox_x / ray.x) : 1e30f;
                const float dtY = (fabsf(ray.y) > 1e-8f) ? fabsf(vox_y / ray.y) : 1e30f;
                const float dtZ = (fabsf(ray.z) > 1e-8f) ? fabsf(vox_z / ray.z) : 1e30f;

                const float proj_val = d_sino[((size_t)ia * Nv + iv) * Nu + iu];

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
                //const float cx = vol_origin.x + ix * vox_x;
                //const float cy = vol_origin.y + iy * vox_y;
                //const float cz = vol_origin.z + iz * vox_z;
                const float cx = VOX_CTR(vol_origin.x, ix, vox_x);
                const float cy = VOX_CTR(vol_origin.y, iy, vox_y);
                const float cz = VOX_CTR(vol_origin.z, iz, vox_z);

                // 体素 AABB
                //const float vx0 = vol_origin.x + (ix - 0.5f) * vox_x;
                //const float vx1 = vol_origin.x + (ix + 0.5f) * vox_x;
                //const float vy0 = vol_origin.y + (iy - 0.5f) * vox_y;
                //const float vy1 = vol_origin.y + (iy + 0.5f) * vox_y;
                //const float vz0 = vol_origin.z + (iz - 0.5f) * vox_z;
                //const float vz1 = vol_origin.z + (iz + 0.5f) * vox_z;
                const float vx0 = VOX_LO(vol_origin.x, ix, vox_x);
                const float vx1 = VOX_HI(vol_origin.x, ix, vox_x);
                const float vy0 = VOX_LO(vol_origin.y, iy, vox_y);
                const float vy1 = VOX_HI(vol_origin.y, iy, vox_y);
                const float vz0 = VOX_LO(vol_origin.z, iz, vox_z);
                const float vz1 = VOX_HI(vol_origin.z, iz, vox_z);

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

                    // 改成（对的，fu 直接对应像素中心整数坐标）
                    const int iu = (int)floorf(fu + 0.5f);  // 四舍五入到最近像素
                    const int iv = (int)floorf(fv + 0.5f);
                    if (iu < 0 || iu >= Nu || iv < 0 || iv >= Nv) continue;

                    // ── Step 2：用探测器像素中心重建射线，与 FP 完全一致 ────────
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




            template<int ZSIZE>
            __global__ void siddon_bp_voxel_v2_kernel(
                cudaTextureObject_t                  sinoTex,
                const SConeProjGeomVec* __restrict__ d_views,
                float* __restrict__                  d_vol,
                float3                               vol_origin,
                float                                vox_x, float vox_y, float vox_z,
                int                                  Nx, int Ny, int Nz,
                int                                  Nu, int Nv, int K)
            {
                const int ix = blockIdx.x * blockDim.x + threadIdx.x;
                const int iy = blockIdx.y * blockDim.y + threadIdx.y;
                if (ix >= Nx || iy >= Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= Nz) return;

                const float cx = VOX_CTR(vol_origin.x, ix, vox_x);
                const float cy = VOX_CTR(vol_origin.y, iy, vox_y);

                const float vx0 = VOX_LO(vol_origin.x, ix, vox_x);
                const float vx1 = VOX_HI(vol_origin.x, ix, vox_x);
                const float vy0 = VOX_LO(vol_origin.y, iy, vox_y);
                const float vy1 = VOX_HI(vol_origin.y, iy, vox_y);

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int ia = 0; ia < K; ++ia)
                {
                    const SConeProjGeomVec& v = d_views[ia];
                    const float ia_tex = (float)ia + 0.5f;

                    const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
                    const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
                    const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;

                    const float SDD_plane =
                        (v.detS.x - v.src.x) * nx +
                        (v.detS.y - v.src.y) * ny +
                        (v.detS.z - v.src.z) * nz;

                    const float rcp_U2 = __frcp_rn(
                        v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z);
                    const float rcp_V2 = __frcp_rn(
                        v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z);

                    const float dx_xy = cx - v.src.x;
                    const float dy_xy = cy - v.src.y;

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz)
                    {
                        const int zIdx = startZ + iz;
                        if (zIdx >= Nz) break;

                        const float cz_ = VOX_CTR(vol_origin.z, zIdx, vox_z);
                        const float vz0 = VOX_LO(vol_origin.z, zIdx, vox_z);
                        const float vz1 = VOX_HI(vol_origin.z, zIdx, vox_z);

                        const float dz = cz_ - v.src.z;

                        // ── Step 1：体素中心投影到探测器
                        const float denom_n = dx_xy * nx + dy_xy * ny + dz * nz;
                        if (fabsf(denom_n) < 1e-8f) continue;

                        const float t_det = __fdividef(SDD_plane, denom_n);
                        if (t_det <= 0.f) continue;

                        const float px = v.src.x + t_det * dx_xy - v.detS.x;
                        const float py = v.src.y + t_det * dy_xy - v.detS.y;
                        const float pz = v.src.z + t_det * dz - v.detS.z;

                        const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) * rcp_U2;
                        const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) * rcp_V2;

                        if (fu < -0.5f || fu >= (float)Nu - 0.5f ||
                            fv < -0.5f || fv >= (float)Nv - 0.5f) continue;

                        // ── Step 2：用浮点坐标重建射线
                        const float rx = v.detS.x + fu * v.detU.x + fv * v.detV.x - v.src.x;
                        const float ry = v.detS.y + fu * v.detU.y + fv * v.detV.y - v.src.y;
                        const float rz = v.detS.z + fu * v.detU.z + fv * v.detV.z - v.src.z;
                        const float ray_len = __fsqrt_rn(rx * rx + ry * ry + rz * rz);
                        if (ray_len < 1e-8f) continue;

                        // ── Step 3：射线与体素 AABB 求交
                        float tmin = 0.f, tmax = 1.f;

#define VSLAB(r, s, b0, b1)                                     \
    if (fabsf(r) > 1e-8f) {                                     \
        float ta = (b0 - s) / r;                                \
        float tb = (b1 - s) / r;                                \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; }       \
        tmin = fmaxf(tmin, ta);                                 \
        tmax = fminf(tmax, tb);                                 \
    } else if (s < b0 || s > b1) { tmin = 1.f; tmax = 0.f; }

                        VSLAB(rx, v.src.x, vx0, vx1)
                            VSLAB(ry, v.src.y, vy0, vy1)
                            VSLAB(rz, v.src.z, vz0, vz1)
#undef VSLAB

                            if (tmin >= tmax) continue;

                        const float seg_len = (tmax - tmin) * ray_len;

                        // ── Step 4：双线性插值读正弦图
                        Z[iz] += tex3D<float>(sinoTex, fu + 0.5f, fv + 0.5f, ia_tex) * seg_len;
                    }
                }

#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= Nz) break;
                    const size_t idx = (size_t)zIdx * Ny * Nx
                        + (size_t)iy * Nx
                        + (size_t)ix;
                    d_vol[idx] += Z[iz];
                }
            }


            template<int ZSIZE>
            __global__ void siddon_bp_voxel_v3_kernel(
                cudaTextureObject_t                  sinoTex,
                const SConeProjGeomVec* __restrict__ d_views,
                float* __restrict__                  d_vol,
                float3                               vol_origin,
                float                                vox_x, float vox_y, float vox_z,
                int                                  Nx, int Ny, int Nz,
                int                                  Nu, int Nv, int K)
            {
                const int ix = blockIdx.x * blockDim.x + threadIdx.x;
                const int iy = blockIdx.y * blockDim.y + threadIdx.y;
                if (ix >= Nx || iy >= Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= Nz) return;

                const float cx = VOX_CTR(vol_origin.x, ix, vox_x);
                const float cy = VOX_CTR(vol_origin.y, iy, vox_y);

                const float vx0 = VOX_LO(vol_origin.x, ix, vox_x);
                const float vx1 = VOX_HI(vol_origin.x, ix, vox_x);
                const float vy0 = VOX_LO(vol_origin.y, iy, vox_y);
                const float vy1 = VOX_HI(vol_origin.y, iy, vox_y);

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int ia = 0; ia < K; ++ia)
                {
                    const SConeProjGeomVec& v = d_views[ia];
                    const float ia_tex = (float)ia + 0.5f;

                    const float nx = v.detU.y * v.detV.z - v.detU.z * v.detV.y;
                    const float ny = v.detU.z * v.detV.x - v.detU.x * v.detV.z;
                    const float nz = v.detU.x * v.detV.y - v.detU.y * v.detV.x;

                    const float SDD_plane =
                        (v.detS.x - v.src.x) * nx +
                        (v.detS.y - v.src.y) * ny +
                        (v.detS.z - v.src.z) * nz;

                    const float rcp_U2 = __frcp_rn(
                        v.detU.x * v.detU.x + v.detU.y * v.detU.y + v.detU.z * v.detU.z);
                    const float rcp_V2 = __frcp_rn(
                        v.detV.x * v.detV.x + v.detV.y * v.detV.y + v.detV.z * v.detV.z);

                    const float dx_xy = cx - v.src.x;
                    const float dy_xy = cy - v.src.y;

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz)
                    {
                        const int zIdx = startZ + iz;
                        if (zIdx >= Nz) break;

                        const float cz_ = VOX_CTR(vol_origin.z, zIdx, vox_z);
                        const float vz0 = VOX_LO(vol_origin.z, zIdx, vox_z);
                        const float vz1 = VOX_HI(vol_origin.z, zIdx, vox_z);

                        const float dz = cz_ - v.src.z;

                        // ── Step 1：体素中心投影到探测器 ─────────────────────
                        const float denom_n = dx_xy * nx + dy_xy * ny + dz * nz;
                        if (fabsf(denom_n) < 1e-8f) continue;

                        const float t_det = __fdividef(SDD_plane, denom_n);
                        if (t_det <= 0.f) continue;

                        const float px = v.src.x + t_det * dx_xy - v.detS.x;
                        const float py = v.src.y + t_det * dy_xy - v.detS.y;
                        const float pz = v.src.z + t_det * dz - v.detS.z;

                        const float fu = (px * v.detU.x + py * v.detU.y + pz * v.detU.z) * rcp_U2;
                        const float fv = (px * v.detV.x + py * v.detV.y + pz * v.detV.z) * rcp_V2;

                        // ── Step 2：取整到最近像素（和 v1 一致）──────────────
                        const int iu = (int)floorf(fu + 0.5f);
                        const int iv = (int)floorf(fv + 0.5f);
                        if (iu < 0 || iu >= Nu || iv < 0 || iv >= Nv) continue;

                        const float rx = v.detS.x + iu * v.detU.x + iv * v.detV.x - v.src.x;
                        const float ry = v.detS.y + iu * v.detU.y + iv * v.detV.y - v.src.y;
                        const float rz = v.detS.z + iu * v.detU.z + iv * v.detV.z - v.src.z;
                        const float ray_len = __fsqrt_rn(rx * rx + ry * ry + rz * rz);
                        if (ray_len < 1e-8f) continue;

                        // ── Step 3：射线与体素 AABB 求交 ─────────────────────
                        float tmin = 0.f, tmax = 1.f;

#define VSLAB(r, s, b0, b1)                                     \
    if (fabsf(r) > 1e-8f) {                                     \
        float ta = (b0 - s) / r;                                \
        float tb = (b1 - s) / r;                                \
        if (ta > tb) { float _t = ta; ta = tb; tb = _t; }       \
        tmin = fmaxf(tmin, ta);                                 \
        tmax = fminf(tmax, tb);                                 \
    } else if (s < b0 || s > b1) { tmin = 1.f; tmax = 0.f; }

                        VSLAB(rx, v.src.x, vx0, vx1)
                            VSLAB(ry, v.src.y, vy0, vy1)
                            VSLAB(rz, v.src.z, vz0, vz1)
#undef VSLAB

                            if (tmin >= tmax) continue;

                        const float seg_len = (tmax - tmin) * ray_len;

                        // ── Step 4：整数坐标采样，退化为最近邻 ───────────────
                        Z[iz] += tex3D<float>(sinoTex,
                            iu + 0.5f, iv + 0.5f, ia_tex) * seg_len;
                    }
                }

#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz)
                {
                    const int zIdx = startZ + iz;
                    if (zIdx >= Nz) break;
                    const size_t idx = (size_t)zIdx * Ny * Nx
                        + (size_t)iy * Nx
                        + (size_t)ix;
                    d_vol[idx] += Z[iz];
                }
            }


        } // namespace detail

        void bp_siddon_voxel_launch(
            const float* d_sino,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            bool accumulate,
            cudaStream_t            stream)
        {
            if (K <= 0) return;
            if (!accumulate) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
                    (size_t)g.Nx * g.Ny * g.Nz * sizeof(float), stream));
            }

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




        void bp_siddon_voxel_v2_launch(
            cudaTextureObject_t              sinoTex,
            float* d_vol,
            const SConeProjGeomVec* d_views,
            const SVolGeom& g,
            int Nu, int Nv, int K,
            bool accumulate,
            cudaStream_t stream)
        {
            if (!accumulate) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0,
                    (size_t)g.Nx * g.Ny * g.Nz * sizeof(float), stream));
            }

            constexpr int ZSIZE = 4;
            const dim3 block(16, 16, 1);
            const dim3 grid(
                (g.Nx + block.x - 1) / block.x,
                (g.Ny + block.y - 1) / block.y,
                (g.Nz + ZSIZE - 1) / ZSIZE);

            detail::siddon_bp_voxel_v2_kernel<ZSIZE> << <grid, block, 0, stream >> > (
                sinoTex, d_views, d_vol,
                g.origin(),
                g.vox_x, g.vox_y, g.vox_z,
                g.Nx, g.Ny, g.Nz,
                Nu, Nv, K);
        }

    } // namespace Bp
} // namespace YK