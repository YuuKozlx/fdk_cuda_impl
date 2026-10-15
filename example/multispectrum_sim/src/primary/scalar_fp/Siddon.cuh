// Independently maintained copy from fdk-test f63a8fe, YkFPSiddonKernel.cu (scalar kernel).
#pragma once
#define AABB_LO(o, v) ((o) - 0.5f * (v))
#define AABB_HI(o, N, v) ((o) + ((N) - 0.5f) * (v))
#define VOX_LO(o, i, v) ((o) + ((i) - 0.5f) * (v))
#define VOX_HI(o, i, v) ((o) + ((i) + 0.5f) * (v))
#define IDX_FROM_WORLD(w, o, v) ((int)floorf(((w) - (o)) / (v) + 0.5f))

__global__ void siddon_fp_kernel(
    const float* __restrict__ d_vol,
    float* d_sino,
    const DeviceView* __restrict__ d_views,
    float3 vol_origin,
    float vox_x, float vox_y, float vox_z,
    int Nx, int Ny, int Nz,
    int Nu, int Nv, int K,
    bool accumulate)
{
    const int iu = blockIdx.x * blockDim.x + threadIdx.x;
    const int iv = blockIdx.y * blockDim.y + threadIdx.y;
    const int ia = blockIdx.z;
    if (iu >= Nu || iv >= Nv || ia >= K)
        return;

    const DeviceView& v = d_views[ia];

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

    const float x0 = AABB_LO(vol_origin.x, vox_x);
    const float x1 = AABB_HI(vol_origin.x, Nx, vox_x);
    const float y0 = AABB_LO(vol_origin.y, vox_y);
    const float y1 = AABB_HI(vol_origin.y, Ny, vox_y);
    const float z0 = AABB_LO(vol_origin.z, vox_z);
    const float z1 = AABB_HI(vol_origin.z, Nz, vox_z);

    float tmin = 0.f, tmax = FLT_MAX;

#define SLAB(r, s, b0, b1)       \
    if (fabsf(r) > 1e-8f) {      \
        float ta = (b0 - s) / r; \
        float tb = (b1 - s) / r; \
        if (ta > tb) {           \
            float _t = ta;       \
            ta = tb;             \
            tb = _t;             \
        }                        \
        tmin = fmaxf(tmin, ta);  \
        tmax = fminf(tmax, tb);  \
    } else if (s < b0 || s > b1) \
        return;

    SLAB(ray.x, v.src.x, x0, x1)
    SLAB(ray.y, v.src.y, y0, y1)
    SLAB(ray.z, v.src.z, z0, z1)
#undef SLAB

    tmax = fminf(tmax, 1.f);
    if (tmin >= tmax)
        return;

    const float ex = v.src.x + tmin * ray.x;
    const float ey = v.src.y + tmin * ray.y;
    const float ez = v.src.z + tmin * ray.z;
    int ix = max(0, min(Nx - 1, IDX_FROM_WORLD(ex, vol_origin.x, vox_x)));
    int iy = max(0, min(Ny - 1, IDX_FROM_WORLD(ey, vol_origin.y, vox_y)));
    int iz = max(0, min(Nz - 1, IDX_FROM_WORLD(ez, vol_origin.z, vox_z)));

    const int stepX = (ray.x >= 0.f) ? 1 : -1;
    const int stepY = (ray.y >= 0.f) ? 1 : -1;
    const int stepZ = (ray.z >= 0.f) ? 1 : -1;

    float bx = VOX_HI(vol_origin.x, ix, vox_x);
    if (stepX < 0)
        bx = VOX_LO(vol_origin.x, ix, vox_x);
    float by = VOX_HI(vol_origin.y, iy, vox_y);
    if (stepY < 0)
        by = VOX_LO(vol_origin.y, iy, vox_y);
    float bz = VOX_HI(vol_origin.z, iz, vox_z);
    if (stepZ < 0)
        bz = VOX_LO(vol_origin.z, iz, vox_z);

    float tX = (fabsf(ray.x) > 1e-8f) ? (bx - v.src.x) / ray.x : 1e30f;
    float tY = (fabsf(ray.y) > 1e-8f) ? (by - v.src.y) / ray.y : 1e30f;
    float tZ = (fabsf(ray.z) > 1e-8f) ? (bz - v.src.z) / ray.z : 1e30f;

    const float dtX = (fabsf(ray.x) > 1e-8f) ? fabsf(vox_x / ray.x) : 1e30f;
    const float dtY = (fabsf(ray.y) > 1e-8f) ? fabsf(vox_y / ray.y) : 1e30f;
    const float dtZ = (fabsf(ray.z) > 1e-8f) ? fabsf(vox_z / ray.z) : 1e30f;

    const float ray_len = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

    float fVal = 0.f;
    float t_cur = tmin;
    while (t_cur < tmax) {
        if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
            break;

        const float t_next = fminf(fminf(tX, tY), fminf(tZ, tmax));

        fVal += d_vol[(size_t)iz * Ny * Nx + (size_t)iy * Nx + ix]
            * (t_next - t_cur) * ray_len;

        t_cur = t_next;

        if (tX <= tY && tX <= tZ) {
            ix += stepX;
            tX += dtX;
        } else if (tY <= tZ) {
            iy += stepY;
            tY += dtY;
        } else {
            iz += stepZ;
            tZ += dtZ;
        }
    }

    const size_t idx = ((size_t)ia * Nv + iv) * Nu + iu;
    if (accumulate)
        d_sino[idx] += fVal;
    else
        d_sino[idx] = fVal;
}
