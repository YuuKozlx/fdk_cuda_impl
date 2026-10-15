#pragma once
#include "ProjectionGeometry.cuh"
namespace yk::spectral::primary_detail {
// Scalar attenuation (mm^-1). Dominant voxel-axis planes with bilinear
// interpolation in the other two axes. Zero extension; physical mm integration.
__host__ __device__ inline float josephScalar(const float* volume, int nx, int ny, int nz, float3 spacing, float3 source, float3 direction)
{
    const int dims[3] = { nx, ny, nz };
    const float h[3] = { spacing.x, spacing.y, spacing.z };
    const float o[3] = { source.x, source.y, source.z }, d[3] = { direction.x, direction.y, direction.z };
    float enter = 0, leave = 1e30f, vo[3], vd[3];
    int axis = 0;
    for (int k = 0; k < 3; ++k) {
        const float lo = -.5f * dims[k] * h[k], hi = -lo;
        vo[k] = (o[k] - lo) / h[k] - .5f;
        vd[k] = d[k] / h[k];
        if (fabsf(d[k]) < 1e-12f) {
            if (o[k] < lo || o[k] > hi)
                return 0;
        } else {
            float a = (lo - o[k]) / d[k], b = (hi - o[k]) / d[k];
            enter = fmaxf(enter, fminf(a, b));
            leave = fminf(leave, fmaxf(a, b));
        }
        if (fabsf(vd[k]) > fabsf(vd[axis]))
            axis = k;
    }
    if (leave <= enter || fabsf(vd[axis]) < 1e-12f)
        return 0;
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    float a = vo[axis] + vd[axis] * enter, b = vo[axis] + vd[axis] * leave;
    int first = int(floorf(fminf(a, b) + .5f)), last = int(floorf(fmaxf(a, b) + .5f));
    first = first < 0 ? 0 : first;
    last = last >= dims[axis] ? dims[axis] - 1 : last;
    const float half = .5f / fabsf(vd[axis]);
    float result = 0;
    for (int k = first; k <= last; ++k) {
        const float t = (k - vo[axis]) / vd[axis];
        const float length = fmaxf(0.f, fminf(leave, t + half) - fmaxf(enter, t - half));
        const float cu = vo[u] + vd[u] * t, cv = vo[v] + vd[v] * t;
        const int iu = int(floorf(cu)), iv = int(floorf(cv));
        const float fu = cu - iu, fv = cv - iv;
        float sample = 0;
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i) {
                int index[3];
                index[axis] = k;
                index[u] = iu + i;
                index[v] = iv + j;
                if (index[u] >= 0 && index[u] < dims[u] && index[v] >= 0 && index[v] < dims[v])
                    sample += volume[(size_t(index[2]) * ny + index[1]) * nx + index[0]] * (i ? fu : 1 - fu) * (j ? fv : 1 - fv);
            }
        result += sample * length;
    }
    return result;
}
}
