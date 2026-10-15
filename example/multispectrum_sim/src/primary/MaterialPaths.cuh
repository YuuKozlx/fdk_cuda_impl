#pragma once
#include "ProjectionGeometry.cuh"

// Shared deterministic Siddon traversal. No spectra, RNG or signal-process state.
namespace yk::spectral::primary_detail {
__device__ inline bool boxHit(float3 s, float3 d, float3 lo, float3 hi, float& a, float& b) {
    a = 0.f; b = 1e30f;
    for (int k = 0; k < 3; ++k) {
        float o = k == 0 ? s.x : (k == 1 ? s.y : s.z);
        float q = k == 0 ? d.x : (k == 1 ? d.y : d.z);
        float l = k == 0 ? lo.x : (k == 1 ? lo.y : lo.z);
        float h = k == 0 ? hi.x : (k == 1 ? hi.y : hi.z);
        if (fabsf(q) < 1e-12f) { if (o < l || o > h) return false; continue; }
        float x = (l-o)/q, y = (h-o)/q; if (x > y) { float t=x; x=y; y=t; }
        a = fmaxf(a, x); b = fminf(b, y); if (a >= b) return false;
    }
    return b > 0.f;
}

// Traverse from the box entry in local coordinates. Keeping the traversal
// parameter relative to the entry avoids subtracting sub-millimeter voxel
// boundaries from the ~500 mm source coordinates.
template <int MaterialCount>
__device__ bool accumulatePaths(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, float3 source, float3 d, float3 lo, float3 hi,
    float* paths,unsigned* visited=nullptr) {
    float entry, exit;
    if (!boxHit(source, d, lo, hi, entry, exit)) return false;
    const float3 entry_point = make_float3(source.x + d.x * entry,
        source.y + d.y * entry, source.z + d.z * entry);
    const float3 q = make_float3(entry_point.x - lo.x,
        entry_point.y - lo.y, entry_point.z - lo.z);
    int ix = max(0, min(g.nx - 1, static_cast<int>(floorf(q.x / g.vx))));
    int iy = max(0, min(g.ny - 1, static_cast<int>(floorf(q.y / g.vy))));
    int iz = max(0, min(g.nz - 1, static_cast<int>(floorf(q.z / g.vz))));
    const int sx = d.x >= 0 ? 1 : -1, sy = d.y >= 0 ? 1 : -1, sz = d.z >= 0 ? 1 : -1;
    const float inf = 1e30f;
    float tx = d.x != 0 ? ((d.x > 0 ? (ix + 1) * g.vx : ix * g.vx) - q.x) / d.x : inf;
    float ty = d.y != 0 ? ((d.y > 0 ? (iy + 1) * g.vy : iy * g.vy) - q.y) / d.y : inf;
    float tz = d.z != 0 ? ((d.z > 0 ? (iz + 1) * g.vz : iz * g.vz) - q.z) / d.z : inf;
    const float dx = d.x != 0 ? g.vx / fabsf(d.x) : inf;
    const float dy = d.y != 0 ? g.vy / fabsf(d.y) : inf;
    const float dz = d.z != 0 ? g.vz / fabsf(d.z) : inf;
    float s = 0.f;
    bool hit_material = false;
    const float length = exit - entry;
    while (s < length && ix >= 0 && ix < g.nx && iy >= 0 && iy < g.ny && iz >= 0 && iz < g.nz) {
        if(visited)++*visited;
        const float next = fminf(tx, fminf(ty, tz));
        const float segment = fminf(next, length) - s;
        const int label = g.use_label_texture
            ? static_cast<int>(tex3D<unsigned char>(g.label_texture,
                ix + 0.5f, iy + 0.5f, iz + 0.5f))
            : static_cast<int>(__ldg(labels + (iz * g.ny + iy) * g.nx + ix));
        const int m = __ldg(label_to_material + label);
        if (m >= 0 && m < MaterialCount && segment > 0.f) {
            paths[m] += segment * 0.1f;
            hit_material = true;
        }
        s = next;
        if (tx <= next) { tx += dx; ix += sx; }
        if (ty <= next) { ty += dy; iy += sy; }
        if (tz <= next) { tz += dz; iz += sz; }
    }
    return hit_material;
}

}
