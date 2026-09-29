#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"

namespace YK::DltFpBp::detail {

template <typename Accumulator>
__device__ __forceinline__ bool traverseDltSiddonRay(
    const SDltRayGeometry& geometry, int detector_u, int detector_v,
    const SVolGeom& volume, Accumulator& accumulator)
{
    const float3 source = make_float3(geometry.source.x, geometry.source.y,
        geometry.source.z);
    const float3 ray = make_float3(
        geometry.ray00.x + detector_u * geometry.rayU.x
            + detector_v * geometry.rayV.x,
        geometry.ray00.y + detector_u * geometry.rayU.y
            + detector_v * geometry.rayV.y,
        geometry.ray00.z + detector_u * geometry.rayU.z
            + detector_v * geometry.rayV.z);
    const float ray_length = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    if (!(ray_length > 1e-12f) || !isfinite(ray_length)) return false;

    const float3 first_center = volume.origin();
    const float3 lower = make_float3(first_center.x - 0.5f * volume.vox_x,
        first_center.y - 0.5f * volume.vox_y,
        first_center.z - 0.5f * volume.vox_z);
    const float3 upper = make_float3(lower.x + volume.Nx * volume.vox_x,
        lower.y + volume.Ny * volume.vox_y,
        lower.z + volume.Nz * volume.vox_z);
    float near_parameter = 0.f;
    float far_parameter = CUDART_INF_F;
    const auto clip_axis = [&](float origin, float direction, float minimum,
                               float maximum) {
        if (fabsf(direction) <= 1e-12f)
            return origin >= minimum && origin <= maximum;
        float first = (minimum - origin) / direction;
        float second = (maximum - origin) / direction;
        if (first > second) { const float swap = first; first = second; second = swap; }
        near_parameter = fmaxf(near_parameter, first);
        far_parameter = fminf(far_parameter, second);
        return near_parameter < far_parameter;
    };
    if (!clip_axis(source.x, ray.x, lower.x, upper.x) ||
        !clip_axis(source.y, ray.y, lower.y, upper.y) ||
        !clip_axis(source.z, ray.z, lower.z, upper.z) ||
        !isfinite(far_parameter) || near_parameter >= far_parameter) return false;

    const float3 entry = make_float3(source.x + near_parameter * ray.x,
        source.y + near_parameter * ray.y,
        source.z + near_parameter * ray.z);
    int ix = max(0, min(volume.Nx - 1,
        static_cast<int>(floorf((entry.x - lower.x) / volume.vox_x))));
    int iy = max(0, min(volume.Ny - 1,
        static_cast<int>(floorf((entry.y - lower.y) / volume.vox_y))));
    int iz = max(0, min(volume.Nz - 1,
        static_cast<int>(floorf((entry.z - lower.z) / volume.vox_z))));
    const int step_x = ray.x >= 0.f ? 1 : -1;
    const int step_y = ray.y >= 0.f ? 1 : -1;
    const int step_z = ray.z >= 0.f ? 1 : -1;
    const float boundary_x = lower.x + (step_x > 0 ? ix + 1 : ix) * volume.vox_x;
    const float boundary_y = lower.y + (step_y > 0 ? iy + 1 : iy) * volume.vox_y;
    const float boundary_z = lower.z + (step_z > 0 ? iz + 1 : iz) * volume.vox_z;
    float next_x = fabsf(ray.x) > 1e-12f
        ? (boundary_x - source.x) / ray.x : CUDART_INF_F;
    float next_y = fabsf(ray.y) > 1e-12f
        ? (boundary_y - source.y) / ray.y : CUDART_INF_F;
    float next_z = fabsf(ray.z) > 1e-12f
        ? (boundary_z - source.z) / ray.z : CUDART_INF_F;
    const float delta_x = fabsf(ray.x) > 1e-12f
        ? fabsf(volume.vox_x / ray.x) : CUDART_INF_F;
    const float delta_y = fabsf(ray.y) > 1e-12f
        ? fabsf(volume.vox_y / ray.y) : CUDART_INF_F;
    const float delta_z = fabsf(ray.z) > 1e-12f
        ? fabsf(volume.vox_z / ray.z) : CUDART_INF_F;

    float current = near_parameter;
    const int maximum_steps = volume.Nx + volume.Ny + volume.Nz + 3;
    for (int step = 0; step < maximum_steps && current < far_parameter; ++step) {
        if (ix < 0 || ix >= volume.Nx || iy < 0 || iy >= volume.Ny ||
            iz < 0 || iz >= volume.Nz) break;
        const float next = fminf(fminf(next_x, next_y),
            fminf(next_z, far_parameter));
        if (next > current) {
            const size_t index = (static_cast<size_t>(iz) * volume.Ny + iy)
                * volume.Nx + ix;
            accumulator.add(index, (next - current) * ray_length);
        }
        const float tolerance = 2e-6f * fmaxf(1.f, fabsf(next));
        if (next_x <= next + tolerance) { ix += step_x; next_x += delta_x; }
        if (next_y <= next + tolerance) { iy += step_y; next_y += delta_y; }
        if (next_z <= next + tolerance) { iz += step_z; next_z += delta_z; }
        current = next;
    }
    return true;
}

} // namespace YK::DltFpBp::detail
