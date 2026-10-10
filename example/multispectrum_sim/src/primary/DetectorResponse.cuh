#pragma once

// Entrance-layer attenuation and scintillator efficiency by pixel and energy.

#include "ProjectionGeometry.cuh"

namespace yk::spectral::primary_detail {

__global__ void detectorEfficiencyKernel(DeviceGeometry g,
    const float* protective_thickness, const float* impurity_thickness,
    const float* scintillator_thickness, const float* protective_mu,
    const float* impurity_mu, const float* scintillator_mu,
    float protective_density, float impurity_density,
    float scintillator_density, int impurity_enabled,
    int oblique_path_correction, float* efficiency) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = g.nu * g.nv * g.ne;
    if (index >= total) return;
    const int e = index % g.ne;
    const int pixel = index / g.ne;
    const int u0 = pixel % g.nu;
    const int v0 = pixel / g.nu;
    const float pi = 3.14159265358979323846f;
    const float frame_angle = g.start - 0.5f * pi;
    const float ca = cosf(frame_angle), sa = sinf(frame_angle);
    float3 uaxis, vaxis, naxis;
    detectorFrame(g, frame_angle, uaxis, vaxis, naxis);
    const float3 source = make_float3(
        g.sid * ca + g.source_x * ca - g.source_y * sa,
        g.sid * sa + g.source_x * sa + g.source_y * ca,
        g.start_z + g.source_z);
    const float detector_distance = g.sdd - g.sid;
    // Source and panel share the axial scan position. A common translation
    // must not change the ray incidence angle or the detector efficiency.
    const float3 detector_principal = add3(make_float3(0.f, 0.f, g.start_z),
        scale3(make_float3(ca, sa, 0.f), -detector_distance));
    const float du = (u0 - 0.5f * (g.nu - 1)) * g.pu;
    const float dv = (v0 - 0.5f * (g.nv - 1)) * g.pv;
    const float3 center = add3(add3(add3(detector_principal,
        scale3(naxis, g.offset_n)), scale3(uaxis, du + g.offset_u)),
        scale3(vaxis, dv + g.offset_v));
    const float3 ray = make_float3(center.x - source.x, center.y - source.y,
        center.z - source.z);
    const float ray_len = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    const float cosine = fabsf(ray.x * naxis.x + ray.y * naxis.y + ray.z * naxis.z)
        / fmaxf(ray_len, 1e-6f);
    const float path_factor = oblique_path_correction
        ? 1.f / fmaxf(cosine, 1e-4f) : 1.f;
    const float mm_to_cm = path_factor * 0.1f;
    float entrance_exponent = mm_to_cm * __ldg(protective_thickness + pixel)
        * protective_density * __ldg(protective_mu + e);
    if (impurity_enabled)
        entrance_exponent += mm_to_cm * __ldg(impurity_thickness + pixel)
            * impurity_density * __ldg(impurity_mu + e);
    const float sensor_exponent = mm_to_cm * __ldg(scintillator_thickness + pixel)
        * scintillator_density * __ldg(scintillator_mu + e);
    efficiency[index] = expf(-entrance_exponent) * (-expm1f(-sensor_exponent));
}


} // namespace yk::spectral::primary_detail
