#pragma once

#include <cmath>
#include <functional>
#include <vector>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK {

// Constructs a fixed-detector-plane cone-beam trajectory.  The detector is in
// y = IDD, its U/V axes follow world X/Z, and the source starts at
// (source_lateral_mm, -SID, 0) before rotating around world Y.  It is a vector
// geometry builder: callers may freely inspect or edit the resulting views.
inline void build_planar_ct_vec_geometry(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float sid_mm, float idd_mm, float source_lateral_mm)
{
    geometry.resize(view_count);
    const float3 det_center = make_float3(0.f, idd_mm, 0.f);
    const float3 det_u = make_float3(du_mm, 0.f, 0.f);
    const float3 det_v = make_float3(0.f, 0.f, dv_mm);
    const float cu = 0.5f * (nu - 1), cv = 0.5f * (nv - 1);
    const float3 det_origin = make_float3(det_center.x - det_u.x * cu - det_v.x * cv,
        det_center.y - det_u.y * cu - det_v.y * cv,
        det_center.z - det_u.z * cu - det_v.z * cv);

    for (int i = 0; i < view_count; ++i) {
        const float angle = angles[i];
        const float c = std::cos(angle), s = std::sin(angle);
        const float3 source = make_float3(
            source_lateral_mm * c, -sid_mm, -source_lateral_mm * s);
        geometry[i] = { make_float4(source.x, source.y, source.z, 0.f),
            make_float4(0.f, 1.f, 0.f, 0.f),
            make_float4(det_origin.x, det_origin.y, det_origin.z, 0.f),
            make_float4(det_u.x, det_u.y, det_u.z, 0.f),
            make_float4(det_v.x, det_v.y, det_v.z, 0.f),
            make_float4(angle, 0.f, 0.f, 0.f) };
    }
}

// Fixed plane detector with a caller-supplied source path.  The callback's
// angle argument is metadata only; the returned source vector is authoritative.
inline void build_planar_ct_vec_geometry_custom(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float idd_mm, const std::function<float3(float)>& source_position)
{
    geometry.resize(view_count);
    const float3 det_u = make_float3(du_mm, 0.f, 0.f);
    const float3 det_v = make_float3(0.f, 0.f, dv_mm);
    const float cu = 0.5f * (nu - 1), cv = 0.5f * (nv - 1);
    const float3 det_origin = make_float3(-det_u.x * cu - det_v.x * cv,
        idd_mm - det_u.y * cu - det_v.y * cv,
        -det_u.z * cu - det_v.z * cv);
    for (int i = 0; i < view_count; ++i) {
        const float angle = angles[i];
        const float3 source = source_position(angle);
        geometry[i] = { make_float4(source.x, source.y, source.z, 0.f),
            make_float4(0.f, 1.f, 0.f, 0.f),
            make_float4(det_origin.x, det_origin.y, det_origin.z, 0.f),
            make_float4(det_u.x, det_u.y, det_u.z, 0.f),
            make_float4(det_v.x, det_v.y, det_v.z, 0.f),
            make_float4(angle, 0.f, 0.f, 0.f) };
    }
}

inline void build_planar_ct_vec_geometry_ellipse(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float sid_mm, float idd_mm, float axis_x_mm, float axis_z_mm)
{
    build_planar_ct_vec_geometry_custom(geometry, angles, view_count, nu, nv,
        du_mm, dv_mm, idd_mm, [=](float angle) {
            return make_float3(axis_x_mm * std::cos(angle), -sid_mm,
                -axis_z_mm * std::sin(angle));
        });
}

} // namespace YK
