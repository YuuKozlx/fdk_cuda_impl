#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkGeometryBuilderTypes.hpp"
#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK {
namespace GeometryBuilderDetail {

inline float dot3(const float3& a, const float3& b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

inline float3 cross3(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float3 add3(const float3& a, const float3& b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline float3 subtract3(const float3& a, const float3& b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

inline float3 scale3(const float3& value, float scale)
{ return make_float3(value.x * scale, value.y * scale, value.z * scale); }

inline float3 normalized3(const float3& value)
{
    const float length2 = dot3(value, value);
    if (!(length2 > 0.f)) return make_float3(0.f, 0.f, 0.f);
    return scale3(value, 1.f / std::sqrt(length2));
}

inline float3 rotateAxisRadians(const float3& value, const float3& axis,
    float angle)
{
    const float3 unit = normalized3(axis);
    const float c = std::cos(angle), s = std::sin(angle);
    return add3(add3(scale3(value, c), scale3(cross3(unit, value), s)),
        scale3(unit, dot3(unit, value) * (1.f - c)));
}

inline void detectorAxes(const SScannerViewFrame& frame,
    const SDetectorPoseSpec& pose, float3& u, float3& v, float3& n)
{
    u = frame.tangent_u;
    v = frame.axis_v;
    n = frame.radial_n;
    v = rotateAxisRadians(v, u, pose.tilt_u_rad);
    n = rotateAxisRadians(n, u, pose.tilt_u_rad);
    u = rotateAxisRadians(u, v, pose.tilt_v_rad);
    n = rotateAxisRadians(n, v, pose.tilt_v_rad);
    u = rotateAxisRadians(u, n, pose.tilt_n_rad);
    v = rotateAxisRadians(v, n, pose.tilt_n_rad);
    u = normalized3(u);
    v = normalized3(v);
    n = normalized3(cross3(u, v));
    const float3 detector_to_source = subtract3(frame.source,
        frame.detector_principal);
    if (dot3(n, detector_to_source) < 0.f) n = scale3(n, -1.f);
}

inline float4 point4(const float3& value)
{ return make_float4(value.x, value.y, value.z, 0.f); }

inline bool validDetectorPose(const SDetectorPoseSpec& pose)
{
    return std::isfinite(pose.offset_unv_mm.x) &&
        std::isfinite(pose.offset_unv_mm.y) &&
        std::isfinite(pose.offset_unv_mm.z) &&
        std::isfinite(pose.tilt_u_rad) && std::isfinite(pose.tilt_v_rad) &&
        std::isfinite(pose.tilt_n_rad);
}

} // namespace GeometryBuilderDetail

inline bool buildFlatProjectionGeometry(
    const std::vector<SScannerViewFrame>& frames,
    const SFlatDetectorSpec& detector,
    std::vector<SConeProjGeomVec>& geometry)
{
    using namespace GeometryBuilderDetail;
    if (frames.empty() || detector.channels <= 0 || detector.rows <= 0 ||
        !(detector.channel_size_mm > 0.f) || !(detector.row_size_mm > 0.f) ||
        !validDetectorPose(detector.pose)) return false;
    geometry.resize(frames.size());
    const float center_u = 0.5f * (detector.channels - 1);
    const float center_v = 0.5f * (detector.rows - 1);
    for (size_t i = 0; i < frames.size(); ++i) {
        float3 u{}, v{}, n{};
        detectorAxes(frames[i], detector.pose, u, v, n);
        const float3 center = add3(frames[i].detector_principal,
            add3(scale3(u, detector.pose.offset_unv_mm.x),
                add3(scale3(n, detector.pose.offset_unv_mm.y),
                    scale3(v, detector.pose.offset_unv_mm.z))));
        const float3 du = scale3(u, detector.channel_size_mm);
        const float3 dv = scale3(v, detector.row_size_mm);
        const float3 start = subtract3(subtract3(center, scale3(du, center_u)),
            scale3(dv, center_v));
        geometry[i] = {point4(frames[i].source), point4(start), point4(du),
            point4(dv), make_float4(frames[i].angle_rad, 0.f, 0.f, 0.f)};
    }
    return true;
}

inline bool buildCylProjectionGeometry(
    const std::vector<SScannerViewFrame>& frames,
    const SCylDetectorSpec& detector,
    std::vector<SCylConeProjGeomVec>& geometry)
{
    using namespace GeometryBuilderDetail;
    if (frames.empty() || detector.channels <= 0 || detector.rows <= 0 ||
        !(detector.channel_arc_mm > 0.f) || !(detector.row_size_mm > 0.f) ||
        !(detector.curvature_radius_mm > 0.f) ||
        !validDetectorPose(detector.pose)) return false;
    geometry.resize(frames.size());
    const float channel_angle = detector.channel_arc_mm /
        detector.curvature_radius_mm;
    const float center_angle = detector.pose.offset_unv_mm.x /
        detector.curvature_radius_mm;
    const float sine = std::sin(center_angle);
    const float cosine = std::cos(center_angle);
    for (size_t i = 0; i < frames.size(); ++i) {
        float3 u{}, v{}, n{};
        detectorAxes(frames[i], detector.pose, u, v, n);
        const float3 surface_principal = add3(frames[i].detector_principal,
            scale3(n, detector.pose.offset_unv_mm.y));
        const float3 cylinder_center = add3(surface_principal,
            scale3(n, detector.curvature_radius_mm));
        const float3 center_radial = add3(scale3(n, -cosine), scale3(u, sine));
        const float3 center_tangent = add3(scale3(n, sine), scale3(u, cosine));
        const float3 center = add3(add3(cylinder_center,
            scale3(center_radial, detector.curvature_radius_mm)),
            scale3(v, detector.pose.offset_unv_mm.z));
        geometry[i] = {point4(frames[i].source), point4(center),
            point4(scale3(center_tangent, detector.curvature_radius_mm *
                channel_angle)), point4(scale3(v, detector.row_size_mm)),
            make_float4(frames[i].angle_rad, detector.curvature_radius_mm,
                0.f, 0.f)};
    }
    return true;
}

} // namespace YK
