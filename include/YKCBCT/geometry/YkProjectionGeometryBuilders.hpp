#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK {

namespace GeometryBuilderDetail {

inline float3 add(const float3& a, const float3& b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline float3 multiply(const float3& value, float scale)
{ return make_float3(value.x * scale, value.y * scale, value.z * scale); }

inline float dot(const float3& a, const float3& b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

inline float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float3 normalized(const float3& value)
{
    const float length = std::sqrt(dot(value, value));
    if (!(length > 0.f)) throw std::invalid_argument("zero geometry axis");
    return multiply(value, 1.f / length);
}

inline float3 rotateZ(const float3& value, float angle)
{
    const float c = std::cos(angle), s = std::sin(angle);
    return make_float3(c * value.x - s * value.y,
        s * value.x + c * value.y, value.z);
}

inline float3 rotateAroundAxis(const float3& value, const float3& axis,
    float angle)
{
    const float3 unit = normalized(axis);
    const float c = std::cos(angle), s = std::sin(angle);
    const float projection = dot(unit, value) * (1.f - c);
    const float3 perpendicular = cross(unit, value);
    return make_float3(value.x * c + perpendicular.x * s + unit.x * projection,
        value.y * c + perpendicular.y * s + unit.y * projection,
        value.z * c + perpendicular.z * s + unit.z * projection);
}

inline float4 asFloat4(const float3& value)
{ return make_float4(value.x, value.y, value.z, 0.f); }

inline void detectorAxes(float3 tilt_degrees, float3& u, float3& v)
{
    const auto radians = [](float degrees) {
        return degrees * 3.14159265358979323846f / 180.f;
    };
    u = make_float3(1.f, 0.f, 0.f);
    v = make_float3(0.f, 0.f, 1.f);
    const float3 axis_u = u;
    u = rotateAroundAxis(u, axis_u, radians(tilt_degrees.x));
    v = rotateAroundAxis(v, axis_u, radians(tilt_degrees.x));
    const float3 axis_v = v;
    u = rotateAroundAxis(u, axis_v, radians(tilt_degrees.z));
    v = rotateAroundAxis(v, axis_v, radians(tilt_degrees.z));
    const float3 normal = normalized(cross(u, v));
    u = rotateAroundAxis(u, normal, radians(tilt_degrees.y));
    v = rotateAroundAxis(v, normal, radians(tilt_degrees.y));
}

inline void validateCircularInputs(const std::vector<float>& angles,
    int views, int channels, int rows)
{
    if (views <= 0 || channels <= 0 || rows <= 0 ||
        angles.size() < static_cast<size_t>(views))
        throw std::invalid_argument("invalid circular cone geometry dimensions");
}

} // namespace GeometryBuilderDetail

// 构造绕世界 Z 轴旋转的锥束圆轨迹。offset 和 tilt 只在 builder 层解释，
// 输出的逐视图 vector geometry 是后续算法唯一的空间几何真源。
inline void buildCircularConeGeometry(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float sid_mm, float idd_mm,
    float3 detector_offset = make_float3(0.f, 0.f, 0.f),
    float3 detector_tilt_degrees = make_float3(0.f, 0.f, 0.f),
    float3 source_offset = make_float3(0.f, 0.f, 0.f))
{
    using namespace GeometryBuilderDetail;
    validateCircularInputs(angles, view_count, nu, nv);
    geometry.resize(view_count);
    float3 local_u{}, local_v{};
    detectorAxes(detector_tilt_degrees, local_u, local_v);
    const float3 local_normal = normalized(cross(local_v, local_u));
    const float3 source_base = make_float3(0.f, -sid_mm, 0.f);
    const float3 detector_base = make_float3(0.f, idd_mm, 0.f);
    const float center_u = 0.5f * (nu - 1);
    const float center_v = 0.5f * (nv - 1);

    for (int view = 0; view < view_count; ++view) {
        const float angle = angles[view];
        const float3 source = rotateZ(add(source_base, source_offset), angle);
        float3 detector_center = detector_base;
        detector_center = add(detector_center,
            multiply(local_u, detector_offset.x));
        detector_center = add(detector_center,
            multiply(local_normal, detector_offset.y));
        detector_center = add(detector_center,
            multiply(local_v, detector_offset.z));
        detector_center = rotateZ(detector_center, angle);
        const float3 detector_u = multiply(rotateZ(local_u, angle), du_mm);
        const float3 detector_v = multiply(rotateZ(local_v, angle), dv_mm);
        const float3 detector_start = add(add(detector_center,
            multiply(detector_u, -center_u)), multiply(detector_v, -center_v));
        geometry[view] = { asFloat4(source), asFloat4(detector_start),
            asFloat4(detector_u), asFloat4(detector_v),
            make_float4(angle, 0.f, 0.f, 0.f) };
    }
}

inline void buildCircularConeGeometry(
    std::vector<SConeProjGeomVec>& geometry, int view_count, int nu, int nv,
    float du_mm, float dv_mm, float sid_mm, float idd_mm,
    float3 detector_offset = make_float3(0.f, 0.f, 0.f),
    float3 detector_tilt_degrees = make_float3(0.f, 0.f, 0.f),
    float3 source_offset = make_float3(0.f, 0.f, 0.f))
{
    if (view_count <= 0) throw std::invalid_argument("view_count must be positive");
    constexpr float kTwoPi = 6.28318530717958647692f;
    std::vector<float> angles(static_cast<size_t>(view_count));
    for (int view = 0; view < view_count; ++view)
        angles[view] = kTwoPi * static_cast<float>(view) / view_count;
    buildCircularConeGeometry(geometry, angles, view_count, nu, nv, du_mm,
        dv_mm, sid_mm, idd_mm, detector_offset, detector_tilt_degrees,
        source_offset);
}

inline void buildCircularConeGeometryPerView(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float sid_mm, float idd_mm, const std::vector<float3>& detector_offsets,
    const std::vector<float3>& source_offsets,
    const std::vector<float3>& detector_tilts_degrees)
{
    using namespace GeometryBuilderDetail;
    validateCircularInputs(angles, view_count, nu, nv);
    const auto valid_size = [view_count](const std::vector<float3>& values) {
        return values.size() == 1 || values.size() >= static_cast<size_t>(view_count);
    };
    if (!valid_size(detector_offsets) || !valid_size(source_offsets) ||
        !valid_size(detector_tilts_degrees))
        throw std::invalid_argument("per-view geometry arrays must have size 1 or view_count");
    const auto value_at = [](const std::vector<float3>& values, int view) {
        return values.size() == 1 ? values.front() : values[view];
    };

    geometry.resize(view_count);
    const float center_u = 0.5f * (nu - 1);
    const float center_v = 0.5f * (nv - 1);
    for (int view = 0; view < view_count; ++view) {
        float3 local_u{}, local_v{};
        detectorAxes(value_at(detector_tilts_degrees, view), local_u, local_v);
        const float3 local_normal = normalized(cross(local_v, local_u));
        const float3 detector_offset = value_at(detector_offsets, view);
        const float angle = angles[view];
        const float3 source = rotateZ(add(make_float3(0.f, -sid_mm, 0.f),
            value_at(source_offsets, view)), angle);
        float3 detector_center = make_float3(0.f, idd_mm, 0.f);
        detector_center = add(detector_center, multiply(local_u, detector_offset.x));
        detector_center = add(detector_center, multiply(local_normal, detector_offset.y));
        detector_center = add(detector_center, multiply(local_v, detector_offset.z));
        detector_center = rotateZ(detector_center, angle);
        const float3 detector_u = multiply(rotateZ(local_u, angle), du_mm);
        const float3 detector_v = multiply(rotateZ(local_v, angle), dv_mm);
        const float3 detector_start = add(add(detector_center,
            multiply(detector_u, -center_u)), multiply(detector_v, -center_v));
        geometry[view] = { asFloat4(source), asFloat4(detector_start),
            asFloat4(detector_u), asFloat4(detector_v),
            make_float4(angle, 0.f, 0.f, 0.f) };
    }
}

// 旧名称仅作为源码兼容入口，实际实现统一收敛到公共 builder。
inline void build_circular_vec_geometry_from_theta(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float sid_mm, float idd_mm,
    float3 detector_offset = make_float3(0.f, 0.f, 0.f),
    float3 detector_tilt_degrees = make_float3(0.f, 0.f, 0.f),
    float3 source_offset = make_float3(0.f, 0.f, 0.f))
{
    buildCircularConeGeometry(geometry, angles, view_count, nu, nv, du_mm,
        dv_mm, sid_mm, idd_mm, detector_offset, detector_tilt_degrees,
        source_offset);
}

inline void build_circular_vec_geometry(
    std::vector<SConeProjGeomVec>& geometry, int view_count, int nu, int nv,
    float du_mm, float dv_mm, float sid_mm, float idd_mm,
    float3 detector_offset = make_float3(0.f, 0.f, 0.f),
    float3 detector_tilt_degrees = make_float3(0.f, 0.f, 0.f),
    float3 source_offset = make_float3(0.f, 0.f, 0.f))
{
    buildCircularConeGeometry(geometry, view_count, nu, nv, du_mm, dv_mm,
        sid_mm, idd_mm, detector_offset, detector_tilt_degrees, source_offset);
}

inline void build_circular_vec_geometry_perframe(
    std::vector<SConeProjGeomVec>& geometry, const std::vector<float>& angles,
    int view_count, int nu, int nv, float du_mm, float dv_mm,
    float sid_mm, float idd_mm, const std::vector<float3>& detector_offsets,
    const std::vector<float3>& source_offsets,
    const std::vector<float3>& detector_tilts_degrees)
{
    buildCircularConeGeometryPerView(geometry, angles, view_count, nu, nv,
        du_mm, dv_mm, sid_mm, idd_mm, detector_offsets, source_offsets,
        detector_tilts_degrees);
}

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
