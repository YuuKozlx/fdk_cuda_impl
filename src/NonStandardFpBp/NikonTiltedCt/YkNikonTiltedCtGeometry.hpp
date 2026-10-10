#pragma once

// Nikon tilted CT will provide its per-view source/detector poses here.  The
// common conversion deliberately exposes only cone-vector geometry downstream.
#include "NonStandardFpBp/Geometry/YkNonStandardGeometry.hpp"
#include "YKCBCT/geometry/YkRigidTransform.hpp"
#include "YKCBCT/geometry/YkDetectorGeometryBuilders.hpp"

namespace YK::NonStandardFpBp::NikonTiltedCt {

using ViewPose = NonStandardViewPose;
using ::YK::NonStandardFpBp::buildConeVecGeometry;

struct GeometryConfig {
    int detector_u = 0;
    int detector_v = 0;
    float detector_pixel_u_mm = 0.f;
    float detector_pixel_v_mm = 0.f;
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float tilt_rad = 0.f;
    // Azimuth of the world-XY axis about which the initial pose is tilted.
    float tilt_axis_azimuth_rad = 0.f;
    float3 source = make_float3(0.f, 0.f, 0.f);
    // Point on the fixed world-Z rotation axis.
    float3 rotation_axis_pivot = make_float3(0.f, 0.f, 0.f);
    // Applied about the panel center before initial tilt and world-Z scanning.
    SDetectorPoseSpec detector_pose{};
};

// Build tilted CT geometry with a fixed world-Z scan axis.  tilt_rad defines
// only the initial (zero-degree) source/detector pose; every view then applies
// the inverse rotation around the world Z axis through rotation_axis_pivot.
// Object rotation is represented by the inverse rigid motion of the detector.
inline bool buildTiltedAxisGeometry(
    const GeometryConfig& config,
    const std::vector<float>& angles_rad,
    std::vector<NonStandardViewPose>& poses,
    std::vector<SConeProjGeomVec>& geometry)
{
    if (config.detector_u <= 0 || config.detector_v <= 0 ||
        !(config.detector_pixel_u_mm > 0.f) ||
        !(config.detector_pixel_v_mm > 0.f) || !(config.sid_mm > 0.f) ||
        !(config.sdd_mm > config.sid_mm) || !std::isfinite(config.tilt_rad) ||
        !std::isfinite(config.tilt_axis_azimuth_rad) || angles_rad.empty()) {
        poses.clear(); geometry.clear(); return false;
    }
    const float ca = std::cos(config.tilt_axis_azimuth_rad);
    const float sa = std::sin(config.tilt_axis_azimuth_rad);
    const float3 tilt_axis = make_float3(ca, sa, 0.f);
    const auto initial_tilt = SRigidTransform::aroundAxisPoint(tilt_axis,
        config.tilt_rad, config.rotation_axis_pivot);
    const float3 nominal_center = make_float3(
        config.source.x, config.source.y + config.sdd_mm,
        config.source.z);
    SScannerViewFrame frame{};
    frame.source = config.source; frame.detector_principal = nominal_center;
    frame.tangent_u = make_float3(1.f, 0.f, 0.f);
    frame.axis_v = make_float3(0.f, 0.f, 1.f);
    frame.radial_n = make_float3(0.f, -1.f, 0.f);
    SFlatDetectorSpec detector{};
    detector.channels = config.detector_u; detector.rows = config.detector_v;
    detector.channel_size_mm = config.detector_pixel_u_mm;
    detector.row_size_mm = config.detector_pixel_v_mm;
    detector.pose = config.detector_pose;
    std::vector<SConeProjGeomVec> initial_geometry;
    if (!buildFlatProjectionGeometry({frame}, detector, initial_geometry)) {
        poses.clear(); geometry.clear(); return false;
    }
    const auto& initial = initial_geometry.front();
    const float3 detector_u0 = make_float3(initial.detU.x, initial.detU.y, initial.detU.z);
    const float3 detector_v0 = make_float3(initial.detV.x, initial.detV.y, initial.detV.z);
    const float3 detector_center0 = GeometryBuilderDetail::add3(
        make_float3(initial.detS.x, initial.detS.y, initial.detS.z),
        GeometryBuilderDetail::add3(
            GeometryBuilderDetail::scale3(detector_u0, .5f * (config.detector_u - 1)),
            GeometryBuilderDetail::scale3(detector_v0, .5f * (config.detector_v - 1))));
    poses.resize(angles_rad.size());
    for (size_t i = 0; i < angles_rad.size(); ++i) {
        const float angle = angles_rad[i];
        if (!std::isfinite(angle)) { poses.clear(); geometry.clear(); return false; }
        const auto initial_source = initial_tilt.transformPoint(config.source);
        const auto initial_detector = initial_tilt.transformPoint(detector_center0);
        const auto initial_u = initial_tilt.transformVector(detector_u0);
        const auto initial_v = initial_tilt.transformVector(detector_v0);
        const auto scan = SRigidTransform::aroundAxisPoint(
            make_float3(0.f, 0.f, 1.f), -angle,
            config.rotation_axis_pivot);
        poses[i] = {scan.transformPoint(initial_source),
            scan.transformPoint(initial_detector),
            scan.transformVector(initial_u), scan.transformVector(initial_v), angle};
    }
    return buildConeVecGeometry(poses, config.detector_u, config.detector_v,
        geometry);
}

} // namespace YK::NonStandardFpBp::NikonTiltedCt
