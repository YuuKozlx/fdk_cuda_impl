#pragma once
#include <cmath>
#include <functional>
#include <vector>
#include "YKCBCT/geometry/YkPlanarGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkDetectorGeometryBuilders.hpp"

// Pure host-side geometry builders shared by SDK clients and reconstruction.
namespace YK::NonStandardFpBp {
struct SourceOrbitGeometryConfig {
    int detector_u = 0;
    int detector_v = 0;
    float detector_pixel_u_mm = 0.f;
    float detector_pixel_v_mm = 0.f;
    float sid_mm = 0.f;
    float detector_y_mm = 0.f;
    float source_orbit_radius_mm = 0.f;
    float source_lateral_offset_mm = 0.f;
    // Static detector-local pose, using the standard flat U/N/V convention.
    SDetectorPoseSpec detector_pose{};
};

inline bool buildSourceTrajectoryGeometry(
    const SourceOrbitGeometryConfig& config,
    const std::vector<float>& trajectory_parameters,
    const std::function<float3(float)>& source_position,
    std::vector<SConeProjGeomVec>& geometry)
{
    if (config.detector_u <= 0 || config.detector_v <= 0 ||
        !(config.detector_pixel_u_mm > 0.f) ||
        !(config.detector_pixel_v_mm > 0.f) || !(config.sid_mm > 0.f) ||
        !std::isfinite(config.detector_y_mm) ||
        trajectory_parameters.empty() || !source_position) return false;

    SFlatDetectorSpec detector{};
    detector.channels = config.detector_u; detector.rows = config.detector_v;
    detector.channel_size_mm = config.detector_pixel_u_mm;
    detector.row_size_mm = config.detector_pixel_v_mm;
    detector.pose = config.detector_pose;
    std::vector<SScannerViewFrame> frames(trajectory_parameters.size());
    for (size_t i = 0; i < trajectory_parameters.size(); ++i) {
        const float angle = trajectory_parameters[i];
        const float3 source = source_position(angle);
        if (!std::isfinite(angle) || !std::isfinite(source.x) ||
            !std::isfinite(source.y) || !std::isfinite(source.z)) {
            geometry.clear(); return false;
        }
        auto& frame = frames[i];
        frame.source = source;
        frame.detector_principal = make_float3(0.f, config.detector_y_mm, 0.f);
        frame.tangent_u = make_float3(1.f, 0.f, 0.f);
        frame.axis_v = make_float3(0.f, 0.f, 1.f);
        frame.radial_n = make_float3(0.f, -1.f, 0.f);
        frame.angle_rad = angle;
    }
    return buildFlatProjectionGeometry(frames, detector, geometry);
}

// Build the per-view ray geometry used by the existing generic flat
// projector/backprojector.  No circular-FDK assumptions are introduced here.
inline bool buildSourceOrbitGeometry(
    const SourceOrbitGeometryConfig& config,
    const std::vector<float>& trajectory_angles_rad,
    std::vector<SConeProjGeomVec>& geometry)
{
    return buildSourceTrajectoryGeometry(config, trajectory_angles_rad,
        [&](float angle) {
            return make_float3(
                config.source_lateral_offset_mm +
                    config.source_orbit_radius_mm * std::cos(angle),
                -config.sid_mm,
                -config.source_orbit_radius_mm * std::sin(angle));
        }, geometry);
}

} // namespace YK::NonStandardFpBp
