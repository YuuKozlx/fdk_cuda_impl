#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "YKCBCT/geometry/YkPlanarGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkDetectorGeometryBuilders.hpp"
#include "NonStandardFpBp/Geometry/YkTrajectorySampling.hpp"

namespace YK::NonStandardFpBp {

enum class SourceOrbitResampleMode {
    // Keep the physical detector plane and rotate only its local U/V grid.
    InPlaneRotated,
    // Map rays directly to a per-view plane normal to source -> volume center.
    PerspectiveOrthogonal
};

// Geometry description for a fixed flat detector and a source orbit around
// the Y axis.  The angle is only a source-trajectory parameter; it does not
// rotate the detector or the object.
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

// Uniform trajectory sampling shared by non-standard source paths.  The
// endpoint is excluded so a full turn does not duplicate its first view.
struct SourceTrajectorySampling {
    int views = 0;
    float start_angle_rad = 0.f;
    float range_rad = 0.f;
    float direction = 1.f;
};

// A fixed-size local projection window.  origin_* is allowed to vary with
// view, while width/height remain constant so filtering sees one grid.
struct ProjectionWindow {
    int width = 0;
    int height = 0;
    int origin_u = 0;
    int origin_v = 0;
};

struct SafeWindowResult {
    bool valid = false;
    float half_diagonal_pixels = 0.f;
    float min_margin_pixels = 0.f;
    ProjectionWindow window{};
};

// The rectangle's circumradius is used for the all-angle safety test.  This
// is deliberately independent of the source trajectory so callers can use
// it for a crop whose center is obtained from their chosen map.
inline SafeWindowResult makeSafeWindow(int detector_u, int detector_v,
    int window_u, int window_v, float center_u, float center_v)
{
    SafeWindowResult result{};
    if (detector_u <= 0 || detector_v <= 0 || window_u <= 0 || window_v <= 0 ||
        window_u > detector_u || window_v > detector_v ||
        !std::isfinite(center_u) || !std::isfinite(center_v)) return result;

    const float half_u = 0.5f * static_cast<float>(window_u - 1);
    const float half_v = 0.5f * static_cast<float>(window_v - 1);
    result.half_diagonal_pixels = std::hypot(half_u, half_v);
    result.min_margin_pixels = std::fmin(
        std::fmin(center_u, static_cast<float>(detector_u - 1) - center_u),
        std::fmin(center_v, static_cast<float>(detector_v - 1) - center_v));
    if (result.half_diagonal_pixels > result.min_margin_pixels + 1e-4f)
        return result;

    // Round the requested center to an integer crop origin while retaining a
    // fixed-size window.  The safety test above guarantees these bounds.
    int origin_u = static_cast<int>(std::lround(center_u - half_u));
    int origin_v = static_cast<int>(std::lround(center_v - half_v));
    origin_u = std::fmax(0, std::fmin(origin_u, detector_u - window_u));
    origin_v = std::fmax(0, std::fmin(origin_v, detector_v - window_v));
    result.window = {window_u, window_v, origin_u, origin_v};
    result.valid = true;
    return result;
}

inline bool buildUniformTrajectoryAngles(
    const SourceTrajectorySampling& sampling,
    std::vector<float>& angles_rad)
{
    TrajectorySampling generic{};
    generic.views = sampling.views;
    generic.start_parameter = sampling.start_angle_rad;
    generic.range = sampling.range_rad;
    generic.direction = sampling.direction;
    return buildUniformTrajectoryParameters(generic, angles_rad);
}

// Generic fixed-detector geometry builder.  The callback receives the
// trajectory parameter for one view and returns that view's source position.
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
