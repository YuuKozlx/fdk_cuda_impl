#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "YKCBCT/geometry/YkSourceOrbitGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkDetectorGeometryBuilders.hpp"
#include "YKCBCT/geometry/YkTrajectorySampling.hpp"

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

} // namespace YK::NonStandardFpBp
