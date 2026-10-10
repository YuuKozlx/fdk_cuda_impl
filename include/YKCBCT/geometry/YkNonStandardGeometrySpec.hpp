#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace YK {

enum class EPlanarResampleMode : int32_t {
    InPlaneRotated,
    PerspectiveOrthogonal,
};

// Fixed detector/object with a source moving on an XZ orbit around the
// reconstruction center. The acquired detector size comes from
// SSystemConfig::flat_detector; reconstruction_* is the fixed filtered window.
struct SPlanarSourceOrbitSpec {
    int total_views = 0;
    float start_angle_rad = 0.f;
    float scan_range_rad = 0.f;
    int rotation_direction = 1;
    float sid_mm = 0.f;
    float detector_y_mm = 0.f;
    float source_orbit_radius_mm = 0.f;
    float source_lateral_offset_mm = 0.f;
    int reconstruction_channels = 0;
    int reconstruction_rows = 0;
    EPlanarResampleMode resample_mode =
        EPlanarResampleMode::PerspectiveOrthogonal;
};

// The object and scan axis remain in world coordinates. tilt_rad changes only
// the zero-degree source/detector pose; subsequent views rotate around world Z.
struct SNikonTiltedCtSpec {
    int total_views = 0;
    float start_angle_rad = 0.f;
    float scan_range_rad = 0.f;
    int rotation_direction = 1;
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float tilt_rad = 0.f;
    float tilt_axis_azimuth_rad = 0.f;
    float3 source_offset_mm = make_float3(0.f, 0.f, 0.f);
    float3 rotation_axis_pivot_mm = make_float3(0.f, 0.f, 0.f);
};

struct SNonStandardGeometrySpec {
    SPlanarSourceOrbitSpec planar_source_orbit{};
    SNikonTiltedCtSpec nikon_tilted{};
};

} // namespace YK
