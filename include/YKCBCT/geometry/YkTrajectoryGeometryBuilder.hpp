#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkGeometryBuilderTypes.hpp"

namespace YK {
namespace GeometryBuilderDetail {

inline float3 rotateZRadians(const float3& value, float angle)
{
    const float c = std::cos(angle), s = std::sin(angle);
    return make_float3(c * value.x - s * value.y,
        s * value.x + c * value.y, value.z);
}

inline bool finite3(const float3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

} // namespace GeometryBuilderDetail

namespace GeometryBuilderDetail {

template <typename Trajectory, typename ZAtAngle>
inline bool buildScannerViewFramesImpl(const Trajectory& trajectory,
    ZAtAngle z_at_angle,
    std::vector<SScannerViewFrame>& frames)
{
    if (trajectory.angles_rad.empty() || !(trajectory.sid_mm > 0.f) ||
        !(trajectory.sdd_mm > trajectory.sid_mm) ||
        !finite3(trajectory.source_offset_mm)) return false;

    const float detector_distance = trajectory.sdd_mm - trajectory.sid_mm;
    frames.resize(trajectory.angles_rad.size());
    for (size_t i = 0; i < trajectory.angles_rad.size(); ++i) {
        const float angle = trajectory.angles_rad[i];
        if (!std::isfinite(angle)) {
            frames.clear();
            return false;
        }
        const float z = z_at_angle(angle);
        if (!std::isfinite(z)) {
            frames.clear();
            return false;
        }
        const float3 z_shift = make_float3(0.f, 0.f, z);
        const float3 source_local = make_float3(
            trajectory.source_offset_mm.x,
            -trajectory.sid_mm + trajectory.source_offset_mm.y,
            trajectory.source_offset_mm.z);
        const float3 detector_local = make_float3(0.f, detector_distance, 0.f);
        const auto shifted = [&](const float3& value) {
            const float3 rotated = rotateZRadians(value, angle);
            return make_float3(rotated.x + z_shift.x, rotated.y + z_shift.y,
                rotated.z + z_shift.z);
        };
        frames[i] = {shifted(source_local), shifted(detector_local),
            rotateZRadians(make_float3(1.f, 0.f, 0.f), angle),
            make_float3(0.f, 0.f, 1.f),
            rotateZRadians(make_float3(0.f, -1.f, 0.f), angle), angle};
    }
    return true;
}

} // namespace GeometryBuilderDetail

inline bool buildScannerViewFrames(const SCircularTrajectorySpec& trajectory,
    std::vector<SScannerViewFrame>& frames)
{
    if (!std::isfinite(trajectory.z_mm)) return false;
    return GeometryBuilderDetail::buildScannerViewFramesImpl(trajectory,
        [&](float) { return trajectory.z_mm; }, frames);
}

inline bool buildScannerViewFrames(const SHelicalTrajectorySpec& trajectory,
    std::vector<SScannerViewFrame>& frames)
{
    if (!std::isfinite(trajectory.start_z_mm) ||
        !std::isfinite(trajectory.pitch_mm_per_turn) ||
        trajectory.pitch_mm_per_turn == 0.f) return false;
    constexpr float kTwoPi = 6.28318530717958647692f;
    return GeometryBuilderDetail::buildScannerViewFramesImpl(trajectory,
        [&](float angle) {
            return trajectory.start_z_mm +
                trajectory.pitch_mm_per_turn * angle / kTwoPi;
        }, frames);
}

} // namespace YK
