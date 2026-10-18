#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK::NonStandardFpBp {

// Complete flat-detector pose for one non-standard view.  detector_u and
// detector_v are physical one-pixel vectors, not unit axes.
struct NonStandardViewPose {
    float3 source = make_float3(0.f, 0.f, 0.f);
    float3 detector_center = make_float3(0.f, 0.f, 0.f);
    float3 detector_u = make_float3(0.f, 0.f, 0.f);
    float3 detector_v = make_float3(0.f, 0.f, 0.f);
    float angle_rad = 0.f;
};

inline bool buildConeVecGeometry(
    const std::vector<NonStandardViewPose>& poses,
    int detector_u, int detector_v,
    std::vector<SConeProjGeomVec>& geometry)
{
    if (poses.empty() || detector_u <= 0 || detector_v <= 0) {
        geometry.clear();
        return false;
    }
    const float half_u = 0.5f * static_cast<float>(detector_u - 1);
    const float half_v = 0.5f * static_cast<float>(detector_v - 1);
    geometry.resize(poses.size());
    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& pose = poses[i];
        const float3 start = make_float3(
            pose.detector_center.x - half_u * pose.detector_u.x -
                half_v * pose.detector_v.x,
            pose.detector_center.y - half_u * pose.detector_u.y -
                half_v * pose.detector_v.y,
            pose.detector_center.z - half_u * pose.detector_u.z -
                half_v * pose.detector_v.z);
        const auto finite3 = [](const float3& value) {
            return std::isfinite(value.x) && std::isfinite(value.y) &&
                std::isfinite(value.z);
        };
        if (!finite3(pose.source) || !finite3(pose.detector_center) ||
            !finite3(pose.detector_u) || !finite3(pose.detector_v) ||
            !std::isfinite(pose.angle_rad)) {
            geometry.clear();
            return false;
        }
        geometry[i] = {
            make_float4(pose.source.x, pose.source.y, pose.source.z, 0.f),
            make_float4(start.x, start.y, start.z, 0.f),
            make_float4(pose.detector_u.x, pose.detector_u.y,
                pose.detector_u.z, 0.f),
            make_float4(pose.detector_v.x, pose.detector_v.y,
                pose.detector_v.z, 0.f),
            make_float4(pose.angle_rad, 0.f, 0.f, 0.f)};
    }
    return true;
}

} // namespace YK::NonStandardFpBp
