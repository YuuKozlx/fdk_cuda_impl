#include <cmath>

#include "YKCBCT/geometry/YkDetectorCoordinateConvention.hpp"

namespace {

bool near(float a, float b, float tolerance = 1e-6f)
{
    return std::fabs(a - b) <= tolerance;
}

bool near3(const float3& a, const float3& b)
{
    return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z);
}

} // namespace

int main_detector_coordinate_convention()
{
    using namespace YK;

    SDetectorCoordinateFrame external{};
    external.center = make_float3(12.f, -7.f, 3.f);
    external.u = make_float3(1.f, 0.f, 0.f);
    external.v = make_float3(0.f, 0.f, 1.f);
    external.n = make_float3(0.f, -1.f, 0.f);

    const auto internal = externalRightHandedToInternal(external);
    if (!near3(internal.center, external.center) ||
        !near3(internal.u, external.u) || !near3(internal.v, external.v) ||
        !near3(internal.n, make_float3(0.f, 1.f, 0.f)))
        return 1;

    const auto external_roundtrip = internalToExternalRightHanded(internal);
    if (!near3(external_roundtrip.center, external.center) ||
        !near3(external_roundtrip.u, external.u) ||
        !near3(external_roundtrip.v, external.v) ||
        !near3(external_roundtrip.n, external.n))
        return 1;

    SDetectorPoseSpec external_pose{};
    external_pose.offset_unv_mm = make_float3(2.085f, 1.25f, 4.17f);
    external_pose.tilt_u_rad = 0.01f;
    external_pose.tilt_v_rad = -0.02f;
    external_pose.tilt_n_rad = 0.0523598776f;

    const auto internal_pose = externalRightHandedToInternal(external_pose);
    if (!near(internal_pose.offset_unv_mm.x, external_pose.offset_unv_mm.x) ||
        !near(internal_pose.offset_unv_mm.y, -external_pose.offset_unv_mm.y) ||
        !near(internal_pose.offset_unv_mm.z, external_pose.offset_unv_mm.z) ||
        !near(internal_pose.tilt_u_rad, external_pose.tilt_u_rad) ||
        !near(internal_pose.tilt_v_rad, external_pose.tilt_v_rad) ||
        !near(internal_pose.tilt_n_rad, -external_pose.tilt_n_rad))
        return 1;

    const auto pose_roundtrip = internalToExternalRightHanded(internal_pose);
    if (!near3(pose_roundtrip.offset_unv_mm, external_pose.offset_unv_mm) ||
        !near(pose_roundtrip.tilt_u_rad, external_pose.tilt_u_rad) ||
        !near(pose_roundtrip.tilt_v_rad, external_pose.tilt_v_rad) ||
        !near(pose_roundtrip.tilt_n_rad, external_pose.tilt_n_rad))
        return 1;

    return 0;
}
