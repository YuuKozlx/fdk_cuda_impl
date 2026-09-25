#pragma once

#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkGeometryBuilderTypes.hpp"

namespace YK {

// Physical detector frame used at API and calibration boundaries. The
// internal YKCBCT convention keeps U/V unchanged and defines N from source
// toward detector, so U x V = -N. External right-handed tools commonly use
// U x V = N, with N pointing from detector toward source.
struct SDetectorCoordinateFrame {
    float3 center = make_float3(0.f, 0.f, 0.f);
    float3 u = make_float3(1.f, 0.f, 0.f);
    float3 v = make_float3(0.f, 0.f, 1.f);
    float3 n = make_float3(0.f, -1.f, 0.f);
};

inline float3 reverseDetectorNormal(float3 value)
{
    return make_float3(-value.x, -value.y, -value.z);
}

inline SDetectorCoordinateFrame externalRightHandedToInternal(
    const SDetectorCoordinateFrame& external)
{
    SDetectorCoordinateFrame internal = external;
    internal.n = reverseDetectorNormal(external.n);
    return internal;
}

inline SDetectorCoordinateFrame internalToExternalRightHanded(
    const SDetectorCoordinateFrame& internal)
{
    SDetectorCoordinateFrame external = internal;
    external.n = reverseDetectorNormal(internal.n);
    return external;
}

// Pose scalars are expressed in the detector-local U/N/V frame. Reversing N
// changes the sign of normal translation and rotation about N. U/V offsets
// and rotations about U/V keep their physical direction.
inline SDetectorPoseSpec externalRightHandedToInternal(
    const SDetectorPoseSpec& external)
{
    SDetectorPoseSpec internal = external;
    internal.offset_unv_mm.y = -external.offset_unv_mm.y;
    internal.tilt_n_rad = -external.tilt_n_rad;
    return internal;
}

inline SDetectorPoseSpec internalToExternalRightHanded(
    const SDetectorPoseSpec& internal)
{
    SDetectorPoseSpec external = internal;
    external.offset_unv_mm.y = -internal.offset_unv_mm.y;
    external.tilt_n_rad = -internal.tilt_n_rad;
    return external;
}

} // namespace YK
