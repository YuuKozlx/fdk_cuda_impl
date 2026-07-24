#pragma once

#include <vector>

#include "Heli/YkHeliCTParams.h"
#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "util/YkVecOperation.hpp"

namespace YK {

// 根据外部提供的累积角度生成逐视角螺旋锥束 geometry。
// angle_list 是唯一角度来源；源点和探测器中心沿 z 同步移动：
//   z(theta) = start_z_mm + pitch_mm * theta / (2*pi)
YK_INLINE void build_helical_vec_geometry(
    std::vector<SConeProjGeomVec>& geometry,
    const SHeliCTParam& params)
{
    const auto& angles = params.angle_list;
    geometry.resize(angles.size());

    const float idd = params.SDD - params.SID;
    float3 det_u = make_float3(1.f, 0.f, 0.f);
    float3 det_v = make_float3(0.f, 0.f, 1.f);
    {
        const float3 axis_u = det_u;
        det_u = f3_rot_axis(det_u, axis_u, params.tiltu_angle_rad);
        det_v = f3_rot_axis(det_v, axis_u, params.tiltu_angle_rad);
        const float3 axis_v = det_v;
        det_u = f3_rot_axis(det_u, axis_v, params.tiltv_angle_rad);
        det_v = f3_rot_axis(det_v, axis_v, params.tiltv_angle_rad);
        const float3 normal = f3_normalize(cross(det_u, det_v));
        det_u = f3_rot_axis(det_u, normal, params.tiltn_angle_rad);
        det_v = f3_rot_axis(det_v, normal, params.tiltn_angle_rad);
    }

    const float3 source0 = make_float3(0.f, -params.SID, 0.f);
    const float3 detector0 = make_float3(
        params.offsetU_mm, idd, params.offsetV_mm);
    const float3 principal0 = make_float3(0.f, 1.f, 0.f);
    const float center_u = 0.5f * (params.iPU - 1);
    const float center_v = 0.5f * (params.iPV - 1);

    for (size_t i = 0; i < angles.size(); ++i) {
        const float theta = angles[i];
        const float z = params.start_z_mm +
            params.pitch_mm * theta / (2.f * CUDA_PI);
        const float3 z_shift = make_float3(0.f, 0.f, z);
        const float3 source = f3_rotz(source0, theta) + z_shift;
        const float3 detector_center = f3_rotz(detector0, theta) + z_shift;
        const float3 principal = f3_rotz(principal0, theta);
        const float3 u = f3_rotz(det_u, theta) * params.du_mm;
        const float3 v = f3_rotz(det_v, theta) * params.dv_mm;
        const float3 detector_start = detector_center - u * center_u - v * center_v;

        geometry[i] = SConeProjGeomVec{
            f3_to_f4(source), f3_to_f4(principal), f3_to_f4(detector_start),
            f3_to_f4(u), f3_to_f4(v), make_float4(theta, 0.f, 0.f, 0.f)
        };
    }
}

} // namespace YK
