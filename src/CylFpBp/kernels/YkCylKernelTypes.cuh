#pragma once
#include "CylFpBp/YkCylFpBpTypes.hpp"

namespace YK::CylFpBp::detail {
// prepare 阶段预计算的圆柱探测器只读几何，避免每条射线重复归一化。
struct SKernelView {
    float4 source{}, cylinder_center{}, radial_unit{}, tangent_unit{}, detector_v{};
    float radius_mm = 0.f, principal_u = 0.f, principal_v = 0.f;
    float channel_angle_step_rad = 0.f;
};
struct SVoxelDrivenView {
    float4 source{}, cylinder_center{}, radial_unit{}, tangent_unit{}, axis_unit{};
    float radius_mm = 0.f, principal_u = 0.f, principal_v = 0.f;
    float inv_channel_angle_step_rad = 0.f, inv_row_step_mm = 0.f;
};
// R=SDD 的解析圆柱 BP 参数；prepare 已验证圆柱轴经过源点。
struct SCylFdkView {
    float4 source{}, radial_unit{}, tangent_unit{}, axis_unit{};
    float radius_mm = 0.f, sid_mm = 0.f, principal_u = 0.f, principal_v = 0.f;
    float inv_channel_angle_step_rad = 0.f, inv_row_step_mm = 0.f;
    float inverse_pixel_area = 0.f;
};
} // namespace YK::CylFpBp::detail
