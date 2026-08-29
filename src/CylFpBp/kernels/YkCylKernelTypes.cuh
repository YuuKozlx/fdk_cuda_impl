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
    float4 source{}, radial_unit{}, tangent_unit{}, axis_unit{}, depth_unit{};
    float radius_mm = 0.f, sid_mm = 0.f, principal_u = 0.f, principal_v = 0.f;
    float inv_channel_angle_step_rad = 0.f, inv_row_step_mm = 0.f;
    float inverse_pixel_area = 0.f;
    // 数组中心射线相对于穿过旋转中心的主射线扇角，用于支持 U offset。
    float detector_center_angle_rad = 0.f;
    // 探测器数组中心相对于源点的轴向坐标；V offset 不为零时不能省略。
    float detector_axial_offset_mm = 0.f;
    float dtheta = 0.f; // 当前视图在角度积分中的梯度权重
};
} // namespace YK::CylFpBp::detail
