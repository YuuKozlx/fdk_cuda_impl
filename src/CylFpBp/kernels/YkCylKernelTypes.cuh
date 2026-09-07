#pragma once
#include "CylFpBp/YkCylFpBpTypes.hpp"

namespace YK::CylFpBp::detail {
// prepare 阶段预计算的圆柱探测器只读几何，避免每条射线重复归一化。
struct SKernelView {
    float4 source{}, cylinder_center{}, radial_unit{}, tangent_unit{}, detector_v{};
    float radius_mm = 0.f, principal_u = 0.f, principal_v = 0.f;
    float channel_angle_step_rad = 0.f;
};

// 严格 Siddon FP/BP 共用的逐通道射线基向量。它表示从源点指向
// principal_v 所在探测器行的像素中心，运行时只需叠加行方向偏移。
// 这里只缓存圆柱像素中心的等价展开，不改变 ray-driven Siddon 的
// 射线集合、体素遍历顺序或真实交长定义。
struct SSiddonChannelRay {
    float4 ray_at_principal_row{};
};
struct SVoxelDrivenView {
    float4 source{}, cylinder_center{}, radial_unit{}, tangent_unit{}, axis_unit{};
    float radius_mm = 0.f, principal_u = 0.f, principal_v = 0.f;
    float inv_channel_angle_step_rad = 0.f, inv_row_step_mm = 0.f;
    // 圆柱求交的每视图不变量，避免 V2/V3 对每个体素重复分解源点。
    float4 source_perp{};
    float source_perp_norm_sq = 0.f;
    unsigned int source_on_axis = 0;
};
// V3 离散像素射线在 prepare 阶段缓存，避免 kernel 内重复 sincosf。
struct SCylVoxelChannelRay { float4 detector_at_principal_row{}; };
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
    float parker_beta_rad = 0.f;
    float parker_scan_range_rad = 0.f;
    float parker_redundancy_half_rad = 0.f;
    unsigned int parker_enabled = 0;
};
} // namespace YK::CylFpBp::detail
