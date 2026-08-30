#pragma once

#include <vector>

#include <cuda_runtime.h>

namespace YK {

// 轨迹只描述焦点与标称探测器主点的运动，不包含探测器表面类型。
// 静态圆扫和螺旋扫描使用不同类型，避免用 pitch=0 隐式切换语义。
struct SCircularTrajectorySpec {
    std::vector<float> angles_rad{};
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float z_mm = 0.f;
    // 随机架旋转的局部 X/Y/Z 偏移；它只移动焦点，不隐式移动探测器。
    float3 source_offset_mm = make_float3(0.f, 0.f, 0.f);
};

struct SHelicalTrajectorySpec {
    std::vector<float> angles_rad{};
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float start_z_mm = 0.f;
    float pitch_mm_per_turn = 0.f;
    float3 source_offset_mm = make_float3(0.f, 0.f, 0.f);
};

// 探测器相对标称主点的局部姿态。offset 分量依次沿 U、表面法向 N、V；
// tilt 全部使用弧度，并依次绕 U、V、N 轴施加。
struct SDetectorPoseSpec {
    float3 offset_unv_mm = make_float3(0.f, 0.f, 0.f);
    float tilt_u_rad = 0.f;
    float tilt_v_rad = 0.f;
    float tilt_n_rad = 0.f;
};

// 单视图校准量相对公共轨迹/探测器配置生效。数组长度必须与 angles 一致，
// 因而角度仍只有 SCircularTrajectorySpec::angles_rad 一个真源。
struct SViewGeometryCalibration {
    float3 source_offset_mm = make_float3(0.f, 0.f, 0.f);
    SDetectorPoseSpec detector_pose{};
};

struct SFlatDetectorSpec {
    int channels = 0;
    int rows = 0;
    float channel_size_mm = 0.f;
    float row_size_mm = 0.f;
    SDetectorPoseSpec pose{};
};

struct SCylDetectorSpec {
    int channels = 0;
    int rows = 0;
    // 相邻通道在圆柱面上的物理弧长。
    float channel_arc_mm = 0.f;
    float row_size_mm = 0.f;
    float curvature_radius_mm = 0.f;
    SDetectorPoseSpec pose{};
};

// 轨迹层输出的探测器无关坐标架。detector_principal 是无 offset/tilt 时
// 中心射线与标称探测器表面的交点。
struct SScannerViewFrame {
    float3 source{};
    float3 detector_principal{};
    float3 tangent_u{};
    float3 axis_v{};
    float3 radial_n{};
    float angle_rad = 0.f;
};

} // namespace YK
