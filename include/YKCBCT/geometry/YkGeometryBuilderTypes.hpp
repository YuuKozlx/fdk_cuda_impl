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

// 规则螺旋扫描的宏观采集参数。常规用户只需要设置这一层，builder 会
// 自动生成 angles_rad 和逐视图 geometry。显式 SHelicalTrajectorySpec
// 仍用于非等间隔角度、外部编码器角度或逐视图校准数据。
struct SRegularHelicalScanSpec {
    int total_views = 0;
    int views_per_turn = 0;
    float start_angle_rad = 0.f;
    // +1/-1 分别表示角度递增/递减。
    int rotation_direction = 1;
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float start_z_mm = 0.f;
    // 带符号的物理进床量。正负号决定 Z 方向，与机架旋转方向无关。
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

// 重建体积的宏观网格。它属于系统配置，但不参与射线几何生成；单独保存
// 可避免将体积中心错误地当作旋转中心或探测器校正量。
struct SVolumeGridSpec {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float voxel_x_mm = 0.f;
    float voxel_y_mm = 0.f;
    float voxel_z_mm = 0.f;
    float3 center_mm = make_float3(0.f, 0.f, 0.f);
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
