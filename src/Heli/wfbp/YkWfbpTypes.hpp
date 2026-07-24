#pragma once

#include <cuda_runtime.h>

#include "global/YkGlobals.h"

namespace YK { namespace Helical { namespace Wfbp {

// FreeCT_wFBP 的原始扫描器模型是等角弧形探测器。平板入口是本工程的
// 输入适配层：先严格插值到等角弧面，之后所有重排、滤波和反投公式均走
// FreeCT 路径。
enum class EInputDetector {
    FlatPanel,
    EquiangularArc
};

// 与 FreeCT_wFBP 的四条重排路径一一对应：n/p/z/a FFS。
enum class EFocalSpotMode {
    None,
    Phi,
    Z,
    PhiAndZ
};

struct FreeCtFilterConfig {
    // FreeCT generate_filter() 中的 c（截止比例）与 a（窗混合比例）。
    // c=1,a=1 对应官方默认的全带宽 ramp。
    float cutoff_c = 1.f;
    float apodization_a = 1.f;
};

struct Config {
    EInputDetector input_detector = EInputDetector::FlatPanel;
    // angle_list/views_per_rot 始终描述原始交错采集帧；FreeCT 重排后的
    // 有效 views/turn 等于原始帧数除以焦点数，角度仍只取自 angle_list。
    EFocalSpotMode focal_spot_mode = EFocalSpotMode::None;

    // FreeCT 固定使用二倍通道过采样。保留字段只用于显式校验，避免调用方
    // 误以为任意倍率都属于官方算法。
    int channel_oversampling = 2;
    float redundancy_flat = 0.6f;
    float angle_tolerance = 1e-3f;

    // 弧形输入的相邻通道扇角和主射线通道。负主通道表示使用探测器中心。
    float arc_channel_angle_step_rad = 0.f;
    float arc_principal_channel = -1.f;

    // z-FFS/联合 FFS 使用。FreeCT 根据阳极角与等中心层厚推导焦点 z 偏移。
    float anode_angle_rad = 0.f;
    bool reverse_row_interleave = false;

    FreeCtFilterConfig filter = {};
    SKernelLaunchPolicy launch = {};
};

struct Geometry {
    int input_channels = 0;
    int output_channels = 0;
    int input_rows = 0;
    int rows = 0;
    int raw_views = 0;
    int sequence_views = 0;
    int views = 0;
    int views_per_turn = 0;
    int raw_views_per_turn = 0;
    int add_projections = 0;
    int focal_spot_count = 1;
    int phi_spot_count = 1;
    int z_spot_count = 1;

    float first_angle = 0.f;
    float raw_angle_step = 0.f;
    float angle_step = 0.f;
    float central_channel = 0.f;
    float central_row = 0.f;
    float raw_central_row = 0.f;
    float parallel_center = 0.f;
    float fan_angle_step = 0.f;
    float parallel_spacing = 0.f;
    float cone_half_angle = 0.f;
    float sid = 0.f;
    float sdd = 0.f;
    float pitch = 0.f;
    float start_z = 0.f;
    float phi_ffs_shift = 0.f;
    float z_ffs_shift = 0.f;
    int reverse_row_interleave = 0;
    EFocalSpotMode focal_spot_mode = EFocalSpotMode::None;
};

} } } // namespace YK::Helical::Wfbp
