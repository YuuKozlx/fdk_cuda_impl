#pragma once

#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"

namespace YK {

enum class EParkerMode : int32_t {
    Auto,       // 小于一整圈时启用，完整圆扫时关闭
    Disabled,
    Enabled,
};

// Parker 不保存扫描范围。范围由 system.scan 唯一派生，防止配置值与实际
// angles/geometry 分裂。Cyl 和迭代算法忽略该项。
struct SParkerScanSpec {
    EParkerMode mode = EParkerMode::Auto;
};

// 算法级配置与系统几何分离，便于同一套 Flat/Cyl geometry 复用到不同算法。
struct SReconstructionSpec {
    EPipeline pipeline = EPipeline::FDK;
    ETask forward_projector = ETask::FP_Joseph;
    // V3 在 Flat/Cyl 中均有明确实现；V2 只属于 Flat，不在柱面上下文中
    // 隐式替换为其他模型。
    ETask back_projector = ETask::BP_Joseph_v3;
    SFdkAlgoParams fdk{};
    SIterAlgoParams iterative{};
    SCglsAlgoParams cgls{};
    SPwlsAlgoParams pwls{};
    STigreGradientAlgoParams tigre{};
    SWfbpAlgoParams wfbp{};
    SCylAnalyticFdkParams cyl_analytic_fdk{};
    SParkerScanSpec parker{};
};

template <typename RegularScan>
inline float regularScanRangeRad(const RegularScan& scan)
{
    constexpr float kTwoPi = 6.28318530717958647692f;
    return kTwoPi * static_cast<float>(scan.total_views) /
        static_cast<float>(scan.views_per_turn);
}

inline bool resolveParkerEnabled(const SSystemConfig& system,
    const SReconstructionSpec& reconstruction)
{
    if (system.detector != EDetectorKind::Flat ||
        system.trajectory != ETrajectoryKind::Circular) return false;
    if (reconstruction.parker.mode == EParkerMode::Enabled) return true;
    if (reconstruction.parker.mode == EParkerMode::Disabled) return false;
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr float kTolerance = 1e-5f;
    return regularScanRangeRad(system.circular) < kTwoPi - kTolerance;
}

} // namespace YK
