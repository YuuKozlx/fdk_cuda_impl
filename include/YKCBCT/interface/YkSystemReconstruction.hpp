#pragma once

#include "YKCBCT/interface/YkReconstructionTypes.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"

namespace YK {

// 算法级配置与系统几何分离，便于同一套 Flat/Cyl geometry 复用到不同算法。
struct SReconstructionSpec {
    EPipeline pipeline = EPipeline::FDK;
    EProjectionModel projection_model = EProjectionModel::Joseph;
    SFdkAlgoParams fdk{};
    SIterAlgoParams iterative{};
    SCglsAlgoParams cgls{};
    SPwlsAlgoParams pwls{};
    STigreGradientAlgoParams tigre{};
    SWfbpAlgoParams wfbp{};
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
    if (reconstruction.fdk.parker.mode == EParkerMode::Enabled) return true;
    if (reconstruction.fdk.parker.mode == EParkerMode::Disabled) return false;
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr float kTolerance = 1e-5f;
    return regularScanRangeRad(system.circular) < kTwoPi - kTolerance;
}

} // namespace YK
