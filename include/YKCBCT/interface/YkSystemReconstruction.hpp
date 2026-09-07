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
    ETask back_projector = ETask::BP_Joseph_v2;
    SFdkAlgoParams fdk{};
    SIterAlgoParams iterative{};
    SCglsAlgoParams cgls{};
    SPwlsAlgoParams pwls{};
    STigreGradientAlgoParams tigre{};
    SWfbpAlgoParams wfbp{};
    SCylAnalyticFdkParams cyl_analytic_fdk{};
    SParkerScanSpec parker{};
};

// 完整重建请求：宏观系统 geometry 加上一份算法配置。Geometry builder
// 负责填充 system；执行层只需消费 request.system 和 request.reconstruction。
template <typename SystemGeometry>
struct SSystemReconstructionRequest {
    SystemGeometry system{};
    SReconstructionSpec reconstruction{};
};

using SStaticFlatReconstructionRequest =
    SSystemReconstructionRequest<SStaticFlatSystemSpec>;
using SStaticCylReconstructionRequest =
    SSystemReconstructionRequest<SStaticCylSystemSpec>;
using SHelicalFlatReconstructionRequest =
    SSystemReconstructionRequest<SHelicalFlatSystemSpec>;
using SHelicalCylReconstructionRequest =
    SSystemReconstructionRequest<SHelicalCylSystemSpec>;

template <typename RegularScan>
inline float regularScanRangeRad(const RegularScan& scan)
{
    constexpr float kTwoPi = 6.28318530717958647692f;
    return kTwoPi * static_cast<float>(scan.total_views) /
        static_cast<float>(scan.views_per_turn);
}

inline bool resolveParkerEnabled(const SStaticFlatReconstructionRequest& request)
{
    if (request.reconstruction.parker.mode == EParkerMode::Enabled) return true;
    if (request.reconstruction.parker.mode == EParkerMode::Disabled) return false;
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr float kTolerance = 1e-5f;
    return regularScanRangeRad(request.system.scan) < kTwoPi - kTolerance;
}

} // namespace YK
