#pragma once

#include "SimConfig.hpp"
#include "YKCBCT/interface/YkReconstructionApi.hpp"

namespace yk::spectral {

enum class GeometryUse {
    PhantomProjection,
    Reconstruction,
};

// 将示例 TOML 的四类宏观几何转换成 DLL 公共系统描述。
// 正投和重建必须调用同一个 builder，避免 SID/SDD、offset 或体素中心
// 在两个路径中各自解释而产生不一致。
YK::SSystemSpec makeLibrarySystem(const SimulationConfig& config,
    YK::EPipeline pipeline, YK::ETask forward_projector = YK::ETask::FP_Joseph,
    YK::EFdkFilter filter = YK::EFdkFilter::RamLak,
    GeometryUse use = GeometryUse::PhantomProjection);

}
