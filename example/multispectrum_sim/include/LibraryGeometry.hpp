#pragma once

#include "config/SimConfig.hpp"
#include "YKCBCT/interface/YkReconstructionApi.hpp"
#include <vector>

namespace yk::spectral {

enum class GeometryUse {
    PhantomProjection,
    Reconstruction,
};

// 将示例 TOML 的四类宏观几何转换成 DLL 公共系统描述。
// 正投和重建必须调用同一个 builder，避免 SID/SDD、offset 或体素中心
// 在两个路径中各自解释而产生不一致。
YK::SSystemSpec makeLibrarySystem(const SimulationConfig& config,
    YK::EPipeline pipeline, YK::EFdkFilter filter = YK::EFdkFilter::RamLak,
    GeometryUse use = GeometryUse::PhantomProjection);

// Per-view flat ConeVec geometry used by the spectral projector. Standard
// geometries return an empty vector and continue through the legacy path.
std::vector<YK::SConeProjGeomVec> buildNonStandardProjectionViews(
    const SimulationConfig& config);

}
