#pragma once

#include <cuda_runtime.h>

#include "global/YkGlobals.h"

namespace YK::CylFpBp {

// ASTRA cyl_cone_vec 风格的圆柱探测器约定，并显式增加 principal_u/v：
// - detector_principal 是主射线与圆柱表面的交点；
// - detector_u_tangent 是主射线处沿圆周的一个像素弧长向量；
// - detector_v 是一个 row 的轴向步长；
// - radius_mm 是圆柱半径。圆柱轴线与 detector_v 平行。
//
// 该结构独立于平面 SConeProjGeomVec，避免 detS/detU 在两种表面中
// 出现不同含义。principal_u/v 允许主射线落在非整数像素位置。
struct SCylConeProjGeomVec {
    float4 source{};
    float4 detector_principal{};
    float4 detector_u_tangent{};
    float4 detector_v{};
    float4 angle{};
    float radius_mm = 0.f;
    float principal_u = 0.f;
    float principal_v = 0.f;
    float reserved = 0.f;
};

struct Config {
    // 每条射线沿最长体素轴每个体素中心采样一次。后续若要超采样，可在
    // 不改变几何结构的情况下扩展该字段。
    float samples_per_voxel = 1.f;
    SKernelLaunchPolicy launch{};
};

} // namespace YK::CylFpBp
