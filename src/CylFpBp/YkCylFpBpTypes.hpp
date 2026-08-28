#pragma once

#include <cuda_runtime.h>

#include "global/YkGlobals.h"
#include "global/YkKernelLaunchPolicy.hpp"

namespace YK::CylFpBp {

struct Config {
    // 每条射线沿最长体素轴每个体素中心采样一次。后续若要超采样，可在
    // 不改变几何结构的情况下扩展该字段。
    float samples_per_voxel = 1.f;
    SKernelLaunchPolicy launch{};
};

} // namespace YK::CylFpBp
