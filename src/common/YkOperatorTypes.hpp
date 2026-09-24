#pragma once
#include "YKCBCT/interface/YkReconstructionTypes.hpp"

namespace YK {
    enum class ETask : int32_t {
        FP_Joseph = 1,
        FP_Siddon = 2,
        BP_Siddon_RayDriven = 4,
        BP_Siddon_VoxDriven = 5,
        BP_FDK = 6,
        BP_FDK_matched = 7,
        BP_Joseph = 8,   //
        BP_Joseph_v2 = 9,
        BP_Joseph_v3 = 10,
        // Flat 体素驱动 Siddon 的两个既有优化版本。追加枚举值以保持前面
        // 已公开任务的 ABI 数值不变；三者数值语义不同，不做隐式替换。
        BP_Siddon_VoxDriven_v2 = 13,
        BP_Siddon_VoxDriven_v3 = 14,
    };
struct ExecuteRequest {
    const float* angles = nullptr;
    int K = 0;
    Buffer projection{};
    Buffer volume{};
    bool clear_output = true;
    int iteration_count = 0;
};
}
