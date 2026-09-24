#pragma once
#include <cstdint>

namespace YK {

enum class ETigreGradientMethodSpec : int32_t {
    Sart,
    OsSart,
    Sirt,
    AsdPocs,
    OsAsdPocs,
    BAsdPocsBeta,
    Pcsd,
    OsPcsd,
    AwPcsd,
    OsAwPcsd,
    AwAsdPocs,
    OsAwAsdPocs,
};

struct STigreGradientAlgoParams {
    ETigreGradientMethodSpec method = ETigreGradientMethodSpec::OsSart;
    int block_size = 20;
    float lambda_reduction = 1.f;
    bool nesterov_relaxation = false;
    bool fdk_initialization = false;
    bool non_negative = true;
    int tv_iterations = 20;
    float tv_alpha = 0.002f;
    float tv_alpha_reduction = 0.95f;
    float maximum_update_ratio = 0.95f;
    float max_l2_error = -1.f;
};

} // namespace YK
