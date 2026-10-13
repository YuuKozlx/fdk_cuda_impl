#pragma once
#include <cstdint>
#include "YkAnalyticFilterParams.hpp"

namespace YK {

struct SWfbpAlgoParams {
    float redundancy_flat = 0.6f;
    SAnalyticFilterParams filter{};
};

} // namespace YK
