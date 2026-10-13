#pragma once
#include <cstdint>
#include "YkAnalyticFilterParams.hpp"

namespace YK {

enum class EParkerMode : int32_t { Auto, Disabled, Enabled };

struct SParkerScanSpec {
    EParkerMode mode = EParkerMode::Auto;
};

struct SFdkAlgoParams {
    SParkerScanSpec parker{};
    SAnalyticFilterParams filter{};
};

} // namespace YK
