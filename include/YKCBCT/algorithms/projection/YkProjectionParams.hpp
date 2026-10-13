#pragma once
#include <cstdint>

namespace YK {

enum class EForwardProjector : int32_t {
    Automatic,
    Siddon,
    Joseph,
    SeparableFootprint,
};

enum class EBackProjector : int32_t {
    Automatic,
    FdkPseudoMatched,
    SiddonRayDriven,
    Joseph,
    JosephV3,
    SeparableFootprint,
};

} // namespace YK
