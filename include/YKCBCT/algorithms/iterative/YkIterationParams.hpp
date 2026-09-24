#pragma once
#include <cstdint>

namespace YK {

struct SIterAlgoParams {
    int   iterations = 10;
    float relaxation = 1.f;
    int   subsets = 1;
};

} // namespace YK
