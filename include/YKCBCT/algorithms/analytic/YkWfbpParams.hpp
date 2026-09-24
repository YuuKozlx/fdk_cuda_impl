#pragma once
#include <cstdint>

namespace YK {

struct SWfbpAlgoParams {
    float redundancy_flat = 0.6f;
    float filter_cutoff = 1.f;
    float filter_apodization = 1.f;
};

} // namespace YK
