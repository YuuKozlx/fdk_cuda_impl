#pragma once
#include <cstdint>

namespace YK {

struct SCglsAlgoParams {
    int minimum_iterations = 0;
    int check_interval = 1;
    int patience = 1;
    float relative_residual = 0.f;
    bool robust_restart = true;
};

} // namespace YK
