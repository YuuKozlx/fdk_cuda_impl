#pragma once

#include <cmath>
#include <vector>

namespace YK::NonStandardFpBp {

struct TrajectorySampling {
    int views = 0;
    float start_parameter = 0.f;
    float range = 0.f;
    float direction = 1.f;
};

inline bool buildUniformTrajectoryParameters(
    const TrajectorySampling& sampling,
    std::vector<float>& parameters)
{
    if (sampling.views <= 0 || !std::isfinite(sampling.start_parameter) ||
        !std::isfinite(sampling.range) || !std::isfinite(sampling.direction) ||
        std::fabs(sampling.range) <= 0.f ||
        std::fabs(sampling.direction) <= 0.f) {
        parameters.clear();
        return false;
    }
    parameters.resize(static_cast<size_t>(sampling.views));
    const float step = sampling.range * sampling.direction /
        static_cast<float>(sampling.views);
    for (int i = 0; i < sampling.views; ++i)
        parameters[static_cast<size_t>(i)] =
            sampling.start_parameter + step * static_cast<float>(i);
    return true;
}

} // namespace YK::NonStandardFpBp
