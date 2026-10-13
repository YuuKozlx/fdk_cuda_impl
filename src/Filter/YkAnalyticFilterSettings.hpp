#pragma once
#include "YKCBCT/algorithms/analytic/YkAnalyticFilterParams.hpp"
#include "global/YkFilterTypes.hpp"
#include <cmath>
namespace YK::Filter {
inline bool validAnalyticFilter(const SAnalyticFilterParams& p) {
    return int(p.type)>=int(EFdkFilter::None) && int(p.type)<=int(EFdkFilter::Tukey) &&
        std::isfinite(p.cutoff) && p.cutoff>0 && p.cutoff<=.5f &&
        std::isfinite(p.gain) && p.gain>0 &&
        std::isfinite(p.butterworth_order) && p.butterworth_order>0 &&
        std::isfinite(p.kaiser_beta) && p.kaiser_beta>=0 &&
        std::isfinite(p.tukey_alpha) && p.tukey_alpha>=0 && p.tukey_alpha<=1;
}
inline SFilterKernelDesc analyticFilterDesc(const SAnalyticFilterParams& p,
    EWeightsBuildSource source=EWeightsBuildSource::AnalyticFreq) {
    SFilterKernelDesc d;
    d.kind=static_cast<EFilterKernel>(p.type);
    d.source=source; d.cutoff=p.cutoff; d.gain=p.gain;
    d.order=p.butterworth_order; d.beta=p.kaiser_beta; d.tukey_alpha=p.tukey_alpha;
    return d;
}
} // namespace YK::Filter
