#pragma once

#include "YKCBCT/algorithms/projection/YkProjectionParams.hpp"
#include "common/YkOperatorTypes.hpp"

namespace YK::detail {

// Automatic choices are intentionally policy, not aliases. Ordinary algebraic
// reconstruction favours the measured practical combination, while solvers
// whose mathematics requires A^T use the verified Siddon pair.
inline ETask resolveForwardProjector(EForwardProjector selection,
    bool cylindrical, bool strict_adjoint)
{
    switch (selection) {
    case EForwardProjector::Siddon: return ETask::FP_Siddon;
    case EForwardProjector::Joseph: return ETask::FP_Joseph;
    case EForwardProjector::SeparableFootprint:
        return ETask::FP_SeparableFootprint;
    case EForwardProjector::Automatic:
        return cylindrical && !strict_adjoint
            ? ETask::FP_Joseph : ETask::FP_Siddon;
    }
    return ETask::FP_Siddon;
}

inline ETask resolveBackProjector(EBackProjector selection,
    bool cylindrical, bool strict_adjoint)
{
    switch (selection) {
    case EBackProjector::FdkPseudoMatched:
        return ETask::BP_FDK_PseudoMatched;
    case EBackProjector::SiddonRayDriven:
        return ETask::BP_Siddon_RayDriven;
    case EBackProjector::Joseph: return ETask::BP_Joseph;
    case EBackProjector::JosephV3: return ETask::BP_Joseph_v3;
    case EBackProjector::SeparableFootprint:
        return ETask::BP_SeparableFootprint;
    case EBackProjector::Automatic:
        if (strict_adjoint) return ETask::BP_Siddon_RayDriven;
        return cylindrical ? ETask::BP_Joseph_v3
                           : ETask::BP_FDK_PseudoMatched;
    }
    return ETask::BP_FDK_PseudoMatched;
}

} // namespace YK::detail
