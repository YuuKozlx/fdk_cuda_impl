#pragma once
#include "SpectralProjectionPipeline.hpp"
namespace yk::spectral::primary_detail {
std::vector<std::unique_ptr<IProjectionStage>> makeProjectionStages(
    const SimulationConfig&);
} // namespace yk::spectral::primary_detail
