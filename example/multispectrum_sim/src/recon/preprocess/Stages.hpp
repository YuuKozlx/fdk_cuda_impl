#pragma once
#include "recon/preprocess/PreprocessPipeline.hpp"
namespace yk::spectral::preprocess {
std::vector<std::unique_ptr<IStage>> makeStages(const SimulationConfig &,
                                                const BeamHardeningConfig &);
}
