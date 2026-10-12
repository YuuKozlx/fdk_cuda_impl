#pragma once
#include "config/SimConfig.hpp"
#include "TaskProgress.hpp"
namespace yk::spectral::preprocess {
BeamHardeningConfig fitWaterPlate(const SimulationConfig &, const BeamHardeningConfig &,
                                  const TaskProgressCallback &);
}
