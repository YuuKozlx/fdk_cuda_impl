#pragma once
#include "input/config/SimConfig.hpp"
#include "processing/support/TaskProgress.hpp"
namespace yk::spectral::calibration {
// Independent calibration task. Correction pipelines never call this producer.
BeamHardeningConfig fitWaterPlate(const SimulationConfig&, const BeamHardeningConfig&, const TaskProgressCallback&);
BeamHardeningConfig calibrateWaterBhc(const SimulationConfig&, const std::filesystem::path&, TaskProgressCallback = {});
}
