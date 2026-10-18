#pragma once

#include "input/config/SimConfig.hpp"
#include "processing/support/TaskProgress.hpp"

#include <filesystem>
#include <string>

namespace yk::spectral {

bool prepareReconstructionProjection(const SimulationConfig& config,
                                     const std::filesystem::path& input_projection,
                                     const std::filesystem::path& output_projection,
                                     std::string& diagnostic, TaskProgressCallback progress={});

}
