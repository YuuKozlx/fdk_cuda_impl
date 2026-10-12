#pragma once

#include "config/SimConfig.hpp"
#include "TaskProgress.hpp"

#include <filesystem>
#include <string>

namespace yk::spectral {

bool prepareReconstructionProjection(const SimulationConfig& config,
                                     const std::filesystem::path& input_projection,
                                     const std::filesystem::path& output_projection,
                                     std::string& diagnostic, TaskProgressCallback progress={});

}
