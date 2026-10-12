#pragma once
#include "config/SimConfig.hpp"
namespace yk::spectral {
bool runImGuiProjection(const SimulationConfig&, const std::filesystem::path&, std::string&);
bool launchImGuiProjection(const std::filesystem::path& config_file, std::string&);
}
