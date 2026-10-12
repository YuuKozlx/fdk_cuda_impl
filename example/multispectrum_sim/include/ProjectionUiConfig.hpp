#pragma once
#include "config/SimConfig.hpp"
#include <string_view>
namespace yk::spectral {
void saveProjectionUiConfig(const SimulationConfig&, const std::filesystem::path&, WorkflowMode mode=WorkflowMode::Project);
void selectBuiltinMaterial(MaterialSpec&, std::string_view id);
void validateProjectionUiInputs(const SimulationConfig&, const std::vector<std::uint8_t>&);
}
