#pragma once
#include "input/config/SimConfig.hpp"
#include <string_view>
namespace yk::spectral {
void selectBuiltinMaterial(MaterialSpec&, std::string_view id);
bool setProjectionUiScanType(SimulationConfig&, GeometryKind);
void validateProjectionUiInputs(const SimulationConfig&, const std::vector<std::uint8_t>&);
}
