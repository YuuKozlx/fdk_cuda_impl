#pragma once
#include "input/materials/SpectralModel.hpp"
#include "input/config/SimConfig.hpp"
#include <filesystem>
#include <string>
namespace yk::spectral {
bool runInteractiveProjection(const SimulationConfig&, const std::vector<std::uint8_t>&,
    const SpectralTransmissionModel&, const std::filesystem::path&, std::string&);
}
