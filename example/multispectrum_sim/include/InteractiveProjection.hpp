#pragma once
#include "SpectralModel.hpp"
#include "config/SimConfig.hpp"
#include <filesystem>
#include <string>
namespace yk::spectral {
bool runInteractiveProjection(const SimulationConfig&, const std::vector<std::uint8_t>&,
    const SpectralTransmissionModel&, const std::filesystem::path&, std::string&);
}
