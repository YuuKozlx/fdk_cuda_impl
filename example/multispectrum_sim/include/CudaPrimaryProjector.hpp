#pragma once

#include "config/SimConfig.hpp"
#include "SpectralModel.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace yk::spectral {

class CudaPrimaryProjector {
public:
    bool run(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&, const std::filesystem::path&,
        std::string&) const;
};

}
