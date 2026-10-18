#pragma once

#include "input/config/SimConfig.hpp"
#include "input/materials/SpectralModel.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace yk::spectral {
class IProjectionObserver;

class CudaPrimaryProjector {
public:
    bool run(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&, const std::filesystem::path&,
        std::string&, IProjectionObserver* observer = nullptr) const;
};

}
