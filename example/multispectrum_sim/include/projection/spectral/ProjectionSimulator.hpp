#pragma once
#include "input/config/SimConfig.hpp"
#include "input/materials/SpectralModel.hpp"
#include <vector>
#include <filesystem>
#include <string>
namespace yk::spectral {
class IProjectionObserver;
class ProjectionSimulator {
public:
    ProjectionSimulator(const SimulationConfig& config, const SpectralTransmissionModel& model)
        : config_(config), model_(model) {}
    bool runToFile(const std::vector<std::uint8_t>& labels,
                   const std::filesystem::path& output_file,
                   std::string& error, IProjectionObserver* observer = nullptr) const;
private:
    SimulationConfig config_; const SpectralTransmissionModel& model_;
};
}
