#pragma once
#include "config/SimConfig.hpp"
#include "SpectralModel.hpp"
#include <vector>
#include <filesystem>
#include <string>
namespace yk::spectral {
bool reconstructWithLibraryFdk(const SimulationConfig&, const std::filesystem::path&,
                               const std::filesystem::path&, const std::filesystem::path&,
                               std::string& diagnostic);
}
namespace yk::spectral {
class ProjectionSimulator {
public:
    ProjectionSimulator(const SimulationConfig& config, const SpectralTransmissionModel& model)
        : config_(config), model_(model) {}
    bool runToFile(const std::vector<std::uint8_t>& labels,
                   const std::filesystem::path& output_file,
                   std::string& error) const;
private:
    SimulationConfig config_; const SpectralTransmissionModel& model_;
};
}
