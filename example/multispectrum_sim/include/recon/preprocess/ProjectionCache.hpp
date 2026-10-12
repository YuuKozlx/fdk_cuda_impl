#pragma once
#include "config/SimConfig.hpp"
#include "TaskProgress.hpp"
#include <functional>
namespace yk::spectral::preprocess {
using ProjectionBuilder = std::function<bool(const std::filesystem::path&, std::string&)>;
// Persistent float32 log-attenuation cache; independent of reconstruction settings.
std::filesystem::path cachedProjection(const SimulationConfig&, const std::filesystem::path&,
    const ProjectionBuilder&, std::string&, TaskProgressCallback = {});
}
