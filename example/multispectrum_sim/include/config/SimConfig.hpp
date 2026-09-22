#pragma once
#include "config/SimTypes.hpp"
#include <filesystem>
namespace yk::spectral {
    struct SimulationConfig {
        int schema_version = 2;
        WorkflowConfig workflow;
        GeometrySpec geometry;
        ProjectionConfig projection;
        ReconstructionConfig reconstruction;

        bool runsProjection() const {
            return workflow.mode == WorkflowMode::Project ||
                workflow.mode == WorkflowMode::ProjectAndReconstruct;
        }
        bool runsReconstruction() const {
            return workflow.mode == WorkflowMode::Reconstruct ||
                workflow.mode == WorkflowMode::ProjectAndReconstruct;
        }
    };
    SimulationConfig loadConfig(const std::filesystem::path&);
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path&);
    std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path&);
}
