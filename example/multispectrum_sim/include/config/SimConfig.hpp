#pragma once
#include "config/SimTypes.hpp"
#include <filesystem>
namespace yk::spectral {
    struct SimulationConfig {
        int schema_version = 2;
        std::filesystem::path source_config_file;
        WorkflowConfig workflow;
        VisualizationConfig visualization;
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
    // GUI setup can defer missing runtime files; execution keeps strict validation.
    SimulationConfig loadConfig(const std::filesystem::path&, bool check_runtime_files = true);
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path&);
    void validateSpectrumSource(const SpectrumSourceConfig&);
    std::vector<SpectrumPoint> loadProjectionSpectrum(const ProjectionConfig&);
    // Center-energy format readable by loadSpectrum; not MCGPU edge format.
    void saveSpectrum(const std::vector<SpectrumPoint>&, const std::filesystem::path&);
    std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path&);
}
