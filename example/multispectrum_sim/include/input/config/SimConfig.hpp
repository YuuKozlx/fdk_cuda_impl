#pragma once
#include "input/config/SimTypes.hpp"
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
    // Normalize explicit defaults, validate values, then check external resources.
    // Validation accepts in-memory examples and never serializes configuration.
    void normalizeConfig(SimulationConfig&);
    void validateConfig(const SimulationConfig&);
    void validateRuntimeInputs(const SimulationConfig&);
    void saveConfig(const SimulationConfig&, const std::filesystem::path&, WorkflowMode);
    inline void saveConfig(const SimulationConfig& c, const std::filesystem::path& path) { saveConfig(c,path,c.workflow.mode); }
    // GUI setup can defer missing runtime files; execution keeps strict validation.
    SimulationConfig loadConfig(const std::filesystem::path&, bool check_runtime_files = true);
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path&);
    void validateSpectrumSource(const SpectrumSourceConfig&);
    std::vector<SpectrumPoint> loadProjectionSpectrum(const ProjectionConfig&);
    // Center-energy format readable by loadSpectrum; not MCGPU edge format.
    void saveSpectrum(const std::vector<SpectrumPoint>&, const std::filesystem::path&);
    std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path&);
}
