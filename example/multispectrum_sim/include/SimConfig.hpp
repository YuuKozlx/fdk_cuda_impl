#pragma once
#include "SimTypes.hpp"
#include <filesystem>
namespace yk::spectral {
struct SimulationConfig {
    GeometryKind geometry=GeometryKind::FlatCbct;
    std::filesystem::path label_volume, spectrum_file, xcom_data_directory, output_file, path_cache_file;
    std::vector<MaterialSpec> materials; DetectorEffectsConfig detector_effects;
    GeometryConfig geometry_config;
    ReconstructionConfig reconstruction;
    bool use_library_fp = false;
    bool use_cuda_spectral = false;
};
SimulationConfig loadConfig(const std::filesystem::path&);
std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path&);
std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path&);
}
