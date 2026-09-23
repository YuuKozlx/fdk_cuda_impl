#include "config/SimConfig.hpp"
#include "SimLogger.hpp"
#include "SpectralModel.hpp"
#include "XcomAttenuationProvider.hpp"
#include "ProjectionSimulator.hpp"
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

namespace {

    void listBuiltinMaterials()
    {
        std::cout << "id\tcategory\tdensity_g_cm3\tdisplay_name\n";
        for (auto material = yk::spectral::builtinMaterialsBegin();
            material != yk::spectral::builtinMaterialsEnd(); ++material) {
            std::cout << material->id << '\t' << material->category << '\t'
                << material->density_g_cm3 << '\t' << material->display_name << '\n';
        }
    }

    bool printBuiltinMaterial(std::string_view name)
    {
        const auto* material = yk::spectral::findBuiltinMaterial(name);
        if (!material) return false;
        std::cout << "id=" << material->id << '\n'
            << "category=" << material->category << '\n'
            << "display_name=" << material->display_name << '\n'
            << "density_g_cm3=" << material->density_g_cm3 << '\n'
            << "aliases=" << material->aliases << '\n'
            << "mass_fractions=" << material->composition << '\n';
        return true;
    }

}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        const auto total_start = std::chrono::steady_clock::now();
        if (argc == 2 && std::string_view(argv[1]) == "--list-materials") {
            listBuiltinMaterials();
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--material-info") {
            if (!printBuiltinMaterial(argv[2])) {
                std::cerr << "未知内置材料: " << argv[2] << '\n';
                return 2;
            }
            return 0;
        }
        const bool validate_config = argc == 3 &&
            std::string(argv[1]) == "--validate-config";
        if (argc != 2 && !validate_config) {
            std::cerr << "用法: multispectrum_sim <config.toml>\n"
                << "   或: multispectrum_sim --validate-config <config.toml>\n"
                << "   或: multispectrum_sim --list-materials\n"
                << "   或: multispectrum_sim --material-info <id-or-alias>\n";
            return 2;
        }
        const std::filesystem::path config_path =
            validate_config ? argv[2] : argv[1];
        auto config = yk::spectral::loadConfig(config_path);
        if (validate_config) {
            std::cout << "valid_config=" << config_path.string() << '\n';
            return 0;
        }
        if (!config.runsProjection()) {
            for (const auto& path : { config.reconstruction.output_volume_file,
                     config.reconstruction.slice_prefix })
                if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            std::string error;
            if (!yk::spectral::reconstructWithLibraryFdk(config,
                config.reconstruction.input_projection_file,
                config.reconstruction.output_volume_file,
                config.reconstruction.slice_prefix, error))
                throw std::runtime_error(error);
            std::cout << "reconstruction_only volume="
                << config.reconstruction.output_volume_file << '\n';
            return 0;
        }
        auto spectrum = yk::spectral::loadSpectrum(config.projection.spectrum_file);
        auto labels = yk::spectral::loadLabelVolume(config.projection.label_volume);
        yk::spectral::XcomAttenuationProvider attenuation(config.projection.xcom_data_directory);
        yk::spectral::SpectralTransmissionModel model(
            config.projection.materials, spectrum, attenuation, config.projection.detector_effects);
        yk::spectral::ProjectionSimulator simulator(config, model);
        std::string error;
        for (const auto& path : { config.projection.output_file,
                 config.projection.energy_output_file,
                 config.reconstruction.output_volume_file,
                 config.reconstruction.slice_prefix }) {
            if (!path.empty() && !path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());
        }
        const auto projection_start = std::chrono::steady_clock::now();
        if (!simulator.runToFile(labels, config.projection.output_file, error))
            throw std::runtime_error(error);
        const auto projection_end = std::chrono::steady_clock::now();
        const auto projection_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            projection_end - projection_start).count();
        std::cout << std::setprecision(10)
            << "geometry=" << static_cast<int>(config.geometry.kind)
            << " phantom=labels"
            << " spectrum_points=" << spectrum.size()
            << " labels=" << labels.size()
            << " output=" << config.projection.output_file.string() << '\n';
        if (config.runsReconstruction()) {
            if (!yk::spectral::reconstructWithLibraryFdk(config,
                config.reconstruction.input_projection_file,
                config.reconstruction.output_volume_file,
                config.reconstruction.slice_prefix, error))
                throw std::runtime_error(error);
            const auto reconstruction_end = std::chrono::steady_clock::now();
            const auto reconstruction_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                reconstruction_end - projection_end).count();
            const auto& reconstruction_name = config.reconstruction.type == "analytic"
                ? config.reconstruction.analytic.pipeline
                : config.reconstruction.iterative.algorithm;
            std::cout << "reconstruction=" << reconstruction_name
                << " volume="
                << config.reconstruction.output_volume_file.string()
                << " slices=" << config.reconstruction.slice_prefix.string() << "-{axial,coronal,sagittal}.bmp\n";
            const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                reconstruction_end - total_start).count();
            yk::spectral::SimLogger::instance().info(
                "timing_ms projection=" + std::to_string(projection_ms) +
                " reconstruction=" + std::to_string(reconstruction_ms) +
                " total=" + std::to_string(total_ms));
        } else {
            const auto total_end = std::chrono::steady_clock::now();
            const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                total_end - total_start).count();
            yk::spectral::SimLogger::instance().info(
                "timing_ms projection=" + std::to_string(projection_ms) +
                " reconstruction=0 total=" + std::to_string(total_ms));
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "multispectrum_sim: " << e.what() << '\n';
        return 1;
    }
}
