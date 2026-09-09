#include "SimConfig.hpp"
#include "SpectralModel.hpp"
#include "XcomAttenuationProvider.hpp"
#include "ProjectionSimulator.hpp"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace {

std::vector<std::uint8_t> makeWaterCylinder(const yk::spectral::SimulationConfig& config)
{
    const auto& g = config.geometry_config;
    if (config.materials.empty())
        throw std::runtime_error("纯水圆柱模体需要至少一个材料标签");
    if (g.volume_x <= 0 || g.volume_y <= 0 || g.volume_z <= 0 ||
        g.voxel_x_mm <= 0 || g.voxel_y_mm <= 0)
        throw std::runtime_error("纯水圆柱模体要求有效的体积和体素尺寸");

    // 圆柱占据 XY 平面，沿 Z 方向贯穿整个体积；半径取较短横向边的
    // 40%，留出背景区域，便于观察中心射线和边界衰减。
    const double width = g.volume_x * g.voxel_x_mm;
    const double height = g.volume_y * g.voxel_y_mm;
    const double radius = 0.4 * std::min(width, height);
    const auto water = std::find_if(config.materials.begin(), config.materials.end(),
        [](const yk::spectral::MaterialSpec& material) {
            return material.name == "water" || material.formula == "H2O";
        });
    if (water == config.materials.end())
        throw std::runtime_error("纯水圆柱模体需要 name=water 或 formula=H2O 的材料");
    const std::uint8_t water_label = water->label;
    const double cx = 0.5 * (g.volume_x - 1);
    const double cy = 0.5 * (g.volume_y - 1);
    std::vector<std::uint8_t> labels(static_cast<std::size_t>(g.volume_x) *
        g.volume_y * g.volume_z, 0);
    for (int z = 0; z < g.volume_z; ++z) {
        for (int y = 0; y < g.volume_y; ++y) {
            for (int x = 0; x < g.volume_x; ++x) {
                const double dx = (x - cx) * g.voxel_x_mm;
                const double dy = (y - cy) * g.voxel_y_mm;
                if (dx * dx + dy * dy <= radius * radius)
                    labels[(static_cast<std::size_t>(z) * g.volume_y + y) *
                        g.volume_x + x] = water_label;
            }
        }
    }
    return labels;
}

}

int main(int argc, char** argv)
{
    try {
        const bool water_cylinder = argc == 3 &&
            std::string(argv[1]) == "--water-cylinder";
        const bool export_water_cylinder = argc == 4 &&
            std::string(argv[1]) == "--export-water-cylinder";
        if (argc != 2 && !water_cylinder && !export_water_cylinder) {
            std::cerr << "用法: multispectrum_sim <config.toml>\n"
                << "   或: multispectrum_sim --water-cylinder <config.toml>\n"
                << "   或: multispectrum_sim --export-water-cylinder <config.toml> <labels.raw>\n";
            return 2;
        }
        const std::filesystem::path config_path =
            water_cylinder || export_water_cylinder ? argv[2] : argv[1];
        auto config = yk::spectral::loadConfig(config_path);
        if (export_water_cylinder) {
            const auto labels = makeWaterCylinder(config);
            const std::filesystem::path output = argv[3];
            if (!output.parent_path().empty())
                std::filesystem::create_directories(output.parent_path());
            std::ofstream stream(output, std::ios::binary);
            if (!stream || !stream.write(reinterpret_cast<const char*>(labels.data()),
                    static_cast<std::streamsize>(labels.size())))
                throw std::runtime_error("无法写入外部标签体: " + output.string());
            std::cout << "exported_water_cylinder=" << output.string()
                << " labels=" << labels.size() << '\n';
            return 0;
        }
        auto spectrum = yk::spectral::loadSpectrum(config.spectrum_file);
        auto labels = water_cylinder
            ? makeWaterCylinder(config)
            : yk::spectral::loadLabelVolume(config.label_volume);
        yk::spectral::XcomAttenuationProvider attenuation(config.xcom_data_directory);
        yk::spectral::SpectralTransmissionModel model(
            config.materials, spectrum, attenuation, config.detector_effects);
        yk::spectral::ProjectionSimulator simulator(config, model);
        std::string error;
        for (const auto& path : {config.output_file, config.path_cache_file,
                 config.reconstruction.output_volume_file,
                 config.reconstruction.slice_prefix}) {
            if (!path.empty() && !path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());
        }
        if (!simulator.runToFile(labels, config.output_file, error))
            throw std::runtime_error(error);
        std::cout << std::setprecision(10)
            << "geometry=" << static_cast<int>(config.geometry)
            << " phantom=" << (water_cylinder ? "water-cylinder" : "labels")
            << " spectrum_points=" << spectrum.size()
            << " labels=" << labels.size()
            << " output=" << config.output_file.string() << '\n';
        if (config.reconstruction.enabled) {
            if (!config.use_library_fp)
                throw std::runtime_error(
                    "reconstruction.enabled 要求 simulation.use_library_fp=true，以便调用 DLL FDK");
            if (!yk::spectral::reconstructWithLibraryFdk(config, config.output_file,
                    config.reconstruction.output_volume_file,
                    config.reconstruction.slice_prefix, error))
                throw std::runtime_error(error);
            std::cout << "reconstruction=fdk volume="
                << config.reconstruction.output_volume_file.string()
                << " slices=" << config.reconstruction.slice_prefix.string() << "-{axial,coronal,sagittal}.bmp\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "multispectrum_sim: " << e.what() << '\n';
        return 1;
    }
}
