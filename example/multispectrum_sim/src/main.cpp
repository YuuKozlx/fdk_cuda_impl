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
            const auto* preset = yk::spectral::findBuiltinMaterial(material.preset);
            return material.name == "water" || material.formula == "H2O" ||
                (preset && std::string_view(preset->id) == "compound.water");
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
        const bool water_cylinder = argc == 3 &&
            std::string(argv[1]) == "--water-cylinder";
        const bool reconstruction_only = argc == 3 &&
            std::string(argv[1]) == "--reconstruct-only";
        const bool export_water_cylinder = argc == 4 &&
            std::string(argv[1]) == "--export-water-cylinder";
        if (argc != 2 && !water_cylinder && !export_water_cylinder && !reconstruction_only) {
            std::cerr << "用法: multispectrum_sim <config.toml>\n"
                << "   或: multispectrum_sim --water-cylinder <config.toml>\n"
                << "   或: multispectrum_sim --reconstruct-only <config.toml>\n"
                << "   或: multispectrum_sim --export-water-cylinder <config.toml> <labels.raw>\n"
                << "   或: multispectrum_sim --list-materials\n"
                << "   或: multispectrum_sim --material-info <id-or-alias>\n";
            return 2;
        }
        const std::filesystem::path config_path =
            water_cylinder || export_water_cylinder || reconstruction_only ? argv[2] : argv[1];
        auto config = yk::spectral::loadConfig(config_path);
        if (reconstruction_only) {
            if (!config.reconstruction.enabled)
                throw std::runtime_error("--reconstruct-only requires reconstruction.enabled=true");
            for (const auto& path : {config.reconstruction.output_volume_file,
                     config.reconstruction.slice_prefix})
                if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            std::string error;
            if (!yk::spectral::reconstructWithLibraryFdk(config, config.output_file,
                    config.reconstruction.output_volume_file, config.reconstruction.slice_prefix, error))
                throw std::runtime_error(error);
            std::cout << "reconstruction_only volume=" << config.reconstruction.output_volume_file << '\n';
            return 0;
        }
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
            std::cout << "reconstruction=" << config.reconstruction.pipeline
                << " volume="
                << config.reconstruction.output_volume_file.string()
                << " slices=" << config.reconstruction.slice_prefix.string() << "-{axial,coronal,sagittal}.bmp\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "multispectrum_sim: " << e.what() << '\n';
        return 1;
    }
}
