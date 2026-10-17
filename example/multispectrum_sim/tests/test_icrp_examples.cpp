#include "ProjectionUiConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include "examples/IcrpExamples.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <toml++/toml.hpp>
using namespace yk::spectral;
namespace {
void require(bool ok, const char* what)
{
    if (!ok)
        throw std::runtime_error(what);
}
}
int main(int argc, char** argv)
{
    try {
        const auto root = std::filesystem::absolute(argc > 1 ? argv[1] : "icrp-tests");
        std::filesystem::create_directories(root);
        const auto table = root / "materials.toml";
        {
            std::ofstream out(table);
            out << "[[materials]]\nfile='air_1-200keV.mcgpu.gz'\ndensity_g_cm3=0.001205\nvoxel_ids=[0,2]\n";
            for (int i = 1; i < 38; ++i)
                out << "[[materials]]\nfile='water_1-200keV.mcgpu.gz'\ndensity_g_cm3=" << 1 + i * .01 << "\nvoxel_ids=[" << i + 2 << "]\n";
            out << "[[materials]]\nfile='CsI_1-200keV.mcgpu.gz'\ndensity_g_cm3=4.51\nvoxel_ids=[254]\n";
        }
        const auto map = examples::loadIcrpMaterialMap(table);
        require(map.materials.size() == 38 && map.labels[39] == 37 && map.labels[2] == 0, "ICRP tissue groups or detector exclusion");
        for (const auto* sex : { "male", "female" }) {
            const auto full = examples::loadIcrpMaterialMap(std::filesystem::path(MULTISPECTRUM_ICRP_MATERIAL_DIRECTORY) / (std::string("materials-") + sex + ".toml"));
            require(full.materials.size() == 53, "full ICRP tissue/density mapping lost");
            for (int medium = 1; medium <= 52; ++medium) {
                const auto* preset = findBuiltinMaterial(std::string("icrp145.") + sex + "." + std::to_string(medium));
                require(preset != nullptr, "missing ICRP medium");
                std::istringstream input(preset->composition);
                std::string item;
                double sum = 0;
                while (std::getline(input, item, ','))
                    sum += std::stod(item.substr(item.find(':') + 1));
                require(std::abs(sum - 1) < .0015, "ICRP mass fractions fail normalization");
            }
            const auto* bone = findBuiltinMaterial(std::string("icrp145.") + sex + ".5");
            require(std::abs(bone->density_g_cm3 - 1.904) < 1e-9 && std::string(bone->composition).find("20:0.213") != std::string::npos, "ICRP cortical bone composition");
            require(full.labels[0] >= 0 && full.labels[28] >= 0 && full.labels[169] >= 0, "ICRP original organ labels lost");
            require(full.labels[29] != full.labels[33], "different skeletal media merged");
            require(full.labels[169] == full.labels[3], "BB source-medium erratum was not applied");
            require(full.labels[168] != full.labels[169], "bronchial tissue treated as air");
        }
        require(findBuiltinMaterial("icrp145.male.10")->density_g_cm3 != findBuiltinMaterial("icrp145.female.10")->density_g_cm3, "sex-specific spongiosa lost");
        require(std::abs(map.materials.back().density_g_cm3 - 1.37) < 1e-9, "ICRP density lost");
        const auto raw = root / "source.raw", converted = root / "mapped.raw";
        {
            std::ofstream out(raw, std::ios::binary);
            const unsigned char data[] = { 0, 2, 3, 39 };
            out.write(reinterpret_cast<const char*>(data), 4);
        }
        std::filesystem::remove(converted);
        examples::remapIcrpVolume(raw, converted, 4, map);
        {
            std::ifstream in(converted, std::ios::binary);
            std::array<unsigned char, 4> values {};
            in.read(reinterpret_cast<char*>(values.data()), 4);
            require(values == std::array<unsigned char, 4> { 0, 0, 1, 37 }, "ICRP RAW remap or byte order");
        }
        bool rejected = false;
        try {
            examples::remapIcrpVolume(raw, converted, 4, map);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "existing output overwritten");
        {
            std::ofstream out(raw, std::ios::binary);
            const unsigned char data[] = { 255 };
            out.write(reinterpret_cast<const char*>(data), 1);
        }
        rejected = false;
        try {
            examples::remapIcrpVolume(raw, root / "invalid.raw", 1, map);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected && !std::filesystem::exists(root / "invalid.raw.partial"), "unknown label accepted or partial output leaked");
        for (const auto& phantom : examples::icrpPhantoms()) {
            const auto config = examples::makeIcrpConfig(phantom, map, root, root / "results");
            const auto file = root / (phantom.id() + ".toml");
            saveProjectionUiConfig(config, file, WorkflowMode::ProjectAndReconstruct);
            const auto loaded = loadConfig(file, false);
            const auto& g = loaded.geometry.parameters;
            require(loaded.projection.materials.size() == 38 && g.volume_x == phantom.dimensions[0] && g.voxel_x_mm == .5, "ICRP config roundtrip");
            require(g.detector_u == 1024 && g.detector_v == 256 && g.reconstruction_volume_x == 256 &&
                g.reconstruction_volume_y == 256 && g.reconstruction_volume_z == (g.volume_z + 3) / 4, "ICRP preview dimensions");
            require(std::abs(g.reconstruction_volume_z * g.reconstruction_voxel_z_mm - g.volume_z * .5) < 1e-9, "ICRP reconstruction Z coverage");
            require(g.detector_u * g.pixel_u_mm * g.sid_mm / g.sdd_mm > std::hypot(g.volume_x * .5, g.volume_y * .5), "ICRP arms truncated transversely");
            require(2 * g.sid_mm * std::sin(std::atan(g.detector_u * g.pixel_u_mm / (2 * g.sdd_mm))) > std::hypot(g.volume_x * .5, g.volume_y * .5), "ICRP fan does not cover bounding circle");
            require(g.start_z_mm < -.25 * g.volume_z && g.start_z_mm + g.pitch_mm_per_turn * g.views / g.views_per_turn > .25 * g.volume_z, "ICRP helical extent");
        }
        // Native projection fixtures exercise material indices above the old 32-entry ceiling.
        for (int count : { 38, 53, 64 }) {
            auto config = examples::makeIcrpConfig(examples::icrpPhantoms()[0], map, root);
            config.projection.materials.clear();
            for (int i = 0; i < count; ++i)
                config.projection.materials.push_back({ static_cast<std::uint8_t>(i), "water", "H2O", "", 1 + i * .001 });
            auto& g = config.geometry.parameters;
            config.geometry.kind = GeometryKind::FlatCbct;
            g.volume_x = g.volume_y = g.volume_z = 2;
            g.voxel_x_mm = g.voxel_y_mm = g.voxel_z_mm = 1;
            g.views = g.views_per_turn = 2;
            g.scan_range_rad = 6.283185307179586;
            g.start_z_mm = g.pitch_mm_per_turn = 0;
            g.detector_u = g.detector_v = 4;
            g.pixel_u_mm = g.pixel_v_mm = .1;
            const auto name = std::to_string(count);
            config.projection.label_volume = root / (name + ".raw");
            {
                std::ofstream out(config.projection.label_volume, std::ios::binary);
                std::array<unsigned char, 8> data;
                data.fill(count - 1);
                data[0] = 0;
                data[4] = count / 2;
                out.write(reinterpret_cast<const char*>(data.data()), 8);
            }
            config.projection.spectrum.source = "file";
            config.projection.spectrum_file = root / (name + ".spec");
            if (count == 64) {
                std::vector<SpectrumPoint> spectrum;
                for (int i = 0; i < 256; ++i)
                    spectrum.push_back({ 20 + i * .5, 1000000. / 256, .5 });
                saveSpectrum(spectrum, config.projection.spectrum_file);
            } else
                saveSpectrum({ { 70, 1000000, 1 } }, config.projection.spectrum_file);
            config.projection.sampling.samples_per_pixel = count == 64 ? 33 : 1;
            config.projection.quantum_noise.enabled = false;
            config.projection.geometry_flux.enabled = false;
            config.projection.output_file = root / (name + "-projection.raw");
            config.projection.energy_output_file = root / (name + "-energy.raw");
            config.projection.air_output_file = root / (name + "-air.raw");
            config.workflow.mode = WorkflowMode::Project;
            config.visualization.enabled = config.visualization.interactive = false;
            saveProjectionUiConfig(config, root / (name + "-projection.toml"), WorkflowMode::Project);
            // The UI save helper deliberately enables interaction for project-only jobs.
            // This GPU regression fixture must stay noninteractive.
            {
                auto document = toml::parse_file((root / (name + "-projection.toml")).string());
                auto& display = *document["visualization"].as_table();
                display.insert_or_assign("enabled", false);
                display.insert_or_assign("interactive", false);
                std::ofstream out(root / (name + "-projection.toml"));
                out << document;
            }
            require(loadConfig(root / (name + "-projection.toml")).projection.materials.size() == size_t(count), "64-material config rejected");
        }
        std::cout << "ICRP ten variants, material densities, label remapping and 38/64 material configurations passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
