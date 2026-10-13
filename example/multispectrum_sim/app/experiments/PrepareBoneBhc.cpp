// Experiment helper: segment a reconstructed attenuation volume, never source labels.
#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include "image/ImageStages.hpp"
#include "recon/postprocess/stages/correction/HuTransforms.hpp"
#include "recon/postprocess/stages/segmentation/HuThresholds.hpp"
#include <fstream>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
using namespace yk::spectral;
namespace pp = yk::spectral::postprocess;
int main(int argc, char** argv)
{
    try {
        const bool bilateral = argc == 5 && std::string(argv[1]) == "--bilateral";
        if (argc != 4 && !bilateral)
            throw std::runtime_error("Usage: multispectrum_bone_bhc_prepare [--bilateral] config.toml reconstruction.raw output-directory");
        const int first = bilateral ? 2 : 1;
        const auto config = loadConfig(argv[first]);
        const auto& g = config.geometry.parameters;
        const std::array<int, 3> dims { g.reconstruction_volume_x, g.reconstruction_volume_y, g.reconstruction_volume_z };
        for (int n : dims)
            if (n <= 0)
                throw std::runtime_error("Experiment requires explicit reconstruction dimensions");
        size_t count = size_t(dims[0]) * dims[1] * dims[2];
        if (std::filesystem::file_size(argv[first + 1]) != count * sizeof(float))
            throw std::runtime_error("Reconstruction size mismatch");
        std::vector<float> values(count);
        std::ifstream input(argv[first + 1], std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
        const std::filesystem::path directory(argv[first + 2]);
        std::filesystem::create_directories(directory);
        auto write = [&](const char* name, const void* data, size_t bytes) {
            std::ofstream out(directory / name, std::ios::binary);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.write(static_cast<const char*>(data), bytes);
            out.close();
        };
        const auto reference = resolveWaterReference(config);
        pp::Volume volume;
        volume.grid.dimensions = dims;
        volume.grid.spacing_mm = { g.reconstruction_voxel_x_mm, g.reconstruction_voxel_y_mm, g.reconstruction_voxel_z_mm };
        const std::array<double, 3> center { g.reconstruction_offset_x_mm, g.reconstruction_offset_y_mm, g.reconstruction_offset_z_mm };
        for (int axis = 0; axis < 3; ++axis)
            volume.grid.origin_mm[axis] = center[axis] - .5 * (dims[axis] - 1) * volume.grid.spacing_mm[axis];
        volume.elements = count;
        volume.host_data = values.data();
        image::VolumeTransfer transfer;
        pp::Context context;
        context.transfer = &transfer;
        if (bilateral) {
            const auto output = directory / "reconstruction.raw";
            if (std::filesystem::weakly_canonical(output) == std::filesystem::weakly_canonical(argv[first + 1]))
                throw std::runtime_error("Filtered output must differ from original reconstruction");
            image::Options options;
            options.operation = image::Operation::Bilateral;
            options.neighborhood = image::Neighborhood::Slice2D;
            options.radius = { 2, 2, 0 };
            options.sigma_mm = { .75, .75, 1. };
            options.range_sigma = 30.;
            image::Workspace workspace(image::Location::Device);
            context.image_workspace = &workspace;
            context.volume_file = output;
            pp::Pipeline filter;
            filter.append(pp::makeMuToHuStage({ reference.mu_water_mm_inv }));
            filter.append(image::makeVolumeStage(options, pp::Domain::HounsfieldUnits));
            filter.append(pp::makeHuToMuStage({ reference.mu_water_mm_inv }));
            filter.appendExporter(pp::makeRawFloatExporter());
            const auto start = std::chrono::steady_clock::now();
            filter.run(volume, context);
            nlohmann::json timing = nlohmann::json::array();
            for (const auto& stage : filter.timings())
                timing.push_back({ { "name", stage.name }, { "milliseconds", stage.milliseconds } });
            nlohmann::json info = { { "operation", "bilateral" }, { "backend", "CUDA" },
                { "neighborhood", "slice2d" }, { "radius_xyz", options.radius },
                { "sigma_xyz_mm", options.sigma_mm }, { "range_sigma_hu", options.range_sigma },
                { "reference_mu_water_mm_inv", reference.mu_water_mm_inv },
                { "reference_energy_keV", reference.energy_keV }, { "output_domain", "attenuation_mm_inv" },
                { "source", std::filesystem::absolute(argv[first + 1]).generic_string() },
                { "output", std::filesystem::absolute(output).generic_string() }, { "dimensions_xyz", dims },
                { "elapsed_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() },
                { "uploads", transfer.uploads }, { "downloads", transfer.downloads }, { "stages", timing } };
            const auto json = info.dump(2);
            write("bilateral.json", json.data(), json.size());
            std::cout << json << '\n';
            return 0;
        }
        pp::Pipeline hu;
        hu.append(pp::makeMuToHuStage({ reference.mu_water_mm_inv }));
        hu.run(volume, context);
        transfer.ensure(volume, pp::Location::Host, context);
        write("initial-hu.raw", volume.host_data, count * sizeof(float));
        pp::Pipeline segment;
        segment.append(pp::makeHuThresholdStage({ { -500., 300. }, { { 0, "air" }, { 1, "soft-water" }, { 2, "bone" } } }));
        segment.run(volume, context);
        transfer.ensure(volume, pp::Location::Host, context);
        std::vector<uint8_t> labels(count);
        std::array<size_t, 3> counts {};
        for (size_t i = 0; i < count; ++i) {
            labels[i] = static_cast<uint8_t>(volume.host_data[i]);
            ++counts.at(labels[i]);
        }
        write("segmented-u8.raw", labels.data(), labels.size());
        nlohmann::json info = { { "reference_energy_keV", reference.energy_keV },
            { "mu_water_mm_inv", reference.mu_water_mm_inv }, { "thresholds_hu", { -500., 300. } },
            { "dimensions_xyz", dims }, { "counts_air_soft_bone", counts },
            { "segmentation_source", std::filesystem::absolute(argv[first + 1]).generic_string() },
            { "original_material_mu_mm_inv", nlohmann::json::object() } };
        XcomAttenuationProvider provider({});
        for (const auto& material : config.projection.materials) {
            std::vector<double> mass;
            std::string error;
            if (!provider.query(material, { reference.energy_keV }, mass, error))
                throw std::runtime_error(error);
            info["original_material_mu_mm_inv"][std::to_string(material.label)] = mass.at(0) * material.density_g_cm3 / 10.;
        }
        const auto json = info.dump(2);
        write("segmentation.json", json.data(), json.size());
        std::cout << json << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
