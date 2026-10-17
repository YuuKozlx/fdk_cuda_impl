#include "examples/IcrpExamples.hpp"
#include "XcomAttenuationProvider.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <toml++/toml.hpp>

namespace yk::spectral::examples {
std::string IcrpPhantom::id() const { return std::string("icrp145-") + sex + "-" + part; }
std::string IcrpPhantom::rawName() const { return std::string(sex) + "_data_" + part + "_" + std::to_string(dimensions[0]) + "x" + std::to_string(dimensions[1]) + "x" + std::to_string(dimensions[2]) + ".raw"; }
const std::array<IcrpPhantom, 10>& icrpPhantoms()
{
    static const std::array<IcrpPhantom, 10> variants { { { "female", "abdomen", { 1024, 512, 512 } }, { "female", "abdomen_no_arm", { 610, 512, 512 } },
        { "female", "chest", { 1024, 512, 612 } }, { "female", "chest_no_arm", { 570, 512, 612 } },
        { "female", "head", { 512, 512, 400 } },
        { "male", "abdomen", { 1024, 512, 512 } }, { "male", "abdomen_no_arm", { 624, 512, 512 } },
        { "male", "chest", { 1024, 512, 612 } }, { "male", "chest_no_arm", { 606, 512, 612 } },
        { "male", "head", { 512, 512, 400 } } } };
    return variants;
}
IcrpMaterialMap loadIcrpMaterialMap(const std::filesystem::path& path)
{
    const auto table = toml::parse_file(path.string());
    IcrpMaterialMap result;
    result.labels.fill(-1);
    const auto* entries = table["materials"].as_array();
    if (!entries)
        throw std::runtime_error("ICRP material TOML has no materials");
    for (const auto& entry : *entries) {
        const auto* m = entry.as_table();
        if (!m)
            throw std::runtime_error("Invalid ICRP material entry");
        const auto* ids = (*m)["voxel_ids"].as_array();
        if (!ids || ids->empty())
            throw std::runtime_error("Missing ICRP voxel_ids");
        // The MC-GPU detector's CsI entry is not a phantom tissue.
        if (ids->size() == 1 && (*ids)[0].value<int>() == 254)
            continue;
        auto name = (*m)["preset"].value_or(std::string {});
        if (name.empty()) {
            const auto filename = std::filesystem::path((*m)["file"].value_or(std::string {})).filename().string();
            const std::string suffix = "_1-200keV.mcgpu.gz";
            if (filename.size() <= suffix.size() || filename.substr(filename.size() - suffix.size()) != suffix)
                throw std::runtime_error("Unsupported ICRP material filename: " + filename);
            name = filename.substr(0, filename.size() - suffix.size());
        }
        const auto* preset = findBuiltinMaterial(name);
        if (!preset)
            throw std::runtime_error("ICRP material has no exact built-in preset: " + name);
        const double density = (*m)["density_g_cm3"].value_or(0.);
        if (!std::isfinite(density) || density <= 0)
            throw std::runtime_error("Invalid ICRP sex-specific material density");
        if (result.materials.size() >= 64)
            throw std::runtime_error("ICRP material map exceeds 64 tissue groups");
        MaterialSpec material;
        material.label = static_cast<std::uint8_t>(result.materials.size());
        material.name = name;
        material.preset = preset->id;
        material.density_g_cm3 = density;
        for (const auto& id : *ids) {
            const auto label = id.value<int>();
            if (!label || *label < 0 || *label > 253 || result.labels[*label] >= 0)
                throw std::runtime_error("Invalid/duplicate ICRP voxel ID");
            result.labels[*label] = material.label;
        }
        result.materials.push_back(std::move(material));
    }
    if (result.materials.empty())
        throw std::runtime_error("Empty ICRP phantom material map");
    return result;
}
void remapIcrpVolume(const std::filesystem::path& input, const std::filesystem::path& output, std::size_t voxels, const IcrpMaterialMap& map)
{
    if (std::filesystem::file_size(input) != voxels)
        throw std::runtime_error("ICRP RAW size does not match the expected uint8 grid: " + input.string());
    if (std::filesystem::exists(output))
        throw std::runtime_error("ICRP output already exists; use a new output directory: " + output.string());
    std::filesystem::create_directories(output.parent_path());
    const auto temporary = std::filesystem::path(output.string() + ".partial");
    if (std::filesystem::exists(temporary))
        throw std::runtime_error("ICRP temporary output already exists");
    try {
        std::ifstream in(input, std::ios::binary);
        std::ofstream out(temporary, std::ios::binary);
        if (!in || !out)
            throw std::runtime_error("Cannot open ICRP volume input/output");
        std::vector<unsigned char> bytes(1024 * 1024);
        for (std::size_t offset = 0; offset < voxels;) {
            const auto count = std::min(bytes.size(), voxels - offset);
            if (!in.read(reinterpret_cast<char*>(bytes.data()), count))
                throw std::runtime_error("Cannot read ICRP RAW");
            for (std::size_t i = 0; i < count; ++i) {
                const auto label = bytes[i];
                if (map.labels[label] < 0)
                    throw std::runtime_error("Unmapped ICRP voxel ID: " + std::to_string(label));
                bytes[i] = static_cast<unsigned char>(map.labels[label]);
            }
            if (!out.write(reinterpret_cast<const char*>(bytes.data()), count))
                throw std::runtime_error("Cannot write ICRP RAW");
            offset += count;
        }
        out.close();
        if (!out)
            throw std::runtime_error("Incomplete ICRP output");
        std::filesystem::rename(temporary, output);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(temporary, error);
        throw;
    }
}
SimulationConfig makeIcrpConfig(const IcrpPhantom& phantom, const IcrpMaterialMap& materials, const std::filesystem::path& directory, const std::filesystem::path& results_directory)
{
    SimulationConfig c;
    const auto root = std::filesystem::absolute(directory);
    const auto id = phantom.id();
    c.workflow.mode = WorkflowMode::ProjectAndReconstruct;
    c.projection.materials = materials.materials;
    c.projection.label_volume = root / "inputs" / (id + ".raw");
    const auto results = (results_directory.empty() ? root / "results" : std::filesystem::absolute(results_directory)) / id;
    c.projection.output_file = results / "projection.raw";
    c.projection.energy_output_file = results / "energy.raw";
    c.projection.air_output_file = results / "air.raw";
    c.projection.mAs_per_view = 5;
    c.projection.quantum_noise.enabled = true;
    c.projection.geometry_flux.enabled = true;
    c.projection.sampling.samples_per_pixel = 4;
    c.projection.scatter.enabled = false;
    c.projection.spectrum.source = "generated";
    c.projection.spectrum.model = "spekpy-v1";
    c.projection.spectrum.kvp = 120;
    c.projection.spectrum.filters = { { "Al", 2 }, { "Ti", .6 } };
    c.projection.spectrum.export_file = results / "spectrum.spec";
    auto& g = c.geometry.parameters;
    g.sid_mm = 709;
    g.sdd_mm = 1143.699951;
    g.volume_x = phantom.dimensions[0];
    g.volume_y = phantom.dimensions[1];
    g.volume_z = phantom.dimensions[2];
    g.voxel_x_mm = g.voxel_y_mm = g.voxel_z_mm = .5;
    const double fov = std::ceil(std::hypot(g.volume_x * .5, g.volume_y * .5) + 20);
    // A fan detector must subtend the bounding circle; simple SID magnification
    // underestimates the width for the arm-inclusive volumes.
    if (fov * .5 >= g.sid_mm)
        throw std::runtime_error("ICRP volume exceeds the source orbit");
    g.detector_u = 1024;
    g.detector_v = 256;
    const double detector_width = 2 * g.sdd_mm * std::tan(std::asin(fov * .5 / g.sid_mm));
    g.pixel_u_mm = detector_width / g.detector_u;
    g.pixel_v_mm = detector_width / 3 / g.detector_v;
    const double row_coverage = g.detector_v * g.pixel_v_mm * g.sid_mm / g.sdd_mm;
    g.pitch_mm_per_turn = row_coverage * .5;
    g.views_per_turn = 360;
    const int turns = int(std::ceil((g.volume_z * .5 + row_coverage) / g.pitch_mm_per_turn));
    g.views = turns * g.views_per_turn;
    g.start_z_mm = -.5 * turns * g.pitch_mm_per_turn;
    g.scan_range_rad = turns * 6.283185307179586;
    c.geometry.kind = GeometryKind::FlatHelical;
    g.reconstruction_volume_x = g.reconstruction_volume_y = 256;
    g.reconstruction_volume_z = (g.volume_z + 3) / 4;
    g.reconstruction_voxel_x_mm = g.reconstruction_voxel_y_mm = fov / 256;
    g.reconstruction_voxel_z_mm = g.volume_z * .5 / g.reconstruction_volume_z;
    c.reconstruction.output_volume_file = results / "reconstruction.raw";
    c.reconstruction.input_projection_file = c.projection.output_file;
    c.reconstruction.type = "analytic";
    c.reconstruction.analytic.pipeline = "wfbp";
    c.reconstruction.analytic.filter = "hann";
    c.reconstruction.preprocessing.enabled = true;
    c.reconstruction.preprocessing.beam_hardening.enabled = true;
    c.reconstruction.preprocessing.beam_hardening.fit_from_water_plate = true;
    c.reconstruction.preprocessing.beam_hardening.reference_energy_keV = 70;
    c.reconstruction.preprocessing.beam_hardening.water_plate_max_mm = 600;
    c.visualization.enabled = true;
    c.visualization.interactive = true;
    return c;
}
}
