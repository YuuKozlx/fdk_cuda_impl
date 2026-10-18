#include "input/config/SimConfig.hpp"
#include <toml++/toml.hpp>
#include <array>
#include <map>
#include <stdexcept>

namespace yk::spectral {
namespace {
namespace fs = std::filesystem;
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
bool inside(const fs::path& file, const fs::path& directory) {
    auto relative = file.lexically_normal().lexically_relative(directory);
    return !relative.empty() && *relative.begin() != ".." && !relative.is_absolute();
}
void checkPaths(const toml::node& node, const fs::path& base, const fs::path& repository) {
    if (auto table = node.as_table()) {
        for (auto&& [key, value] : *table) {
            const std::string k(key.str());
            if (value.is_string() && (k == "label_volume" || k == "slice_prefix" || k == "snapshot_directory" ||
                    (k.size() > 5 && k.substr(k.size() - 5) == "_file"))) {
                const auto text = value.value<std::string>().value();
                if (text.empty()) continue;
                const auto path = fs::u8path(text);
                check(path.is_relative(), "Catalog contains a machine-specific path: " + text);
                const auto absolute = (base / path).lexically_normal();
                check(inside(absolute, repository), "Catalog path escapes repository: " + text);
                if (k == "output_file" || k == "output_volume_file" || k == "energy_output_file" ||
                    k == "air_output_file" || k == "export_file" || k == "slice_prefix")
                    check(inside(absolute, repository / "out"), "Catalog output is not under out/: " + text);
            } else checkPaths(value, base, repository);
        }
    } else if (auto array = node.as_array()) for (const auto& value : *array) checkPaths(value, base, repository);
}
using Acquisition = std::array<double, 13>;
Acquisition acquisition(const SimulationConfig& c) {
    const auto& g = c.geometry.parameters;
    return {double(c.geometry.kind), double(g.views), double(g.detector_u), double(g.detector_v),
        g.pixel_u_mm, g.pixel_v_mm, g.sid_mm, g.sdd_mm, double(g.views_per_turn),
        g.start_angle_rad, g.pitch_mm_per_turn, g.start_z_mm, double(g.rotation_direction)};
}
}
void testConfigCatalog() {
    const fs::path root = MULTISPECTRUM_CONFIG_DIRECTORY;
    const auto repository = root.parent_path().parent_path().parent_path();
    std::array<int, 3> counts{};
    std::map<fs::path, Acquisition> products;
    std::vector<SimulationConfig> consumers;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.path().extension() != ".toml") continue;
        const auto name = entry.path().lexically_relative(root).generic_string();
        const auto c = loadConfig(entry.path(), false);
        const auto table = toml::parse_file(entry.path().string());
        const int mode = int(c.workflow.mode);
        const char* folders[] = {"projection/", "reconstruction/", "project-and-reconstruct/"};
        check(name.rfind(folders[mode], 0) == 0, "Workflow/folder mismatch: " + name);
        ++counts[mode];
        check(table["schema_version"].value_or(0) == 2, "Old schema in " + name);
        checkPaths(table, entry.path().parent_path(), repository);
        if (c.runsProjection())
            check(c.projection.spectrum.source == "generated", "Projection example unexpectedly requires a legacy spectrum file: " + name);
        check(!c.visualization.enabled, "CLI catalog unexpectedly opens a viewer: " + name);
        if (c.workflow.mode == WorkflowMode::Project)
            check(!table.contains("reconstruction"), "Pure projection exposes reconstruction settings: " + name);
        if (c.runsProjection()) {
            check(!c.projection.label_volume.empty(), "Missing projection phantom: " + name);
            check(c.projection.batching.max_buffer_mib == 256, "Stale batching budget: " + name);
            auto key = c.projection.output_file.lexically_normal();
            auto [it, added] = products.emplace(key, acquisition(c));
            check(added || it->second == acquisition(c), "Two incompatible acquisitions overwrite the same output: " + name);
        } else {
            check(!table["projection"].as_table()->contains("label_volume"), "Pure reconstruction requires a phantom: " + name);
            if (name.rfind("reconstruction/external/", 0) == 0) {
                check(!table["projection"].as_table()->contains("materials"), "External reconstruction requires phantom materials: " + name);
                check(!table["projection"].as_table()->contains("spectrum"), "Uncalibrated external data assumes a source spectrum: " + name);
                check(!c.reconstruction.preprocessing.enabled, "External template applies unverified corrections: " + name);
            }
            consumers.push_back(c);
        }
        if (c.reconstruction.type == "iterative") {
            const auto& i = c.reconstruction.iterative;
            if (i.algorithm == "cgls" || i.algorithm.find("tigre") != std::string::npos)
                check(!table["reconstruction"]["iterative"]["regularization"], "Unrelated internal TV settings: " + name);
        }
    }
    for (int count : counts) check(count > 0, "Missing workflow category");
    for (const auto& c : consumers) {
        auto it = products.find(c.reconstruction.input_projection_file.lexically_normal());
        if (it != products.end()) check(it->second == acquisition(c), "Reconstruction acquisition does not match its producer");
        else check(inside(c.reconstruction.input_projection_file, repository / "out/config-examples/imported"), "Unpaired reconstruction input is not explicitly imported");
    }
}
}
