#include "examples/ExampleStage.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
using namespace yk::spectral;
void require(bool ok, const char* what)
{
    if (!ok)
        throw std::runtime_error(what);
}
int main()
{
    try {
        auto registry = examples::builtinExamples();
        require(registry.list().size() == 8, "example count");
        auto destination = std::filesystem::temp_directory_path() / ("example-no-io-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const int counts[] = { 0, 0, 12, 25, 14, 25, 14, 26 };
        const char* ids[] = { "water-circle", "water-helix", "bead-line", "bead-double-ring", "bead-cylinder-marker", "bead-spiral", "bead-sync-6", "bead-sync-12" };
        for (int k = 0; k < 8; ++k) {
            auto result = registry.create(ids[k])->generate({ destination, 3 });
            const auto& g = result.config.geometry.parameters;
            require(result.config.source_config_file.empty() && result.config.projection.label_volume.empty(), "must be in memory");
            require(!std::filesystem::exists(destination), "generation must not create files");
            require(result.labels.size() == size_t(g.volume_x) * g.volume_y * g.volume_z, "volume size");
            require(result.landmarks.size() == counts[k], "landmark count");
            require(g.views == 3, "view override");
            // Independent spatial oracle over a grid including support boundaries and air.
            for (int z = 0; z < g.volume_z; z += 7)
                for (int y = 0; y < g.volume_y; y += 11)
                    for (int x = 0; x < g.volume_x; x += 13) {
                        double p[] = { (x - (g.volume_x - 1) * .5) * g.voxel_x_mm, (y - (g.volume_y - 1) * .5) * g.voxel_y_mm, (z - (g.volume_z - 1) * .5) * g.voxel_z_mm };
                        int expected = std::hypot(p[0], p[1]) <= result.support_radius_mm && std::abs(p[2]) <= result.support_half_height_mm ? 1 : 0;
                        for (const auto& l : result.landmarks) {
                            double dx = p[0] - l.center_mm[0], dy = p[1] - l.center_mm[1], dz = p[2] - l.center_mm[2];
                            bool inside = l.shape == examples::Shape::Sphere ? std::sqrt(dx * dx + dy * dy + dz * dz) <= l.radius_mm : std::hypot(dx, dy) <= l.radius_mm && std::abs(dz) <= l.half_height_mm;
                            if (inside)
                                expected = l.label;
                        }
                        require(result.labels[(size_t(z) * g.volume_y + y) * g.volume_x + x] == expected, "voxel oracle");
                    }
            for (const auto& l : result.landmarks) {
                int x = int(std::round(l.center_mm[0] / g.voxel_x_mm + (g.volume_x - 1) * .5));
                int y = int(std::round(l.center_mm[1] / g.voxel_y_mm + (g.volume_y - 1) * .5));
                int z = int(std::round(l.center_mm[2] / g.voxel_z_mm + (g.volume_z - 1) * .5));
                require(result.labels[(size_t(z) * g.volume_y + y) * g.volume_x + x] == 100, "landmark center");
            }
            std::cout << ids[k] << " passed\n";
        }
        for (const auto* id : ids) {
            auto stage = registry.create(id);
            auto preset = stage->scanPreset();
            auto sample = stage->generate({ destination });
            require(sample.config.geometry.parameters.views == preset.views, "bound view count");
            require(sample.config.geometry.parameters.detector_u == preset.detector_u && sample.config.geometry.parameters.sid_mm == preset.sid_mm, "bound geometry");
            require(sample.config.reconstruction.analytic.pipeline == preset.reconstruction, "bound reconstruction");
            const auto& g = sample.config.geometry.parameters;
            require(g.reconstruction_volume_x == (g.volume_x + 1) / 2 &&
                g.reconstruction_volume_y == (g.volume_y + 1) / 2 &&
                g.reconstruction_volume_z == (g.volume_z + 1) / 2, "preview reconstruction size");
            require(std::abs(g.reconstruction_volume_x * g.reconstruction_voxel_x_mm - g.volume_x * g.voxel_x_mm) < 1e-9 &&
                std::abs(g.reconstruction_volume_y * g.reconstruction_voxel_y_mm - g.volume_y * g.voxel_y_mm) < 1e-9 &&
                std::abs(g.reconstruction_volume_z * g.reconstruction_voxel_z_mm - g.volume_z * g.voxel_z_mm) < 1e-9, "reconstruction physical extent");
        }
        // Non-default values must affect physical geometry, not only UI metadata.
        examples::ExampleRequest custom { destination, 5 };
        custom.parameters = { { "count", 4 }, { "spacing", 8 }, { "bead_diameter", 2 }, { "support_height", 40 } };
        auto line = registry.create("bead-line")->generate(custom);
        require(line.landmarks.size() == 4 && line.landmarks[0].center_mm[2] == -12 && line.landmarks[3].center_mm[2] == 12, "custom line");
        require(line.landmarks[0].radius_mm == 1 && line.support_half_height_mm == 20, "custom dimensions");
        auto reject = [&](examples::ExampleRequest request) {bool threw=false;try{registry.create("bead-line")->generate(request);}catch(const std::invalid_argument&){threw=true;}require(threw,"invalid phantom parameters"); };
        custom.parameters["count"] = 2.5;
        reject(custom);
        custom.parameters["count"] = 4;
        custom.parameters["nx"] = 8;
        reject(custom);
        custom.parameters.erase("nx");
        custom.parameters["not_a_parameter"] = 1;
        reject(custom);
        custom.parameters.erase("not_a_parameter");
        custom.parameters["spacing"] = 100;
        reject(custom);
        bool rejected = false;
        try {
            registry.create("water-circle")->generate({ destination, -1 });
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid views");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
