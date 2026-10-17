#pragma once
#include "config/SimConfig.hpp"
#include <array>

namespace yk::spectral::examples {
struct IcrpPhantom {
    const char* sex;
    const char* part;
    std::array<int, 3> dimensions;
    std::string id() const;
    std::string rawName() const;
};
const std::array<IcrpPhantom, 10>& icrpPhantoms();
struct IcrpMaterialMap {
    std::vector<MaterialSpec> materials;
    std::array<int, 256> labels;
};
IcrpMaterialMap loadIcrpMaterialMap(const std::filesystem::path&);
void remapIcrpVolume(const std::filesystem::path& input, const std::filesystem::path& output,
    std::size_t voxels, const IcrpMaterialMap&);
SimulationConfig makeIcrpConfig(const IcrpPhantom&, const IcrpMaterialMap&, const std::filesystem::path& output_directory, const std::filesystem::path& results_directory = {});
}
