#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace yk::spectral {
using OrganLabelMap = std::unordered_map<std::uint64_t, std::uint8_t>;
// Numeric dense 3-D MAT v5 array. MATLAB's first dimension is x (fastest).
// Without an organ map, values must be integer material labels in [0,255].
std::vector<std::uint8_t> readMatLabels(const std::filesystem::path&,
    const std::array<int, 3>& dimensions, const OrganLabelMap& organs = {});
void writeMatLabels(const std::filesystem::path&, const std::array<int, 3>&,
    const std::vector<std::uint8_t>&);
std::vector<std::uint8_t> readClientLabels(const std::filesystem::path&,
    const std::array<int, 3>& dimensions);
}
