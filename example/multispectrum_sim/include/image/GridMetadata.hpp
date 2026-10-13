#pragma once
#include "ImageGrid.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
namespace yk::spectral::image {
// Versioned float32 RAW sidecar, independent of the original reconstruction TOML.
inline std::filesystem::path gridMetadataPath(const std::filesystem::path& raw) { return raw.string() + ".grid"; }
inline void writeGridMetadata(const std::filesystem::path& raw, const Grid& grid)
{
    std::ofstream f(gridMetadataPath(raw), std::ios::trunc);
    f << "YK_FLOAT32_GRID 1\n"
      << std::setprecision(17);
    for (int v : grid.dimensions)
        f << v << ' ';
    f << '\n';
    for (double v : grid.spacing_mm)
        f << v << ' ';
    f << '\n';
    for (double v : grid.origin_mm)
        f << v << ' ';
    f << '\n';
    for (double v : grid.direction)
        f << v << ' ';
    f << '\n';
    f.close();
    if (!f)
        throw std::runtime_error("cannot write RAW grid metadata");
}
inline bool readGridMetadata(const std::filesystem::path& raw, Grid& grid)
{
    const auto path = gridMetadataPath(raw);
    if (!std::filesystem::exists(path))
        return false;
    std::ifstream f(path);
    std::string magic;
    int version = 0;
    f >> magic >> version;
    if (magic != "YK_FLOAT32_GRID" || version != 1)
        throw std::runtime_error("unsupported RAW grid metadata");
    Grid result;
    for (int& v : result.dimensions)
        f >> v;
    for (double& v : result.spacing_mm)
        f >> v;
    for (double& v : result.origin_mm)
        f >> v;
    for (double& v : result.direction)
        f >> v;
    if (!f)
        throw std::runtime_error("truncated RAW grid metadata");
    std::string extra;
    if (f >> extra)
        throw std::runtime_error("unexpected RAW grid metadata field");
    size_t count = 1;
    for (int a = 0; a < 3; ++a) {
        if (result.dimensions[a] <= 0 || count > std::numeric_limits<size_t>::max() / sizeof(float) / size_t(result.dimensions[a]) || !std::isfinite(result.spacing_mm[a]) || result.spacing_mm[a] <= 0 || !std::isfinite(result.origin_mm[a]))
            throw std::runtime_error("invalid RAW grid metadata");
        count *= size_t(result.dimensions[a]);
    }
    for (double v : result.direction)
        if (!std::isfinite(v))
            throw std::runtime_error("invalid RAW grid direction");
    grid = result;
    return true;
}
}
