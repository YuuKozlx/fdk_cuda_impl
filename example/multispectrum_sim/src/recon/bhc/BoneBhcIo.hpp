#pragma once
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace yk::spectral::bone_bhc::detail {
inline std::ifstream openInput(const std::filesystem::path& p, size_t bytes)
{
    if (std::filesystem::file_size(p) != bytes)
        throw std::runtime_error("iBHC input size mismatch: " + p.string());
    std::ifstream f(p, std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    return f;
}
inline std::ofstream openOutput(const std::filesystem::path& p)
{
    std::ofstream f(p, std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    return f;
}
inline size_t product(int a, int b, int c)
{
    if (a <= 0 || b <= 0 || c <= 0 || size_t(a) > std::numeric_limits<size_t>::max() / size_t(b) / size_t(c) / sizeof(float))
        throw std::invalid_argument("invalid iBHC dimensions");
    return size_t(a) * b * c;
}
}
