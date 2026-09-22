#include "PathCache.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

int main()
{
    namespace fs = std::filesystem;
    const fs::path cache = fs::temp_directory_path() /
        "yk_multispectrum_path_cache_test.ykpc";
    std::string error;
    yk::spectral::PathCacheWriter writer;
    if (!writer.open(cache, 3, 2, 1, 2, error, true)) return 1;
    if (fs::file_size(cache) != sizeof(yk::spectral::PathCacheHeader)) return 6;

    // DLL FP 按材料完成整套视图，因此刻意使用 material-major 顺序写入。
    for (std::uint32_t material = 0; material < 2; ++material) {
        for (std::uint32_t view = 0; view < 3; ++view) {
            const float values[2] = {
                static_cast<float>(100 * view + 10 * material + 1),
                static_cast<float>(100 * view + 10 * material + 2)};
            if (!writer.writeMaterialView(view, material, values, 2, error))
                return 2;
        }
        const auto expected_size = sizeof(yk::spectral::PathCacheHeader) +
            static_cast<std::uintmax_t>(material + 1) * 3 * 2 * sizeof(float);
        if (fs::file_size(cache) != expected_size) return 7;
    }
    writer.close();

    yk::spectral::PathCacheReader reader;
    yk::spectral::PathCacheHeader header;
    if (!reader.open(cache, header, error)) return 3;
    for (std::uint32_t view = 0; view < 3; ++view) {
        std::vector<float> values;
        if (!reader.readView(values, error) || values.size() != 4) return 4;
        for (std::uint32_t material = 0; material < 2; ++material) {
            for (std::uint32_t pixel = 0; pixel < 2; ++pixel) {
                const float expected = static_cast<float>(
                    100 * view + 10 * material + pixel + 1);
                if (std::abs(values[material * 2 + pixel] - expected) > 1e-6f) {
                    std::cerr << "path cache layout mismatch\n";
                    return 5;
                }
            }
        }
    }
    reader.close();
    std::error_code ignored;
    fs::remove(cache, ignored);
    return 0;
}
