#include "ProjectionSimulator.hpp"

namespace yk::spectral {
bool generatePathCacheWithLibraryFp(const SimulationConfig&, const std::vector<std::uint8_t>&,
                                    const std::filesystem::path&, std::string& diagnostic)
{
    diagnostic = "当前独立构建未链接 YKCBCT.dll，不能使用 DLL FP；请关闭 use_library_fp 或使用顶层工程并开启 MULTISPECTRUM_USE_LIBRARY_FP";
    return false;
}

bool reconstructWithLibraryFdk(const SimulationConfig&, const std::filesystem::path&,
                               const std::filesystem::path&, const std::filesystem::path&,
                               std::string& diagnostic)
{
    diagnostic = "当前独立构建未链接 YKCBCT.dll，不能使用 DLL FDK 重建";
    return false;
}
}
