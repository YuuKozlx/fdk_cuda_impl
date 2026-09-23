#include "LibraryReconstructor.hpp"

namespace yk::spectral {
bool reconstructWithLibraryFdk(const SimulationConfig&, const std::filesystem::path&,
                               const std::filesystem::path&, const std::filesystem::path&,
                               std::string& diagnostic)
{
    diagnostic = "当前构建未链接 YKCBCT.dll，不能执行重建";
    return false;
}
}
