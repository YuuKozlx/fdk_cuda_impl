#include "recon/backend/LibraryReconstructor.hpp"

namespace yk::spectral {
bool reconstructWithLibrary(const SimulationConfig&, const std::filesystem::path&,
                               const std::filesystem::path&, const std::filesystem::path&,
                               std::string& diagnostic,TaskProgressCallback,PostprocessConfigurer)
{
    diagnostic = "当前构建未链接 YKCBCT.dll，不能执行重建";
    return false;
}
}
