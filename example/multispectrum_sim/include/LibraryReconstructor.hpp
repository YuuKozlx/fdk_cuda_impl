#pragma once

#include "config/SimConfig.hpp"

#include <filesystem>
#include <string>

namespace yk::spectral {

// 从 float32 -log 投影调用 DLL：FDK 分包，wFBP/迭代管线全量输入。
// 两条路径的显存均由 DLL Session 管理；wFBP 尚不支持分包。
bool reconstructWithLibraryFdk(const SimulationConfig& config,
    const std::filesystem::path& projection_file,
    const std::filesystem::path& volume_file,
    const std::filesystem::path& slice_prefix,
    std::string& diagnostic);

}
