#pragma once

#include "SimConfig.hpp"

#include <filesystem>
#include <string>

namespace yk::spectral {

// 从多能谱仿真生成的 float32 -log 投影中调用 DLL FDK。
// 读取按视图分块进行，避免一次性复制完整投影到显存。
bool reconstructWithLibraryFdk(const SimulationConfig& config,
    const std::filesystem::path& projection_file,
    const std::filesystem::path& volume_file,
    const std::filesystem::path& slice_prefix,
    std::string& diagnostic);

}
