#pragma once

#include <filesystem>
#include <string>

namespace YK::TestConfig {

// 运行 task="cylindrical-reconstruction"。螺旋 case 经 Heli/Cyl 迭代
// 门面执行；静态和螺旋扫描共用 SCylConeProjGeomVec 数据流。
int runConfiguredCylReconstruction(const std::filesystem::path& file,
    const std::string& case_name);

} // namespace YK::TestConfig
