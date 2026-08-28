#pragma once

#include <filesystem>
#include <string>

namespace YK::TestConfig {

// 运行 wFBP、PWLS 或 OS-SART 螺旋重建配置。未启用螺旋模块时返回明确错误。
int runConfiguredWfbp(const std::filesystem::path& file,
    const std::string& case_name);

} // namespace YK::TestConfig
