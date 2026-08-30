#pragma once

#include <filesystem>
#include <string>

namespace YK::TestConfig {

// 运行 Flat 螺旋 wFBP、SIRT、SART、OS-SART、CGLS 或 PWLS 配置。
// 未启用螺旋模块时返回明确错误。
int runConfiguredWfbp(const std::filesystem::path& file,
    const std::string& case_name);

} // namespace YK::TestConfig
