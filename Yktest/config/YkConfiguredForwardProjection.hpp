#pragma once

#include <filesystem>
#include <string>

namespace YK::TestConfig {

// 运行 task="forward-projection" 配置。case_name 为空时执行文件内全部 case。
int runConfiguredForwardProjection(const std::filesystem::path& file,
    const std::string& case_name);

} // namespace YK::TestConfig
