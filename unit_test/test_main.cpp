#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "CLI11/CLI11.hpp"
#include "global/YkLog.h"

namespace {

YK::LogLevel parseLogLevel(const std::string& value)
{
    if (value == "trace") return YK::LogLevel::Trace;
    if (value == "debug") return YK::LogLevel::Debug;
    if (value == "info") return YK::LogLevel::Info;
    if (value == "warn") return YK::LogLevel::Warn;
    if (value == "error") return YK::LogLevel::Error;
    if (value == "critical") return YK::LogLevel::Critical;
    return YK::LogLevel::Off;
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app{"YKCBCT 轻量组件单元测试"};
    std::string log_level = "debug";
    app.add_option("--yk-log-level", log_level, "YK 日志等级")
        ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}));
    // GoogleTest owns --gtest_* options. CLI11 consumes project options and
    // returns unknown arguments for InitGoogleTest.
    app.allow_extras();
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    YK::Logger::instance().set_level(parseLogLevel(log_level));

    std::vector<std::string> gtest_arguments;
    gtest_arguments.emplace_back(argv[0]);
    const auto passthrough = app.remaining_for_passthrough();
    gtest_arguments.insert(gtest_arguments.end(), passthrough.begin(), passthrough.end());
    std::vector<char*> gtest_argv;
    gtest_argv.reserve(gtest_arguments.size());
    for (auto& argument : gtest_arguments)
        gtest_argv.push_back(argument.data());
    int gtest_argc = static_cast<int>(gtest_argv.size());

    ::testing::InitGoogleTest(&gtest_argc, gtest_argv.data());
    return RUN_ALL_TESTS();
}
