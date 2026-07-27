#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "global/YkLog.h"

namespace {

struct CapturedLog {
    YK::LogLevel level;
    std::string tag;
    std::string message;
};

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto& logger = YK::Logger::instance();
        logger.clear_sinks();
        logger.set_level(YK::LogLevel::Trace);
    }
};

TEST_F(LoggerTest, CallbackReceivesFormattedMessage)
{
    std::vector<CapturedLog> logs;
    auto& logger = YK::Logger::instance();
    logger.add_sink([&](YK::LogLevel level, const char* tag, const char* message) {
        logs.push_back({level, tag, message});
    });

    YK_LOGI("value={}, name={}", 42, "filter");

    ASSERT_EQ(logs.size(), 1u);
    EXPECT_EQ(logs[0].level, YK::LogLevel::Info);
    EXPECT_EQ(logs[0].tag, "YK");
    EXPECT_NE(logs[0].message.find("value=42, name=filter"), std::string::npos);
}

TEST_F(LoggerTest, LevelFiltersLowerSeverity)
{
    std::vector<CapturedLog> logs;
    auto& logger = YK::Logger::instance();
    logger.add_sink([&](YK::LogLevel level, const char* tag, const char* message) {
        logs.push_back({level, tag, message});
    });
    logger.set_level(YK::LogLevel::Warn);

    YK_LOGI("hidden");
    YK_LOGW("visible");

    ASSERT_EQ(logs.size(), 1u);
    EXPECT_EQ(logs[0].level, YK::LogLevel::Warn);
    EXPECT_NE(logs[0].message.find("visible"), std::string::npos);
}

TEST_F(LoggerTest, CallbackCanLogWithoutDeadlock)
{
    auto& logger = YK::Logger::instance();
    int callback_count = 0;
    logger.add_sink([&](YK::LogLevel, const char*, const char*) {
        ++callback_count;
        if (callback_count == 1)
            logger.log(YK::LogLevel::Info, "nested", "second message");
    });

    logger.log(YK::LogLevel::Info, "root", "first message");
    EXPECT_EQ(callback_count, 2);
}

} // namespace
