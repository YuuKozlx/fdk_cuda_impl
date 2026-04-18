#include <gtest/gtest.h>
#include "global/YkLog.h"

int main(int argc, char** argv)
{
    // 初始化日志
    YK::Logger::instance().set_level(YK::LogLevel::Debug);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
