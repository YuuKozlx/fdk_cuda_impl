# 轻量单元测试只依赖 GoogleTest 和 Logger 实现，不复制算法 CUDA 源码。
find_package(GTest CONFIG REQUIRED)

# CLI11 是仓库内单头文件，通过 src/util include 路径直接使用。
add_executable(YKCBCT_unit_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logger.cpp
    ${PROJECT_SOURCE_DIR}/src/global/YkLog.cpp
)

target_link_libraries(YKCBCT_unit_test PRIVATE
    GTest::gtest
)

target_compile_options(YKCBCT_unit_test PRIVATE
    # 各编译器使用一致的基础告警等级。
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /utf-8 /Zc:__cplusplus>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

target_include_directories(YKCBCT_unit_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/util
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(YKCBCT_unit_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
include(GoogleTest)
gtest_discover_tests(YKCBCT_unit_test)
