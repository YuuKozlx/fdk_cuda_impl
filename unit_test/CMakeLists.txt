find_package(GTest CONFIG REQUIRED)

# Keep this target lightweight. CUDA kernels and algorithm-level numerical
# regressions belong to Yktest, whose target owns the required device link.
add_executable(YKCBCT_unit_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logger.cpp
    ${CMAKE_SOURCE_DIR}/src/global/YkLog.cpp
)

target_link_libraries(YKCBCT_unit_test PRIVATE
    GTest::gtest
)

target_compile_options(YKCBCT_unit_test PRIVATE
#    $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8>
        # MSVC 编译器选项
        $<$<CXX_COMPILER_ID:MSVC>:/W3 /utf-8 /Zc:__cplusplus>

        # GCC 编译器选项
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>

        # Clang 编译器选项
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

target_include_directories(YKCBCT_unit_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/util
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

set_target_properties(YKCBCT_unit_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
)
include(GoogleTest)
gtest_discover_tests(YKCBCT_unit_test)
