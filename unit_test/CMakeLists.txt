find_package(GTest CONFIG REQUIRED)

add_executable(YKCBCT_unit_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fdk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp.cpp 
    $<TARGET_OBJECTS:YKCBCT_obj>
)

target_link_libraries(YKCBCT_unit_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
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
