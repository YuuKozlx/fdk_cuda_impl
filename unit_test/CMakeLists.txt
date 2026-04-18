find_package(GTest CONFIG REQUIRED)

add_executable(YKCBCT_unit_test
    test_main.cpp
    test_filter.cpp
    test_fdk.cpp
    test_bp.cpp
    test_fp.cu           # 包含 CUDA kernel 调用，必须 .cu
    $<TARGET_OBJECTS:YKCBCT_obj>
)

target_link_libraries(YKCBCT_unit_test PRIVATE
    YKCBCT_obj
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

target_compile_options(YKCBCT_unit_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8>
)

target_include_directories(YKCBCT_unit_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

set_target_properties(YKCBCT_unit_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 20
)

include(GoogleTest)
gtest_discover_tests(YKCBCT_unit_test)
