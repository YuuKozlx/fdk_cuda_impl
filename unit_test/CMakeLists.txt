# 轻量单元测试只依赖 GoogleTest 和 Logger 实现，不复制算法 CUDA 源码。
find_package(GTest CONFIG REQUIRED)

# CLI11 是仓库内单头文件，通过 src/util include 路径直接使用。
add_executable(YKCBCT_unit_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_small_reconstruction.cpp
    ${PROJECT_SOURCE_DIR}/src/global/YkLog.cpp
)

target_link_libraries(YKCBCT_unit_test PRIVATE
    GTest::gtest
    YKCBCT::YKCBCT
    CUDA::cudart
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

target_compile_definitions(YKCBCT_unit_test PRIVATE
    YKCBCT_TEST_HAS_HELICAL=$<BOOL:${YKCBCT_BUILD_HELICAL}>
)

if(WIN32)
    add_custom_command(TARGET YKCBCT_unit_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:YKCBCT_unit_test>
            $<TARGET_FILE_DIR:YKCBCT_unit_test>
        COMMAND_EXPAND_LISTS
    )
endif()
include(GoogleTest)
gtest_discover_tests(YKCBCT_unit_test)

# Keep the large/manual Yktest program unchanged.  When it is available,
# expose its compact synthetic operator and reconstruction regressions through
# CTest so one test run covers public API tests plus the existing FP/BP kernels.
if(TARGET ykcbct_manual_tests)
    set(YKCBCT_MANUAL_CTEST_CASES
        framework/operator-roundtrip
        framework/operator-matrix
        fp/siddon-uniform-center
        fp/siddon-single-voxel
        dlt/adjoint
        dlt/algebraic-methods
        recon/algebraic-ex
        recon/convergence
        recon/ossart-tv
        recon/tigre-gradient-family
        recon/cgls-unified
    )
    if(YKCBCT_BUILD_HELICAL)
        list(APPEND YKCBCT_MANUAL_CTEST_CASES
            fpcyl/adjoint
            fpcyl/fdk-adjoint
            fpcyl/fdk-backprojectors
        )
    endif()
    foreach(test_case IN LISTS YKCBCT_MANUAL_CTEST_CASES)
        string(REPLACE "/" "." test_name "${test_case}")
        add_test(NAME "YKCBCT.manual.${test_name}"
            COMMAND ykcbct_manual_tests "${test_case}" --log-level error)
        set_tests_properties("YKCBCT.manual.${test_name}" PROPERTIES
            LABELS "cuda;manual-regression")
    endforeach()
endif()

# Representative-size internal operator benchmark.  It is a separate target
# because it directly exercises private FP/BP kernels and is intentionally
# heavier than the fast public-API unit tests above.
add_executable(YKCBCT_operator_benchmark_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_operator_benchmarks.cpp
    ${PROJECT_SOURCE_DIR}/src/global/YkLog.cpp
    ${PROJECT_SOURCE_DIR}/src/FlatFpBp/FP/kernels/YkFPKernels.cu
    ${PROJECT_SOURCE_DIR}/src/FlatFpBp/FP/kernels/YkFPSiddonKernel.cu
    ${PROJECT_SOURCE_DIR}/src/FlatFpBp/BP/kernels/YkBPSiddonKernel.cu
    ${PROJECT_SOURCE_DIR}/src/FlatFpBp/BP/kernels/YkBPJosephKernel.cu
    ${PROJECT_SOURCE_DIR}/src/FlatFpBp/BP/kernels/YkBpFdkKernel.cu
    ${PROJECT_SOURCE_DIR}/src/Reconstruction/Analytic/Circular/Flat/FDK/kernels/YkFDKBpPrecompute.cu
    ${PROJECT_SOURCE_DIR}/src/CylFpBp/kernels/joseph/YkCylFpBpJosephKernels.cu
    ${PROJECT_SOURCE_DIR}/src/CylFpBp/kernels/joseph/YkCylJosephBackprojectV3Kernels.cu
    ${PROJECT_SOURCE_DIR}/src/CylFpBp/kernels/siddon/YkCylSiddonForwardKernels.cu
    ${PROJECT_SOURCE_DIR}/src/CylFpBp/kernels/siddon/YkCylSiddonBackprojectKernels.cu
)
target_link_libraries(YKCBCT_operator_benchmark_test PRIVATE
    GTest::gtest CUDA::cudart CUDA::cufft OpenMP::OpenMP_CXX)
target_include_directories(YKCBCT_operator_benchmark_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/util
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(YKCBCT_operator_benchmark_test PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3 /utf-8 /Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>)
target_compile_definitions(YKCBCT_operator_benchmark_test PRIVATE
    YKCBCT_TEST_HAS_HELICAL=$<BOOL:${YKCBCT_BUILD_HELICAL}>)
set_target_properties(YKCBCT_operator_benchmark_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON)
gtest_discover_tests(YKCBCT_operator_benchmark_test
    TEST_PREFIX "benchmark."
    PROPERTIES LABELS "cuda;operator-benchmark")
