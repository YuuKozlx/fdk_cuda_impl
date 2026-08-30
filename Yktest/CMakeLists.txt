# Yktest 负责 CUDA、算法集成和数值回归测试。源码按职责分组，便于定位。
set(YKCBCT_MANUAL_TEST_SOURCES
    # 测试入口与轻量数值验证
    ${CMAKE_CURRENT_SOURCE_DIR}/testmain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config/YkConfiguredReconstruction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config/YkConfiguredCylReconstruction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config/YkConfiguredForwardProjection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config/YkConfiguredWfbp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_filter_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_forward_projectors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_geometry_builders.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/YkTextureTestKernels.cu

    # 框架、几何与重建冒烟测试
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pipeline_smoke.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_operator_properties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reconstructors_smoke.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_large_water.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_large_arrow.cpp

)

# 螺旋和圆柱探测器测试直接包含 Heli 头文件并引用对应实现，必须与主库的
# YKCBCT_BUILD_HELICAL 开关保持一致。
if(YKCBCT_BUILD_HELICAL)
    list(APPEND YKCBCT_MANUAL_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fpcyl.cpp
    )
endif()

add_executable(ykcbct_manual_tests
    ${YKCBCT_MANUAL_TEST_SOURCES}
    ${YKCBCT_MANUAL_SUPPORT_SOURCES_ABS}
)

# toml++ 优先由 vcpkg/CMake package 提供。外部安装目录必须遵循标准布局：
# <include-root>/toml++/toml.hpp。
# 这里不记录开发机绝对路径，保证测试工程可以被其他机器和 CI 复现。
set(YKCBCT_TOMLPLUSPLUS_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/thirdparty/tomlplusplus/include" CACHE PATH
    "toml++ 的 include 根目录（其下应有 toml++/toml.hpp）")
find_package(tomlplusplus CONFIG QUIET)
if(EXISTS "${YKCBCT_TOMLPLUSPLUS_INCLUDE_DIR}/toml++/toml.hpp")
    target_include_directories(ykcbct_manual_tests PRIVATE
        "${YKCBCT_TOMLPLUSPLUS_INCLUDE_DIR}")
elseif(TARGET tomlplusplus::tomlplusplus)
    target_link_libraries(ykcbct_manual_tests PRIVATE tomlplusplus::tomlplusplus)
else()
    message(FATAL_ERROR
        "BUILD_TESTS=ON 需要 toml++。请启用 vcpkg manifest，或设置 "
        "YKCBCT_TOMLPLUSPLUS_INCLUDE_DIR 指向包含 toml++/toml.hpp 的 include 根目录。")
endif()
target_link_libraries(ykcbct_manual_tests PRIVATE
    CUDA::cudart CUDA::cufft OpenMP::OpenMP_CXX)
target_include_directories(ykcbct_manual_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/util
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(ykcbct_manual_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3 /utf-8 /Zc:__cplusplus>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>
)
target_compile_definitions(ykcbct_manual_tests PRIVATE
    YKCBCT_TEST_HAS_HELICAL=$<BOOL:${YKCBCT_BUILD_HELICAL}>
    YKCBCT_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(ykcbct_manual_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
