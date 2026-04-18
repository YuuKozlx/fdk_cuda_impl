cmake_minimum_required(VERSION 3.20)
project(YKCBCT LANGUAGES CXX CUDA)

# ----------------------------------------------------------------
# 标准
# ----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------
# CUDA 架构（按实际 GPU 调整）
# ----------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# ----------------------------------------------------------------
# 默认构建类型
# ----------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------
# 依赖
# ----------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# ----------------------------------------------------------------
# 主库目标
# ----------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_library(YKCBCT SHARED)
    target_compile_definitions(YKCBCT PRIVATE NDEBUG YKTASK_EXPORTS)
else()
    add_library(YKCBCT STATIC)
    target_compile_definitions(YKCBCT PRIVATE _DEBUG)
endif()

target_compile_definitions(YKCBCT PRIVATE _WINDOWS)

target_compile_options(YKCBCT PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
        /Zc:__cplusplus
        /utf-8
        $<$<CONFIG:Release>:/O2>
    >
)

# 对外头文件：include/YKCBCT/
# 内部头文件：src/ 下各模块
target_include_directories(YKCBCT
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/global
        ${CMAKE_SOURCE_DIR}/src/interface
        ${CMAKE_SOURCE_DIR}/src/util
        ${CMAKE_SOURCE_DIR}/src/Filter
        ${CMAKE_SOURCE_DIR}/src/FDK
        ${CMAKE_SOURCE_DIR}/src/FDK/cuh
        ${CMAKE_SOURCE_DIR}/src/BP
        ${CMAKE_SOURCE_DIR}/src/FP
        ${CMAKE_SOURCE_DIR}/src/FP/kernels
        ${CMAKE_SOURCE_DIR}/src/CVP
)

target_link_libraries(YKCBCT PRIVATE
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

set_target_properties(YKCBCT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ----------------------------------------------------------------
# 子模块
# ----------------------------------------------------------------
add_subdirectory(src/global)
add_subdirectory(src/interface)
add_subdirectory(src/util)
add_subdirectory(src/Filter)
add_subdirectory(src/FDK)
add_subdirectory(src/BP)
add_subdirectory(src/FP)

# ----------------------------------------------------------------
# 测试
# ----------------------------------------------------------------
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    add_subdirectory(test)
endif()
