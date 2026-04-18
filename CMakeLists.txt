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
# CUDA 架构
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
# OBJECT 库：所有源文件编译一次，DLL 和测试共用
# ----------------------------------------------------------------
add_library(YKCBCT_obj OBJECT)

target_compile_definitions(YKCBCT_obj PRIVATE
    _WINDOWS
    $<$<CONFIG:Release>:NDEBUG YKTASK_EXPORTS>
    $<$<CONFIG:Debug>:_DEBUG>
)

target_compile_options(YKCBCT_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
        /Zc:__cplusplus
        /utf-8
        $<$<CONFIG:Release>:/O2>
    >
)

target_include_directories(YKCBCT_obj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(YKCBCT_obj PRIVATE
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

set_target_properties(YKCBCT_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 子模块往 YKCBCT_objects 里添加源文件
add_subdirectory(src/global)
add_subdirectory(src/interface)
add_subdirectory(src/util)
add_subdirectory(src/Filter)
add_subdirectory(src/FDK)
add_subdirectory(src/BP)
add_subdirectory(src/FP)

# ----------------------------------------------------------------
# 主库：链接 OBJECT 库
# ----------------------------------------------------------------
add_library(YKCBCT SHARED
    $<TARGET_OBJECTS:YKCBCT_obj>
)

target_link_libraries(YKCBCT PRIVATE
    YKCBCT_obj
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

set_target_properties(YKCBCT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ----------------------------------------------------------------
# 测试
# ----------------------------------------------------------------
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    add_subdirectory(Yktest)
endif()