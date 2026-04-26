cmake_minimum_required(VERSION 3.26)
project(YKCBCT LANGUAGES CXX CUDA)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# 环境隔离（避免 conda 污染）
# ============================================================
set(CMAKE_IGNORE_PATH "/root/miniconda")

# ----------------------------------------------------------------
# 安装路径
# ----------------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Install path" FORCE)
endif()

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
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ----------------------------------------------------------------
# OBJECT 库
# ----------------------------------------------------------------
add_library(YKCBCT_obj OBJECT)

target_compile_definitions(YKCBCT_obj PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
)

# 跨平台编译选项
if(WIN32)
    target_compile_options(YKCBCT_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /arch:AVX2>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler="/W3"
            $<$<CONFIG:Release>:-O3 -use_fast_math>
        >
    )
elseif(UNIX)
    target_compile_options(YKCBCT_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -pthread>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=-Wall,-Wextra,-Wpedantic
            -Xcompiler=-fPIC
            $<$<CONFIG:Release>:-O3 -use_fast_math>
        >
    )

endif()

target_include_directories(YKCBCT_obj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(YKCBCT_obj PRIVATE
    CUDA::cudart
    CUDA::cufft
    OpenMP::OpenMP_CXX
)

set_target_properties(YKCBCT_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
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
add_subdirectory(src/Heli)

# ----------------------------------------------------------------
# 主库
# ----------------------------------------------------------------
add_library(YKCBCT SHARED
    $<TARGET_OBJECTS:YKCBCT_obj>
)

target_link_libraries(YKCBCT PRIVATE
    YKCBCT_obj
    CUDA::cudart
    CUDA::cufft
    OpenMP::OpenMP_CXX
)

set_target_properties(YKCBCT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Linux 下设置版本号
if(UNIX)
    set_target_properties(YKCBCT PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
    )
endif()

# ----------------------------------------------------------------
# 安装配置
# ----------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS YKCBCT
    EXPORT  YKCBCTTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/YKCBCT
    DESTINATION include
)

install(EXPORT YKCBCTTargets
    FILE        YKCBCTTargets.cmake
    NAMESPACE   YKCBCT::
    DESTINATION lib/cmake/YKCBCT
)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/YKCBCTConfig.cmake.in
    ${CMAKE_BINARY_DIR}/YKCBCTConfig.cmake
    INSTALL_DESTINATION lib/cmake/YKCBCT
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/YKCBCTConfigVersion.cmake
    VERSION       1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/YKCBCTConfig.cmake
    ${CMAKE_BINARY_DIR}/YKCBCTConfigVersion.cmake
    DESTINATION lib/cmake/YKCBCT
)

# ----------------------------------------------------------------
# 测试选项
# ----------------------------------------------------------------
option(BUILD_TESTS "Build integration test executable" ON)
if(BUILD_TESTS)
    add_subdirectory(Yktest)
endif()

option(BUILD_UNIT_TESTS "Build unit tests" ON)
if(BUILD_UNIT_TESTS)
    enable_testing()
    add_subdirectory(unit_test)
endif()

option(BUILD_DLLTEST "Build dynamic library interface test" ON)
if(BUILD_DLLTEST)
    add_subdirectory(dlltest)
endif()

# ----------------------------------------------------------------
# pre-commit hook
# ----------------------------------------------------------------
find_program(PRE_COMMIT pre-commit)
if(PRE_COMMIT AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${PRE_COMMIT} install
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()