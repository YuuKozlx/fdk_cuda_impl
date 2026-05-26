cmake_minimum_required(VERSION 3.26)
project(YKCBCT LANGUAGES CXX CUDA)

# ----------------------------------------------------------------
# 安装路径
# 只有当用户没有指定安装路径时才设置默认值，避免覆盖用户的选择。
# 例如用户通过 -DCMAKE_INSTALL_PREFIX=... 指定了安装路径，就不修改它。
# ----------------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Install path" FORCE)
endif()

# ----------------------------------------------------------------
# 标准
# ----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
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
# find_package(<包名> CONFIG REQUIRED)
#   CONFIG    : 使用包自带的 *Config.cmake，而不是 CMake 内置的 Find*.cmake
#   REQUIRED  : 找不到直接报错终止
# 找到后会创建 imported target，例如：
#   spdlog::spdlog   CUDAToolkit 提供 CUDA::cudart / CUDA::cufft
# ----------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)   # 新增

# ----------------------------------------------------------------
# OBJECT 库：所有源文件编译一次，DLL 和测试共用同一批 .obj
#
# OBJECT 库不产生 .lib/.dll，只编译出 .obj 文件集合。
# 后续 DLL 和测试都消费这批 .obj，源文件不重复编译。
# ----------------------------------------------------------------
add_library(YKCBCT_obj OBJECT)

target_compile_definitions(YKCBCT_obj PRIVATE
    _WINDOWS
    YKCBCT_EXPORTS
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
)

# 跨平台编译选项
# Windows 编译选项
# 跨平台编译选项
if(WIN32)
    target_compile_options(YKCBCT_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /arch:AVX2>
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            -Xcompiler="/W3"
            $<$<CONFIG:Release>:-O3 -use_fast_math>
        >
    )

elseif(UNIX)
    target_compile_options(YKCBCT_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -pthread>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler="-Wall" -Xcompiler="-Wextra" -Xcompiler="-Wpedantic"
            -Xcompiler="-fPIC"
             --extended-lambda
            $<$<CONFIG:Release>:-O3 -use_fast_math>
        >
    )
    
endif()

# PUBLIC  : 自身和消费者都能看到（include 路径会传递给链接 YKCBCT_obj 的目标）
# PRIVATE : 只有自身编译时用（内部头文件路径不对外暴露）
target_include_directories(YKCBCT_obj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(YKCBCT_obj PRIVATE
    CUDA::cudart
    CUDA::cufft
    OpenMP::OpenMP_CXX   # 新增
)

set_target_properties(YKCBCT_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ----------------------------------------------------------------
# 子模块：各自往 YKCBCT_obj 里添加源文件
# ----------------------------------------------------------------
add_subdirectory(src/global)
add_subdirectory(src/interface)
add_subdirectory(src/util)
add_subdirectory(src/Filter)
add_subdirectory(src/FDK)
add_subdirectory(src/BP)
add_subdirectory(src/FP)
add_subdirectory(src/Heli)
add_subdirectory(src/Iter)
# ----------------------------------------------------------------
# 主库 DLL：只做链接，不重复编译源文件
#
# $<TARGET_OBJECTS:YKCBCT_obj>
#   生成器表达式，展开为 YKCBCT_obj 编译出的所有 .obj 文件路径。
#   CUDA 场景下显式引用比只用 target_link_libraries 更稳定。
# ----------------------------------------------------------------
add_library(YKCBCT SHARED
    $<TARGET_OBJECTS:YKCBCT_obj>
)

target_link_libraries(YKCBCT PRIVATE
    YKCBCT_obj
    CUDA::cudart
    CUDA::cufft
    OpenMP::OpenMP_CXX   # 新增
)

set_target_properties(YKCBCT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ----------------------------------------------------------------
# 包安装配置
#
# 目标：让外部项目能用以下方式使用本库：
#   find_package(YKCBCT CONFIG REQUIRED)
#   target_link_libraries(my_app PRIVATE YKCBCT::YKCBCT)
# ----------------------------------------------------------------
include(CMakePackageConfigHelpers)
# CMake 内置模块，提供下面用到的两个函数

# 安装库产物（.dll/.lib）并记录到导出集合 YKCBCTTargets
# EXPORT YKCBCTTargets : 把 YKCBCT target 信息收集到这个导出集合里
# RUNTIME → .dll 装到 bin/
# ARCHIVE → .lib（导入库）装到 lib/
install(TARGETS YKCBCT
    EXPORT  YKCBCTTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装对外头文件（include/YKCBCT/ 整个目录）
# 外部用户 #include "YKCBCT/interface/IYkTask.hpp" 即可找到
install(DIRECTORY include/YKCBCT
    DESTINATION include
)

# 把导出集合写成 YKCBCTTargets.cmake 文件
# NAMESPACE YKCBCT:: : target 名字加前缀，外部用 YKCBCT::YKCBCT
install(EXPORT YKCBCTTargets
    FILE        YKCBCTTargets.cmake
    NAMESPACE   YKCBCT::
    DESTINATION lib/cmake/YKCBCT
)

# 把 YKCBCTConfig.cmake.in 模板渲染成 YKCBCTConfig.cmake
# @PACKAGE_INIT@ 占位符会被替换成路径处理代码，
# 使配置文件在任意安装位置都能正确定位 YKCBCTTargets.cmake
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/YKCBCTConfig.cmake.in
    ${CMAKE_BINARY_DIR}/YKCBCTConfig.cmake
    INSTALL_DESTINATION lib/cmake/YKCBCT
)

# 生成版本文件
# SameMajorVersion : 主版本号相同即兼容（1.2.0 兼容 1.0.0 要求，但不兼容 2.0.0）
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/YKCBCTConfigVersion.cmake
    VERSION       1.0.0
    COMPATIBILITY SameMajorVersion
)

# 安装配置文件和版本文件
install(FILES
    ${CMAKE_BINARY_DIR}/YKCBCTConfig.cmake
    ${CMAKE_BINARY_DIR}/YKCBCTConfigVersion.cmake
    DESTINATION lib/cmake/YKCBCT
)

# ----------------------------------------------------------------
# 集成测试（Yktest）
# ----------------------------------------------------------------
option(BUILD_TESTS "Build integration test executable" ON)
if(BUILD_TESTS)
    add_subdirectory(Yktest)
endif()

#option(NEW_TEST "Build integration test executable" ON)
#if(NEW_TEST)
#    add_subdirectory(SimultaneousTest)
#endif()

# ----------------------------------------------------------------
# 单元测试（GoogleTest）
# ----------------------------------------------------------------
option(BUILD_UNIT_TESTS "Build unit tests" ON)
if(BUILD_UNIT_TESTS)
    enable_testing()
    add_subdirectory(unit_test)
endif()



# ----------------------------------------------------------------
# dlltest：测试 DLL 导出接口的正确性和兼容性
# ----------------------------------------------------------------
option(BUILD_DLLTEST "Build DLL interface test" ON)
if(BUILD_DLLTEST)
    add_subdirectory(dlltest)
endif()


# ----------------------------------------------------------------
# extra: pre-commit hook 自动安装,禁止大文件提交等
# ----------------------------------------------------------------
find_program(PRE_COMMIT pre-commit)
if(PRE_COMMIT AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${PRE_COMMIT} install
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()