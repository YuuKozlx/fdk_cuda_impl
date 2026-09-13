cmake_minimum_required(VERSION 3.24)

project(YKCBCT VERSION 1.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# 顶层构建说明
# ============================================================================
#
# 本项目生成四类目标：
#
#   YKCBCT                主动态库，拥有全部正式 C++/CUDA 实现源码；
#   ykcbct_manual_tests   CUDA、算法集成和数值回归测试（Yktest）；
#   YKCBCT_unit_test      不依赖算法 CUDA device-link 的轻量单元测试；
#   dlltest               只通过安装风格公共头文件和 DLL ABI 调用主库。
#
# 源码组织规则：
#
#   - 顶层先创建 YKCBCT 目标；
#   - src/*/CMakeLists.txt 仅通过 target_sources() 向主目标追加源码；
#   - Yktest 需要直接访问内部 kernel，因此复制主目标的实现源码并自行完成
#     CUDA device-link，而不是链接后再重复编译同一批实现；
#   - unit_test 只放轻量组件测试，算法和 CUDA 数值测试统一放入 Yktest；
#   - 安装导出提供 YKCBCT::YKCBCT，供外部工程 find_package() 使用。
#
# 常用配置示例：
#
#   cmake -S . -B out/build/release -G Ninja \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_CUDA_ARCHITECTURES=86
#
#   cmake -S . -B out/build/tests -G Ninja \
#       -DBUILD_TESTS=ON -DBUILD_UNIT_TESTS=ON \
#       -DYKCBCT_BUILD_HELICAL=ON
#
# 所有功能选项均在首次使用前于本文件统一声明。子目录不得重复声明全局选项。

# ============================================================================
# 构建选项
# ============================================================================

option(YKCBCT_BUILD_HELICAL "构建实验性螺旋 CT 算法" OFF)
option(YKCBCT_BUILD_CVP "构建实验性 CVP 正投影算法" OFF)
option(BUILD_TESTS "构建 Yktest 集成与数值回归测试" OFF)
option(BUILD_UNIT_TESTS "构建轻量 GoogleTest 单元测试" OFF)
option(BUILD_DLLTEST "构建 DLL 公共接口测试" OFF)
option(BUILD_MULTISPECTRUM_SIM "构建独立多能谱 CBCT 仿真示例" OFF)
option(BUILD_MULTISPECTRUM_PLOT_TOOLS "构建基于 Matplot++ 的重建结果分析工具" OFF)

# ============================================================================
# 工具链和全局默认值
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 仅在用户和 toolchain 都未指定时提供默认架构，避免覆盖命令行配置。
# 75/80/86 分别覆盖 Turing、Ampere A100 和 Ampere 消费级 GPU。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING
        "构建 YKCBCT 使用的 CUDA 架构")
endif()

# Ninja 等单配置生成器需要默认构建类型；Visual Studio 等多配置生成器在
# 构建阶段选择 Debug/Release，不应强行设置 CMAKE_BUILD_TYPE。
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install"
        CACHE PATH "安装路径" FORCE)
endif()

# ============================================================================
# 外部依赖
# ============================================================================

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ============================================================================
# 主动态库目标
# ============================================================================

# 正式 CUDA 源码全部归属于同一个动态库目标，以保证 MSVC/CUDA 工具链生成的
# device-link 对象不会因拆分 OBJECT library 而丢失。
add_library(YKCBCT SHARED
    src/interface/YkLibraryAnchor.cpp
)
add_library(YKCBCT::YKCBCT ALIAS YKCBCT)

target_compile_definitions(YKCBCT PRIVATE
    YKCBCT_EXPORTS
    YKCBCT_HAS_HELICAL=$<BOOL:${YKCBCT_BUILD_HELICAL}>
    $<$<PLATFORM_ID:Windows>:_WINDOWS>
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
)

target_include_directories(YKCBCT
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/util
)

target_link_libraries(YKCBCT
    PUBLIC
        CUDA::cudart
    PRIVATE
        CUDA::cufft
        OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(YKCBCT PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /arch:AVX2>
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            -Xcompiler=/W3
            -Xcompiler=/utf-8
            $<$<CONFIG:Release>:-O3 -use_fast_math>
        >
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(YKCBCT PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -pthread>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            -Xcompiler=-Wall
            -Xcompiler=-Wextra
            -Xcompiler=-Wpedantic
            -Xcompiler=-fPIC
            $<$<CONFIG:Release>:-O3 -use_fast_math>
        >
    )
endif()

set_target_properties(YKCBCT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# 功能模块源码
# ============================================================================

add_subdirectory(src/global)
add_subdirectory(src/interface)
add_subdirectory(src/util)
add_subdirectory(src/common)
add_subdirectory(src/Filter)
add_subdirectory(src/FDK)
add_subdirectory(src/BP)
add_subdirectory(src/FP)
add_subdirectory(src/CylFpBp)
add_subdirectory(src/Iter)

if(YKCBCT_BUILD_HELICAL)
    add_subdirectory(src/Heli)
endif()

# ============================================================================
# 测试和开发工具
# ============================================================================

# Yktest 直接测试私有 kernel，不能只依赖公共 DLL ABI。这里从 YKCBCT 目标
# 读取实现源码，转成绝对路径并排除 DLL 入口和 Session 导出实现，再交给
# Yktest 自行完成 CUDA 编译及 device-link。
function(ykcbct_collect_manual_test_sources output_variable)
    get_target_property(manual_sources YKCBCT SOURCES)
    set(absolute_sources)
    foreach(source IN LISTS manual_sources)
        if(IS_ABSOLUTE "${source}")
            set(absolute_source "${source}")
        else()
            set(absolute_source "${PROJECT_SOURCE_DIR}/${source}")
        endif()
        cmake_path(NORMAL_PATH absolute_source)
        if(absolute_source STREQUAL
                "${PROJECT_SOURCE_DIR}/src/interface/YkSession.cpp" OR
           absolute_source STREQUAL
                "${PROJECT_SOURCE_DIR}/src/interface/YkReconstructionApi.cpp" OR
           absolute_source STREQUAL
                "${PROJECT_SOURCE_DIR}/src/interface/YkLibraryAnchor.cpp")
            continue()
        endif()
        list(APPEND absolute_sources "${absolute_source}")
    endforeach()
    set(${output_variable} "${absolute_sources}" PARENT_SCOPE)
endfunction()

if(BUILD_TESTS)
    ykcbct_collect_manual_test_sources(YKCBCT_MANUAL_SUPPORT_SOURCES_ABS)
    add_subdirectory(Yktest)
endif()

if(BUILD_UNIT_TESTS)
    enable_testing()
    add_subdirectory(unit_test)
endif()

if(BUILD_DLLTEST)
    add_subdirectory(dlltest)
endif()

if(BUILD_MULTISPECTRUM_SIM)
    add_subdirectory(example/multispectrum_sim)
endif()

if(BUILD_MULTISPECTRUM_PLOT_TOOLS)
    find_package(Matplot++ CONFIG REQUIRED)
    find_program(GNUPLOT_EXECUTABLE gnuplot)
    if(NOT GNUPLOT_EXECUTABLE)
        message(FATAL_ERROR
            "BUILD_MULTISPECTRUM_PLOT_TOOLS requires the gnuplot executable "
            "at build/runtime; install gnuplot and add it to PATH")
    endif()
    message(STATUS "Matplot++ plotting backend: ${GNUPLOT_EXECUTABLE}")
    add_subdirectory(example/multispectrum_sim/plot)
endif()

# ============================================================================
# 安装与 CMake 包导出
# ============================================================================

include(CMakePackageConfigHelpers)

install(TARGETS YKCBCT
    EXPORT YKCBCTTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/YKCBCT
    DESTINATION include
)

install(EXPORT YKCBCTTargets
    FILE YKCBCTTargets.cmake
    NAMESPACE YKCBCT::
    DESTINATION lib/cmake/YKCBCT
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/YKCBCTConfig.cmake.in
    ${PROJECT_BINARY_DIR}/YKCBCTConfig.cmake
    INSTALL_DESTINATION lib/cmake/YKCBCT
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/YKCBCTConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${PROJECT_BINARY_DIR}/YKCBCTConfig.cmake
    ${PROJECT_BINARY_DIR}/YKCBCTConfigVersion.cmake
    DESTINATION lib/cmake/YKCBCT
)

# ============================================================================
# 开发辅助
# ============================================================================

find_program(PRE_COMMIT pre-commit)
if(PRE_COMMIT AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${PRE_COMMIT} install
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif()
