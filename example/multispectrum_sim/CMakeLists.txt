cmake_minimum_required(VERSION 3.24)
project(multispectrum_sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
if(NOT TARGET yk::spdlog)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/spdlog" "${CMAKE_CURRENT_BINARY_DIR}/spdlog")
endif()

# Options and external/component dependencies.
option(BUILD_VTK_VISUALIZATION "构建多能前投实时 VTK 显示层" OFF)
option(BUILD_IMGUI_GUI "构建 Windows ImGui/VTK 多能前投界面" OFF)
if(BUILD_IMGUI_GUI AND NOT BUILD_VTK_VISUALIZATION)
    message(FATAL_ERROR "BUILD_IMGUI_GUI requires BUILD_VTK_VISUALIZATION=ON")
endif()
if(BUILD_VTK_VISUALIZATION)
    add_subdirectory(modules/visualization)
endif()
option(MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER
    "Build the optional experimental CUDA adaptive scatter solver plugin" OFF)
if(MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER)
    if(NOT TARGET YKCBCT)
        message(FATAL_ERROR "Adaptive scatter plugin requires the CUDA-enabled parent build")
    endif()
endif()

# Project-maintained numerical components; neither requires Python at runtime.
if(NOT TARGET spek_spectrum::core)
    set(SPEK_SPECTRUM_BUILD_TESTS OFF CACHE BOOL "Build upstream spectrum tests")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/modules/spectrum"
        "${CMAKE_CURRENT_BINARY_DIR}/spek_spectrum")
endif()

# 该示例直接编译 XCOM 数值核心，不依赖 YKCBCT DLL 的混杂接口层。
# 因此既可独立配置，也可由顶层 BUILD_MULTISPECTRUM_SIM 接入。
set(SPECTRAL_SIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPECTRAL_XCOM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/modules/attenuation")
add_subdirectory(modules/attenuation)

# MAT import is an application SDK dependency.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(matio 1.5.28 REQUIRED)

# Shared CPU linear algebra SDK; visible to calibration and application targets.
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG REQUIRED)
endif()

# Reusable projection and geometry components.
if(TARGET YKCBCT AND NOT TARGET multispectrum_scalar_fp)
    add_subdirectory(modules/scalar_projection)
endif()
# Geometry calibration is an upper-layer stage; the solver remains independently buildable.
if(NOT TARGET cbct_calibration)
    set(CBCT_CALIBRATION_BUILD_TESTS ${BUILD_TESTING} CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../tools/cbct_calibration_cpp" "${CMAKE_CURRENT_BINARY_DIR}/calibration-core")
endif()

# Internal sources, application, tools, then tests: no backward target wiring.
include(cmake/ProcessingSources.cmake)
include(cmake/Application.cmake)
include(cmake/Tools.cmake)
if(BUILD_TESTING)
    include(cmake/Tests.cmake)
endif()
