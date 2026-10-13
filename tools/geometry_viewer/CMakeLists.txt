cmake_minimum_required(VERSION 3.24)
set(YKCBCT_VTK_ROOT "" CACHE PATH
    "Root of a native VTK C++ SDK (contains include, lib and bin)")
if(NOT YKCBCT_VTK_ROOT AND DEFINED ENV{YKCBCT_VTK_ROOT})
    set(YKCBCT_VTK_ROOT "$ENV{YKCBCT_VTK_ROOT}")
endif()
if(YKCBCT_VTK_ROOT)
    if(NOT EXISTS "${YKCBCT_VTK_ROOT}/lib/cmake")
        message(FATAL_ERROR
            "YKCBCT_VTK_ROOT does not contain lib/cmake: ${YKCBCT_VTK_ROOT}")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${YKCBCT_VTK_ROOT}")
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(YKCBCTGeometryViewer LANGUAGES CXX)
    find_package(CUDAToolkit REQUIRED)
endif()
get_filename_component(YKCBCT_VIEWER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(VTK 9.3 REQUIRED COMPONENTS
    CommonCore CommonDataModel FiltersCore FiltersGeneral FiltersSources
    InteractionStyle InteractionWidgets IOImage IOXML
    RenderingAnnotation RenderingCore RenderingOpenGL2 RenderingVolumeOpenGL2)
add_executable(geometry_viewer main.cpp)
target_compile_features(geometry_viewer PRIVATE cxx_std_17)
target_include_directories(geometry_viewer PRIVATE
    "${YKCBCT_VIEWER_ROOT}/include" "${YKCBCT_VIEWER_ROOT}/src"
    "${YKCBCT_VIEWER_ROOT}/src/util" ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(geometry_viewer PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8 /Zc:__cplusplus /W3>)
target_link_libraries(geometry_viewer PRIVATE ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS geometry_viewer MODULES ${VTK_LIBRARIES})
# No link to YKCBCT or CUDA kernels: visualization consumes the actual host builders.
