cmake_minimum_required(VERSION 3.24)
option(BUILD_VTK_VISUALIZATION_TESTS "Build generated CUDA projection/display fixtures" OFF)
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
    project(YKCBCTVisualization LANGUAGES CXX)
    if(BUILD_VTK_VISUALIZATION_TESTS)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
    endif()
endif()
find_package(VTK 9.3 REQUIRED COMPONENTS
    CommonCore CommonDataModel FiltersCore FiltersGeneral FiltersSources ImagingCore
    InteractionStyle InteractionWidgets IOImage IOXML RenderingAnnotation
    RenderingCore RenderingOpenGL2 RenderingVolumeOpenGL2)
add_library(ykcbct_visualization STATIC LiveViewWindow.cpp ParameterInspector.cpp Mpr.cpp)
get_filename_component(YKCBCT_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    option(BUILD_IMGUI_GUI "Build the Windows ImGui/VTK projection interface" OFF)
endif()
if(BUILD_IMGUI_GUI)
    if(NOT WIN32)
        message(FATAL_ERROR "The initial ImGui/VTK input bridge supports Windows only")
    endif()
    set(IMGUI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/imgui")
    add_library(yk_imgui STATIC
        "${IMGUI_ROOT}/imgui.cpp" "${IMGUI_ROOT}/imgui_draw.cpp"
        "${IMGUI_ROOT}/imgui_tables.cpp" "${IMGUI_ROOT}/imgui_widgets.cpp"
        "${IMGUI_ROOT}/backends/imgui_impl_win32.cpp"
        "${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp")
    target_include_directories(yk_imgui PUBLIC "${IMGUI_ROOT}")
    target_compile_features(yk_imgui PUBLIC cxx_std_17)
    target_compile_definitions(yk_imgui PRIVATE NOMINMAX)
    target_link_libraries(yk_imgui PUBLIC opengl32 imm32 dwmapi)
    target_sources(ykcbct_visualization PRIVATE ImGuiOverlay.cpp)
    target_compile_definitions(ykcbct_visualization PRIVATE YK_VTK_IMGUI=1 NOMINMAX)
    target_link_libraries(ykcbct_visualization PRIVATE yk_imgui comctl32)
endif()

target_compile_features(ykcbct_visualization PUBLIC cxx_std_17)
target_include_directories(ykcbct_visualization
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${YKCBCT_REPOSITORY_ROOT}/src")
target_link_libraries(ykcbct_visualization PRIVATE ${VTK_LIBRARIES})
if(MSVC)
    target_compile_options(ykcbct_visualization PRIVATE /utf-8 /Zc:__cplusplus /W3)
endif()
vtk_module_autoinit(TARGETS ykcbct_visualization MODULES ${VTK_LIBRARIES})

if(BUILD_TESTING)
    enable_testing()
    add_executable(multispectrum_mpr_test "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_mpr.cpp")
    target_link_libraries(multispectrum_mpr_test PRIVATE ykcbct_visualization)
    if(MSVC)
        target_compile_options(multispectrum_mpr_test PRIVATE /utf-8)
    endif()
    vtk_module_autoinit(TARGETS multispectrum_mpr_test MODULES ${VTK_LIBRARIES})
    add_test(NAME multispectrum_mpr COMMAND multispectrum_mpr_test)
    if(BUILD_IMGUI_GUI)
        add_executable(multispectrum_mpr_interaction_test "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_mpr_interaction.cpp")
        target_include_directories(multispectrum_mpr_interaction_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
        target_link_libraries(multispectrum_mpr_interaction_test PRIVATE ykcbct_visualization yk_imgui)
        if(MSVC)
            target_compile_options(multispectrum_mpr_interaction_test PRIVATE /utf-8)
        endif()
        vtk_module_autoinit(TARGETS multispectrum_mpr_interaction_test MODULES ${VTK_LIBRARIES})
        add_test(NAME multispectrum_mpr_interaction COMMAND multispectrum_mpr_interaction_test)
    endif()
endif()

if(BUILD_VTK_VISUALIZATION_TESTS)
    get_filename_component(SPECTRAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    add_executable(spectral_live_test
        "${SPECTRAL_ROOT}/tests/test_live_visualization.cpp"
        "${SPECTRAL_ROOT}/src/primary/CudaProjectionBackend.cu"
        "${SPECTRAL_ROOT}/src/LibraryGeometry.cpp"
        "${SPECTRAL_ROOT}/src/SpectralProjectionPipeline.cpp"
        "${SPECTRAL_ROOT}/src/MultiSpecFpStage.cpp"
        "${SPECTRAL_ROOT}/src/SpectralModel.cpp"
        "${SPECTRAL_ROOT}/src/SimLogger.cpp"
        "${SPECTRAL_ROOT}/src/FasksRuntimeModel.cpp"
        "${SPECTRAL_ROOT}/src/CudaFasksScatter.cu"
        "${SPECTRAL_ROOT}/src/primary/CudaSignalStages.cu"
        "${SPECTRAL_ROOT}/src/cuda/DeviceMemoryPool.cpp"
        "${SPECTRAL_ROOT}/src/primary/ProjectionStages.cpp"
        "${SPECTRAL_ROOT}/src/primary/ProjectionOutput.cpp"
        "${SPECTRAL_ROOT}/src/primary/DetectorSignal.cpp")
    target_include_directories(spectral_live_test PRIVATE "${SPECTRAL_ROOT}/include"
        "${SPECTRAL_ROOT}/../../include" "${SPECTRAL_ROOT}/../../src"
        "${SPECTRAL_ROOT}/../../src/util" "${SPECTRAL_ROOT}/../../thirdparty/nlohmann/include")
    set_property(TARGET spectral_live_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(spectral_live_test PRIVATE ykcbct_visualization CUDA::cudart CUDA::cufft ${VTK_LIBRARIES})
    target_compile_definitions(spectral_live_test PRIVATE YK_SPECTRAL_VTK=1)
    if(MSVC)
        target_compile_options(spectral_live_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    vtk_module_autoinit(TARGETS spectral_live_test MODULES ${VTK_LIBRARIES})
    enable_testing()
    add_test(NAME spectral_live_generated COMMAND spectral_live_test "${CMAKE_CURRENT_BINARY_DIR}/live-checks")
endif()
