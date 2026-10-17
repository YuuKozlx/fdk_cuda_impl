cmake_minimum_required(VERSION 3.24)
project(multispectrum_sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(multispectrum_bone_bhc STATIC src/recon/bhc/BoneBhcStage.cpp)
target_include_directories(multispectrum_bone_bhc PUBLIC include)
if(MSVC)
    target_compile_options(multispectrum_bone_bhc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()
if(BUILD_TESTING)
    add_executable(multispectrum_bone_bhc_test tests/test_bone_bhc.cpp)
    target_link_libraries(multispectrum_bone_bhc_test PRIVATE multispectrum_bone_bhc)
    add_test(NAME multispectrum_bone_bhc COMMAND multispectrum_bone_bhc_test "${CMAKE_CURRENT_BINARY_DIR}/bone-bhc-tests")
endif()
option(BUILD_VTK_VISUALIZATION "构建多能前投实时 VTK 显示层" OFF)
option(BUILD_IMGUI_GUI "构建 Windows ImGui/VTK 多能前投界面" OFF)
if(BUILD_IMGUI_GUI AND NOT BUILD_VTK_VISUALIZATION)
    message(FATAL_ERROR "BUILD_IMGUI_GUI requires BUILD_VTK_VISUALIZATION=ON")
endif()
if(BUILD_VTK_VISUALIZATION)
    add_subdirectory(visualization)
endif()
option(MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER
    "Build the optional experimental CUDA adaptive scatter solver plugin" OFF)
if(MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER)
    if(NOT TARGET YKCBCT)
        message(FATAL_ERROR "Adaptive scatter plugin requires the CUDA-enabled parent build")
    endif()
    add_library(multispectrum_adaptive_scatter STATIC
        plugins/adaptive_scatter/AdaptiveScatterSolver.cu)
    target_include_directories(multispectrum_adaptive_scatter PUBLIC
        include plugins/adaptive_scatter)
    target_link_libraries(multispectrum_adaptive_scatter PUBLIC CUDA::cudart)
    target_compile_definitions(multispectrum_adaptive_scatter INTERFACE YK_SPECTRAL_ADAPTIVE_SCATTER=1)
    if(MSVC)
        target_compile_options(multispectrum_adaptive_scatter PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

# Vendored native generator: no Python or runtime table lookup.
if(NOT TARGET spek_spectrum::core)
    set(SPEK_SPECTRUM_BUILD_TESTS OFF CACHE BOOL "Build upstream spectrum tests")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/spek_spectrum"
        "${CMAKE_CURRENT_BINARY_DIR}/spek_spectrum")
endif()

# 该示例直接编译 XCOM 数值核心，不依赖 YKCBCT DLL 的混杂接口层。
# 因此既可独立配置，也可由顶层 BUILD_MULTISPECTRUM_SIM 接入。
set(SPECTRAL_SIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
# 无论独立配置还是作为顶层子目录加入，示例目录到仓库 thirdparty 均为 ../../。
set(SPECTRAL_XCOM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nist_xcom")
add_library(nist_xcom_core STATIC
    "${SPECTRAL_XCOM_ROOT}/xcom.cpp"
    "${SPECTRAL_XCOM_ROOT}/xcom_c_api.cpp")
target_include_directories(nist_xcom_core PUBLIC "${SPECTRAL_XCOM_ROOT}")
target_compile_definitions(nist_xcom_core PUBLIC XCOM_C_API_STATIC)
add_executable(multispectrum_sim app/main.cpp src/SimLogger.cpp src/config/SimConfig.cpp src/SpectralModel.cpp src/FasksRuntimeModel.cpp src/XcomAttenuationProvider.cpp src/ProjectionSimulator.cpp)
target_include_directories(multispectrum_sim PRIVATE
    include src "${SPECTRAL_XCOM_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
target_link_libraries(multispectrum_sim PRIVATE nist_xcom_core)
if(TARGET multispectrum_adaptive_scatter)
    target_link_libraries(multispectrum_sim PRIVATE multispectrum_adaptive_scatter)
endif()

# 在线多能谱前投由示例自身的 CUDA kernel 完成；YKCBCT DLL 只提供重建后端。
if(TARGET YKCBCT)
    target_sources(multispectrum_sim PRIVATE
        src/recon/bhc/BoneBhcJob.cpp
        src/LibraryGeometry.cpp src/LibraryReconstructor.cpp src/ReconstructionPreprocessor.cpp
        src/recon/postprocess/PostprocessPipeline.cpp src/recon/postprocess/OutputStages.cpp
        src/recon/preprocess/ProjectionCache.cpp
        src/CudaPrimaryProjector.cu src/SpectralProjectionPipeline.cpp src/MultiSpecFpStage.cpp
        src/primary/CudaProjectionBackend.cu src/primary/DetectorSignal.cpp
        src/primary/ProjectionStages.cpp src/primary/ProjectionOutput.cpp
        src/CudaFasksScatter.cu)
    target_include_directories(multispectrum_sim PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_sim PRIVATE YKCBCT)
    target_link_libraries(multispectrum_sim PRIVATE multispectrum_bone_bhc)
    target_link_libraries(multispectrum_sim PRIVATE CUDA::cufft)
    target_compile_definitions(multispectrum_sim PRIVATE
        YK_MULTISPECTRUM_CUDA_PRIMARY=1)
    if(TARGET ykcbct_visualization)
        target_sources(multispectrum_sim PRIVATE src/InteractiveProjection.cpp)
        target_link_libraries(multispectrum_sim PRIVATE ykcbct_visualization)
        target_compile_definitions(multispectrum_sim PRIVATE YK_SPECTRAL_VTK=1)
        if(WIN32)
            add_custom_command(TARGET multispectrum_sim POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:multispectrum_sim>
                    $<TARGET_FILE_DIR:multispectrum_sim>
                COMMAND_EXPAND_LISTS)
        endif()
        if(TARGET yk_imgui)
            target_sources(multispectrum_sim PRIVATE src/ImGuiProjection.cpp src/ProjectionUiConfig.cpp src/DisplayGeometry.cpp src/ReconstructionDisplay.cpp)
            target_link_libraries(multispectrum_sim PRIVATE yk_imgui comdlg32)
            target_compile_definitions(multispectrum_sim PRIVATE YK_SPECTRAL_IMGUI=1 NOMINMAX)
        endif()
    endif()
    # Refresh the runtime even when DLL exports/import-library contents do not
    # change and Ninja correctly leaves the executable itself untouched.
    add_custom_target(multispectrum_runtime_sync
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:multispectrum_sim>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_sim>
        DEPENDS YKCBCT)
    add_dependencies(multispectrum_sim multispectrum_runtime_sync)
    add_custom_command(TARGET multispectrum_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_sim>)
else()
    target_sources(multispectrum_sim PRIVATE src/LibraryReconstructorStub.cpp src/CudaPrimaryProjectorStub.cpp)
endif()
if(MSVC)
    target_compile_options(nist_xcom_core PRIVATE /utf-8)
    target_compile_options(multispectrum_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(multispectrum_projection_cache_test tests/test_projection_cache.cpp src/recon/preprocess/ProjectionCache.cpp)
    target_include_directories(multispectrum_projection_cache_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    if(MSVC)
        target_compile_options(multispectrum_projection_cache_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_projection_cache COMMAND multispectrum_projection_cache_test "${CMAKE_CURRENT_BINARY_DIR}/projection-cache-tests")
    add_executable(multispectrum_signal_unit_test tests/primary/test_detector_signal.cpp
        src/primary/DetectorSignal.cpp src/primary/ProjectionOutput.cpp)
    target_include_directories(multispectrum_signal_unit_test PRIVATE include src)
    if(MSVC)
        target_compile_options(multispectrum_signal_unit_test PRIVATE /utf-8)
    endif()
    foreach(module IN ITEMS crosstalk afterglow das thickness output air_input)
        add_test(NAME multispectrum_unit_${module} COMMAND multispectrum_signal_unit_test
            ${module} "${CMAKE_CURRENT_BINARY_DIR}/unit-test-output/${module}")
        set_tests_properties(multispectrum_unit_${module} PROPERTIES LABELS "unit;cpu;multispectrum")
    endforeach()
    if(TARGET YKCBCT)
        add_executable(multispectrum_pose_unit_test tests/primary/test_nonstandard_pose.cpp
            src/LibraryGeometry.cpp)
        target_include_directories(multispectrum_pose_unit_test PRIVATE include src
            "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
        target_link_libraries(multispectrum_pose_unit_test PRIVATE YKCBCT)
        if(MSVC)
            target_compile_options(multispectrum_pose_unit_test PRIVATE /utf-8)
        endif()
        add_test(NAME multispectrum_unit_nonstandard_pose COMMAND multispectrum_pose_unit_test)
        set_tests_properties(multispectrum_unit_nonstandard_pose PROPERTIES LABELS "unit;cpu;multispectrum")
        add_executable(multispectrum_cuda_unit_test tests/primary/test_cuda_modules.cu)
        target_include_directories(multispectrum_cuda_unit_test PRIVATE include src)
        target_link_libraries(multispectrum_cuda_unit_test PRIVATE CUDA::cudart)
        if(MSVC)
            target_compile_options(multispectrum_cuda_unit_test PRIVATE -Xcompiler=/utf-8)
        endif()
        foreach(module IN ITEMS geometry paths noise response transmission focal multisample warp warp_coefficients warp_profile warp_policy)
            add_test(NAME multispectrum_unit_${module} COMMAND multispectrum_cuda_unit_test ${module})
            set_tests_properties(multispectrum_unit_${module} PROPERTIES LABELS "unit;gpu;multispectrum")
        endforeach()
        add_executable(multispectrum_pipeline_test tests/test_projection_pipeline.cpp tests/test_device_signal_stages.cpp
            src/primary/CudaProjectionBackend.cu src/LibraryGeometry.cpp
            src/SpectralProjectionPipeline.cpp src/MultiSpecFpStage.cpp src/SpectralModel.cpp src/SimLogger.cpp
            src/primary/ProjectionStages.cpp src/primary/ProjectionOutput.cpp
            src/primary/DetectorSignal.cpp src/FasksRuntimeModel.cpp src/CudaFasksScatter.cu)
        target_include_directories(multispectrum_pipeline_test PRIVATE include
            "${CMAKE_CURRENT_SOURCE_DIR}/../../src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
        target_link_libraries(multispectrum_pipeline_test PRIVATE YKCBCT CUDA::cudart CUDA::cufft)
        if(MSVC)
            target_compile_options(multispectrum_pipeline_test PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        endif()
        add_test(NAME multispectrum_projection_pipeline COMMAND multispectrum_pipeline_test
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline-test-output")
        add_executable(multispectrum_scatter_test tests/test_scatter.cpp
            src/FasksRuntimeModel.cpp src/CudaFasksScatter.cu)
        target_include_directories(multispectrum_scatter_test PRIVATE include
            "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
        target_link_libraries(multispectrum_scatter_test PRIVATE CUDA::cudart)
        target_link_libraries(multispectrum_scatter_test PRIVATE CUDA::cufft)
        add_test(NAME multispectrum_scatter COMMAND multispectrum_scatter_test)
        add_executable(multispectrum_preprocessing_test tests/test_preprocessing.cpp tests/test_preprocess_state.cpp
            src/ReconstructionPreprocessor.cpp src/FasksRuntimeModel.cpp
            src/config/SimConfig.cpp src/XcomAttenuationProvider.cpp
            src/CudaFasksScatter.cu)
        target_include_directories(multispectrum_preprocessing_test PRIVATE include
            "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
            "${SPECTRAL_XCOM_ROOT}")
        target_link_libraries(multispectrum_preprocessing_test PRIVATE nist_xcom_core CUDA::cudart CUDA::cufft)
        if(TARGET multispectrum_adaptive_scatter)
            target_link_libraries(multispectrum_preprocessing_test PRIVATE multispectrum_adaptive_scatter)
            add_executable(multispectrum_adaptive_scatter_test tests/test_adaptive_scatter.cpp)
            if(MSVC)
                target_compile_options(multispectrum_adaptive_scatter_test PRIVATE /utf-8)
            endif()
            target_link_libraries(multispectrum_adaptive_scatter_test PRIVATE multispectrum_adaptive_scatter)
            add_test(NAME multispectrum_adaptive_scatter COMMAND multispectrum_adaptive_scatter_test)
        endif()
        if(MSVC)
            target_compile_options(multispectrum_preprocessing_test PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        endif()
        add_test(NAME multispectrum_preprocessing COMMAND multispectrum_preprocessing_test)
    endif()
    if(TARGET YKCBCT AND YKCBCT_BUILD_HELICAL)
        add_executable(multispectrum_wfbp_buffers_test tests/test_wfbp_buffers.cpp src/LibraryGeometry.cpp)
        target_include_directories(multispectrum_wfbp_buffers_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
        target_link_libraries(multispectrum_wfbp_buffers_test PRIVATE YKCBCT)
        if(MSVC)
            target_compile_options(multispectrum_wfbp_buffers_test PRIVATE /utf-8)
        endif()
        add_custom_command(TARGET multispectrum_wfbp_buffers_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_wfbp_buffers_test>)
        add_test(NAME multispectrum_wfbp_buffers COMMAND multispectrum_wfbp_buffers_test)
    endif()
    add_executable(multispectrum_config_test tests/test_config_validation.cpp
        src/config/SimConfig.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_config_test PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_config_test PRIVATE nist_xcom_core)
    target_compile_definitions(multispectrum_config_test PRIVATE XCOM_C_API_STATIC)
    if(MSVC)
        target_compile_options(multispectrum_config_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_config_validation COMMAND multispectrum_config_test)
    add_executable(multispectrum_xcom_test
        tests/test_xcom_adapter.cpp
        src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_xcom_test PRIVATE
        include "${SPECTRAL_XCOM_ROOT}")
    target_link_libraries(multispectrum_xcom_test PRIVATE nist_xcom_core)
    target_compile_definitions(multispectrum_xcom_test PRIVATE XCOM_C_API_STATIC)
    if(MSVC)
        target_compile_options(multispectrum_xcom_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_xcom_water COMMAND multispectrum_xcom_test)

endif()

foreach(spectrum_consumer IN ITEMS multispectrum_sim multispectrum_config_test multispectrum_preprocessing_test)
    if(TARGET ${spectrum_consumer})
        target_sources(${spectrum_consumer} PRIVATE src/config/SpectrumSource.cpp)
        target_link_libraries(${spectrum_consumer} PRIVATE spek_spectrum::core)
    endif()
endforeach()
if(BUILD_TESTING)
    add_executable(multispectrum_spectrum_test tests/test_spectrum_source.cpp
        src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_spectrum_test PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_spectrum_test PRIVATE nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_spectrum_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_spectrum_source COMMAND multispectrum_spectrum_test)
endif()

if(BUILD_TESTING AND TARGET yk_imgui)
    add_executable(multispectrum_ui_config_test tests/test_ui_config.cpp
        tests/test_scan_panel.cpp
        src/ProjectionUiConfig.cpp src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_ui_config_test PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_ui_config_test PRIVATE nist_xcom_core spek_spectrum::core yk_imgui)
    target_compile_definitions(multispectrum_ui_config_test PRIVATE NOMINMAX)
    if(MSVC)
        target_compile_options(multispectrum_ui_config_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_ui_config COMMAND multispectrum_ui_config_test)
endif()

if(BUILD_TESTING)
    add_executable(multispectrum_reconstruction_display_test tests/test_reconstruction_display.cpp src/ReconstructionDisplay.cpp)
    target_include_directories(multispectrum_reconstruction_display_test PRIVATE include)
    if(MSVC)
        target_compile_options(multispectrum_reconstruction_display_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_reconstruction_display COMMAND multispectrum_reconstruction_display_test)
endif()

foreach(water_consumer IN ITEMS multispectrum_sim multispectrum_preprocessing_test)
    if(TARGET ${water_consumer})
        target_sources(${water_consumer} PRIVATE src/WaterReference.cpp)
    endif()
endforeach()
if(BUILD_TESTING)
    add_executable(multispectrum_water_reference_test tests/test_water_reference.cpp
        src/WaterReference.cpp src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_water_reference_test PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_water_reference_test PRIVATE nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_water_reference_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_water_reference COMMAND multispectrum_water_reference_test)
endif()

if(BUILD_TESTING AND TARGET YKCBCT)
    add_executable(multispectrum_reconstruction_progress_test tests/test_reconstruction_progress.cpp src/LibraryGeometry.cpp)
    target_include_directories(multispectrum_reconstruction_progress_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_reconstruction_progress_test PRIVATE YKCBCT CUDA::cudart)
    if(MSVC)
        target_compile_options(multispectrum_reconstruction_progress_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_reconstruction_progress COMMAND multispectrum_reconstruction_progress_test)
endif()

foreach(preprocess_consumer IN ITEMS multispectrum_sim multispectrum_preprocessing_test)
    if(TARGET ${preprocess_consumer} AND TARGET YKCBCT)
        target_sources(${preprocess_consumer} PRIVATE
            src/recon/preprocess/PreprocessPipeline.cpp src/recon/preprocess/Stages.cpp
            src/recon/preprocess/BeamHardeningFit.cpp src/recon/preprocess/CudaPixelOps.cu)
    endif()
endforeach()

foreach(cuda_consumer IN ITEMS multispectrum_sim multispectrum_preprocessing_test multispectrum_adaptive_scatter)
    if(TARGET ${cuda_consumer})
        set_property(TARGET ${cuda_consumer} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    endif()
endforeach()

if(TARGET YKCBCT)
    add_library(multispectrum_device_pool STATIC src/cuda/DeviceMemoryPool.cpp)
    target_include_directories(multispectrum_device_pool PUBLIC include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_device_pool PUBLIC CUDA::cudart)
    if(MSVC)
        target_compile_options(multispectrum_device_pool PRIVATE /utf-8)
    endif()
    foreach(pool_consumer IN ITEMS multispectrum_sim multispectrum_preprocessing_test
            multispectrum_scatter_test multispectrum_pipeline_test multispectrum_adaptive_scatter)
        if(TARGET ${pool_consumer})
            target_link_libraries(${pool_consumer} PUBLIC multispectrum_device_pool)
            set_property(TARGET ${pool_consumer} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
        endif()
    endforeach()
endif()

foreach(device_stage_consumer IN ITEMS multispectrum_sim multispectrum_pipeline_test)
    if(TARGET ${device_stage_consumer} AND TARGET YKCBCT)
        target_sources(${device_stage_consumer} PRIVATE src/primary/CudaSignalStages.cu)
    endif()
endforeach()
set_source_files_properties(src/primary/CudaSignalStages.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")

if(BUILD_TESTING AND TARGET YKCBCT)
    add_executable(multispectrum_ossart_weight_cache_test tests/test_ossart_weight_cache.cpp src/LibraryGeometry.cpp)
    target_include_directories(multispectrum_ossart_weight_cache_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_ossart_weight_cache_test PRIVATE YKCBCT CUDA::cudart)
    if(MSVC)
        target_compile_options(multispectrum_ossart_weight_cache_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_ossart_weight_cache COMMAND multispectrum_ossart_weight_cache_test)
    add_executable(multispectrum_tigre_weight_cache_test tests/test_tigre_weight_cache.cpp src/LibraryGeometry.cpp)
    target_include_directories(multispectrum_tigre_weight_cache_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_tigre_weight_cache_test PRIVATE YKCBCT CUDA::cudart)
    if(MSVC)
        target_compile_options(multispectrum_tigre_weight_cache_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_tigre_weight_cache COMMAND multispectrum_tigre_weight_cache_test)
    add_executable(multispectrum_batched_reconstruction_test tests/test_batched_reconstruction.cpp src/LibraryGeometry.cpp)
    target_include_directories(multispectrum_batched_reconstruction_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_batched_reconstruction_test PRIVATE YKCBCT CUDA::cudart)
    if(MSVC)
        target_compile_options(multispectrum_batched_reconstruction_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_batched_reconstruction COMMAND multispectrum_batched_reconstruction_test)
endif()

if(BUILD_TESTING AND TARGET YKCBCT)
    add_executable(multispectrum_tv_plugin_test tests/test_tv_plugin.cpp
        ../../src/Reconstruction/Iterative/Regularization/HuberTvKernels.cu)
    target_include_directories(multispectrum_tv_plugin_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_tv_plugin_test PRIVATE YKCBCT CUDA::cudart)
    set_property(TARGET multispectrum_tv_plugin_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_tv_plugin_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    endif()
    add_test(NAME multispectrum_tv_plugin COMMAND multispectrum_tv_plugin_test)
endif()

if(BUILD_TESTING AND TARGET YKCBCT)
    add_executable(multispectrum_statistical_test tests/test_statistical_reconstruction.cpp src/LibraryGeometry.cpp)
    target_include_directories(multispectrum_statistical_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_statistical_test PRIVATE YKCBCT CUDA::cudart)
    if(MSVC)
        target_compile_options(multispectrum_statistical_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_statistical COMMAND multispectrum_statistical_test)
endif()

if(BUILD_TESTING)
    add_executable(multispectrum_relative_geometry_test tests/test_relative_geometry.cpp)
    target_compile_features(multispectrum_relative_geometry_test PRIVATE cxx_std_17)
    add_test(NAME multispectrum_relative_geometry COMMAND multispectrum_relative_geometry_test)
endif()

if(BUILD_TESTING)
    add_executable(multispectrum_postprocess_test tests/test_postprocess.cpp src/recon/postprocess/PostprocessPipeline.cpp src/recon/postprocess/OutputStages.cpp src/SimLogger.cpp)
    target_include_directories(multispectrum_postprocess_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
    if(MSVC)
        target_compile_options(multispectrum_postprocess_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_postprocess COMMAND multispectrum_postprocess_test "${CMAKE_CURRENT_BINARY_DIR}/postprocess-tests")
endif()

if(BUILD_TESTING)
    add_executable(multispectrum_stage_runtime_test tests/test_stage_runtime.cpp)
    target_include_directories(multispectrum_stage_runtime_test PRIVATE include)
    add_test(NAME multispectrum_stage_runtime COMMAND multispectrum_stage_runtime_test)
endif()

if(BUILD_TESTING)
    add_executable(multispectrum_plugin_metadata_test tests/test_plugin_metadata.cpp)
    target_include_directories(multispectrum_plugin_metadata_test PRIVATE include)
    add_test(NAME multispectrum_plugin_metadata COMMAND multispectrum_plugin_metadata_test)
endif()

if(TARGET YKCBCT)
    add_library(multispectrum_image_processing STATIC
        src/image/ImageProcessing.cpp src/image/ImageKernels.cu src/image/ImageStages.cpp)
    target_include_directories(multispectrum_image_processing PUBLIC include)
    target_link_libraries(multispectrum_image_processing PUBLIC multispectrum_device_pool CUDA::cudart)
    set_property(TARGET multispectrum_image_processing PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_image_processing PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    target_link_libraries(multispectrum_sim PRIVATE multispectrum_image_processing)
    if(TARGET multispectrum_preprocessing_test)
        target_link_libraries(multispectrum_preprocessing_test PRIVATE multispectrum_image_processing)
    endif()
    if(BUILD_TESTING)
        add_executable(multispectrum_image_test tests/test_image_processing.cpp
            src/recon/postprocess/PostprocessPipeline.cpp src/SimLogger.cpp)
        target_include_directories(multispectrum_image_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
        target_link_libraries(multispectrum_image_test PRIVATE multispectrum_image_processing)
        if(MSVC)
            target_compile_options(multispectrum_image_test PRIVATE /utf-8)
        endif()
        add_test(NAME multispectrum_image_processing COMMAND multispectrum_image_test)
    endif()
endif()

add_library(multispectrum_intensity_stages STATIC
    src/recon/postprocess/stages/IntensityOps.cpp
    src/recon/postprocess/stages/correction/HuTransforms.cpp
    src/recon/postprocess/stages/calibration/HuCalibration.cpp
    src/recon/postprocess/stages/segmentation/HuThresholds.cpp)
target_include_directories(multispectrum_intensity_stages PUBLIC include)
if(TARGET YKCBCT)
    # Explicitly built experiment helper; no additional application/runtime dependency.
    add_executable(multispectrum_bone_bhc_singlepass EXCLUDE_FROM_ALL
        app/experiments/IdealBoneCorrection.cu src/config/SimConfig.cpp src/config/SpectrumSource.cpp
        src/WaterReference.cpp src/XcomAttenuationProvider.cpp src/SimLogger.cpp
        src/recon/preprocess/BeamHardeningFit.cpp)
    target_include_directories(multispectrum_bone_bhc_singlepass PRIVATE include src
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
    target_link_libraries(multispectrum_bone_bhc_singlepass PRIVATE
        nist_xcom_core spek_spectrum::core multispectrum_device_pool CUDA::cudart)
    set_property(TARGET multispectrum_bone_bhc_singlepass PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(BUILD_TESTING)
        add_dependencies(multispectrum_bone_bhc_test multispectrum_bone_bhc_singlepass)
        add_test(NAME multispectrum_bone_bhc_physics COMMAND multispectrum_bone_bhc_singlepass --self-test)
    endif()
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc_singlepass PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_executable(multispectrum_bone_bhc_prepare EXCLUDE_FROM_ALL
        app/experiments/PrepareBoneBhc.cpp src/config/SimConfig.cpp src/config/SpectrumSource.cpp
        src/WaterReference.cpp src/XcomAttenuationProvider.cpp
        src/recon/postprocess/PostprocessPipeline.cpp src/recon/postprocess/OutputStages.cpp src/SimLogger.cpp)
    target_include_directories(multispectrum_bone_bhc_prepare PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
    target_link_libraries(multispectrum_bone_bhc_prepare PRIVATE multispectrum_intensity_stages
        multispectrum_image_processing nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc_prepare PRIVATE /utf-8)
    endif()
endif()
if(MSVC)
    target_compile_options(multispectrum_intensity_stages PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
if(TARGET YKCBCT)
    target_sources(multispectrum_intensity_stages PRIVATE src/recon/postprocess/stages/IntensityKernels.cu)
    target_compile_definitions(multispectrum_intensity_stages PRIVATE YK_INTENSITY_CUDA=1)
    target_link_libraries(multispectrum_intensity_stages PRIVATE multispectrum_device_pool CUDA::cudart)
    set_property(TARGET multispectrum_intensity_stages PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
foreach(intensity_consumer IN ITEMS multispectrum_sim multispectrum_postprocess_test multispectrum_image_test)
    if(TARGET ${intensity_consumer})
        target_link_libraries(${intensity_consumer} PRIVATE multispectrum_intensity_stages)
    endif()
endforeach()
if(BUILD_TESTING)
    add_executable(multispectrum_intensity_test tests/test_intensity_stages.cpp
        src/recon/postprocess/PostprocessPipeline.cpp src/SimLogger.cpp)
    target_include_directories(multispectrum_intensity_test PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
    target_link_libraries(multispectrum_intensity_test PRIVATE multispectrum_intensity_stages)
    if(TARGET YKCBCT)
        target_link_libraries(multispectrum_intensity_test PRIVATE multispectrum_image_processing)
        target_compile_definitions(multispectrum_intensity_test PRIVATE YK_INTENSITY_CUDA=1)
    endif()
    if(MSVC)
        target_compile_options(multispectrum_intensity_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_intensity_stages COMMAND multispectrum_intensity_test)
endif()

# Plug-in calibration and correction: independent of reconstruction backend and UI.
add_library(multispectrum_two_material_calibration STATIC src/recon/preprocess/calibration/TwoMaterialCalibration.cpp src/recon/preprocess/calibration/EmpiricalTwoMaterialFit.cpp)
target_include_directories(multispectrum_two_material_calibration PUBLIC include PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
add_executable(multispectrum_fit_material_phantom EXCLUDE_FROM_ALL app/experiments/FitMaterialPhantom.cpp)
target_link_libraries(multispectrum_fit_material_phantom PRIVATE multispectrum_two_material_calibration)
target_include_directories(multispectrum_fit_material_phantom PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
if(TARGET YKCBCT)
    add_library(multispectrum_two_material_bhc STATIC src/recon/preprocess/correction/TwoMaterialBhcStage.cu)
    target_include_directories(multispectrum_two_material_bhc PUBLIC include)
    target_link_libraries(multispectrum_two_material_bhc PUBLIC multispectrum_two_material_calibration multispectrum_device_pool CUDA::cudart)
    target_sources(multispectrum_bone_bhc PRIVATE src/recon/bhc/BoneBhcAlgorithms.cu)
    target_include_directories(multispectrum_bone_bhc PRIVATE src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_bone_bhc PUBLIC multispectrum_two_material_bhc multispectrum_device_pool multispectrum_scalar_fp CUDA::cudart)
    set_property(TARGET multispectrum_bone_bhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    set_property(TARGET multispectrum_two_material_bhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    add_executable(multispectrum_bone_bhc_2dcal EXCLUDE_FROM_ALL
        app/experiments/TwoMaterialBoneCorrection.cu src/config/SimConfig.cpp src/config/SpectrumSource.cpp
        src/WaterReference.cpp src/XcomAttenuationProvider.cpp src/SimLogger.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_include_directories(multispectrum_bone_bhc_2dcal PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
    target_link_libraries(multispectrum_bone_bhc_2dcal PRIVATE multispectrum_two_material_bhc nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_bone_bhc_2dcal PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
foreach(t IN ITEMS multispectrum_two_material_calibration multispectrum_two_material_bhc multispectrum_bone_bhc_2dcal)
    if(MSVC AND TARGET ${t})
        target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endforeach()

if(TARGET YKCBCT AND BUILD_TESTING)
    add_executable(multispectrum_two_material_bhc_test tests/test_two_material_bhc.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_include_directories(multispectrum_two_material_bhc_test PRIVATE include src)
    target_link_libraries(multispectrum_two_material_bhc_test PRIVATE multispectrum_two_material_bhc)
    set_property(TARGET multispectrum_two_material_bhc_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_two_material_bhc_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_test(NAME multispectrum_two_material_bhc COMMAND multispectrum_two_material_bhc_test "${CMAKE_CURRENT_BINARY_DIR}/two-material-test.json")
    add_executable(multispectrum_bone_scalar_test tests/test_bone_bhc_scalar.cu src/WaterReference.cpp
        src/XcomAttenuationProvider.cpp src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_include_directories(multispectrum_bone_scalar_test PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_bone_scalar_test PRIVATE multispectrum_bone_bhc nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_bone_scalar_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_bone_scalar_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_test(NAME multispectrum_bone_scalar COMMAND multispectrum_bone_scalar_test "${CMAKE_CURRENT_BINARY_DIR}/bone-scalar-tests")
    set_tests_properties(multispectrum_bone_scalar PROPERTIES LABELS "gpu;unit;bone_bhc")
endif()

if(TARGET YKCBCT)
    add_library(multispectrum_empirical_bhc STATIC src/recon/preprocess/correction/EmpiricalBhcStage.cu)
    target_include_directories(multispectrum_empirical_bhc PUBLIC include)
    target_link_libraries(multispectrum_empirical_bhc PUBLIC multispectrum_device_pool CUDA::cudart)
    set_property(TARGET multispectrum_empirical_bhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    add_executable(multispectrum_ebhc EXCLUDE_FROM_ALL app/experiments/EmpiricalBoneCorrection.cu
        src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_include_directories(multispectrum_ebhc PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_sources(multispectrum_ebhc PRIVATE src/XcomAttenuationProvider.cpp)
    target_link_libraries(multispectrum_ebhc PRIVATE multispectrum_empirical_bhc nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_empirical_bhc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    set_property(TARGET multispectrum_ebhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

add_library(multispectrum_ebhc_fit STATIC src/recon/preprocess/calibration/EmpiricalBhcFit.cpp)
target_include_directories(multispectrum_ebhc_fit PUBLIC include)
add_executable(multispectrum_fit_ebhc EXCLUDE_FROM_ALL app/experiments/FitEmpiricalBhc.cpp)
target_link_libraries(multispectrum_fit_ebhc PRIVATE multispectrum_ebhc_fit)
target_include_directories(multispectrum_fit_ebhc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")

if(TARGET YKCBCT AND BUILD_TESTING)
    add_executable(multispectrum_ebhc_test tests/test_empirical_bhc.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_link_libraries(multispectrum_ebhc_test PRIVATE multispectrum_empirical_bhc multispectrum_ebhc_fit)
    target_include_directories(multispectrum_ebhc_test PRIVATE include src)
    set_property(TARGET multispectrum_ebhc_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_test(NAME multispectrum_ebhc COMMAND multispectrum_ebhc_test)
endif()

if(TARGET YKCBCT)
    add_executable(multispectrum_cone_two_pass EXCLUDE_FROM_ALL app/experiments/ConeArtifactTwoPass.cu src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_cone_two_pass PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_cone_two_pass PRIVATE multispectrum_image_processing nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_cone_two_pass PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_cone_two_pass PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_library(multispectrum_ebhc_paper_core STATIC src/recon/postprocess/stages/correction/EmpiricalBoneCorrection.cu)
    target_include_directories(multispectrum_ebhc_paper_core PUBLIC include)
    target_link_libraries(multispectrum_ebhc_paper_core PUBLIC multispectrum_device_pool CUDA::cudart)
    set_property(TARGET multispectrum_ebhc_paper_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    add_executable(multispectrum_ebhc_paper EXCLUDE_FROM_ALL app/experiments/EbhcPaper.cu src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_ebhc_paper PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_ebhc_paper PRIVATE multispectrum_ebhc_paper_core nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_ebhc_paper PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc_paper PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(multispectrum_ebhc_paper_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

if(TARGET YKCBCT AND BUILD_TESTING)
    add_executable(multispectrum_ebhc_paper_test tests/test_ebhc_paper.cu)
    target_include_directories(multispectrum_ebhc_paper_test PRIVATE include src)
    target_link_libraries(multispectrum_ebhc_paper_test PRIVATE multispectrum_ebhc_paper_core)
    set_property(TARGET multispectrum_ebhc_paper_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc_paper_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_test(NAME multispectrum_ebhc_paper COMMAND multispectrum_ebhc_paper_test)
endif()

if(BUILD_TESTING)
    add_executable(multispectrum_bone_bhc_contract_test tests/test_bone_bhc_contract.cpp)
    target_link_libraries(multispectrum_bone_bhc_contract_test PRIVATE multispectrum_bone_bhc)
    add_test(NAME multispectrum_bone_bhc_contract COMMAND multispectrum_bone_bhc_contract_test "${CMAKE_CURRENT_BINARY_DIR}/bone-contract-tests")
    add_executable(multispectrum_two_material_calibration_test tests/test_two_material_calibration.cpp)
    target_link_libraries(multispectrum_two_material_calibration_test PRIVATE multispectrum_two_material_calibration)
    target_include_directories(multispectrum_two_material_calibration_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    add_test(NAME multispectrum_two_material_calibration COMMAND multispectrum_two_material_calibration_test "${CMAKE_CURRENT_BINARY_DIR}/calibration-cpu-tests/table.json")
    set_tests_properties(multispectrum_bone_bhc_contract multispectrum_two_material_calibration PROPERTIES LABELS "cpu;unit;multispectrum")
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc_contract_test PRIVATE /utf-8)
        target_compile_options(multispectrum_two_material_calibration_test PRIVATE /utf-8)
    endif()
endif()
if(BUILD_TESTING)
    include(tests/StageCoverage.cmake)
endif()

if(TARGET YKCBCT)
    if(NOT TARGET multispectrum_scalar_fp)
        add_subdirectory(src/primary/scalar_fp)
    endif()
    target_link_libraries(multispectrum_sim PRIVATE multispectrum_scalar_fp)
endif()

if(TARGET multispectrum_scalar_fp)
    add_executable(multispectrum_benchmark_scalar_fp EXCLUDE_FROM_ALL app/BenchmarkScalarFp.cpp
        src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_benchmark_scalar_fp PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_benchmark_scalar_fp PRIVATE multispectrum_scalar_fp nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_benchmark_scalar_fp PRIVATE /utf-8)
    endif()
endif()

if(CMAKE_CUDA_COMPILER)
    add_executable(multispectrum_benchmark_warp EXCLUDE_FROM_ALL app/BenchmarkWarpPrimary.cu
        src/config/SimConfig.cpp src/config/SpectrumSource.cpp src/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_benchmark_warp PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_benchmark_warp PRIVATE CUDA::cudart nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_benchmark_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    endif()
endif()
