cmake_minimum_required(VERSION 3.24)
project(multispectrum_sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    include "${SPECTRAL_XCOM_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
target_link_libraries(multispectrum_sim PRIVATE nist_xcom_core)

# 在线多能谱前投由示例自身的 CUDA kernel 完成；YKCBCT DLL 只提供重建后端。
if(TARGET YKCBCT)
    target_sources(multispectrum_sim PRIVATE
        src/LibraryGeometry.cpp src/LibraryReconstructor.cpp
        src/CudaPrimaryProjector.cu src/SpectralProjectionPipeline.cpp
        src/primary/CudaProjectionBackend.cu src/primary/DetectorSignal.cpp
        src/primary/ProjectionStages.cpp src/primary/ProjectionOutput.cpp
        src/CudaFasksScatter.cu)
    target_include_directories(multispectrum_sim PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(multispectrum_sim PRIVATE YKCBCT)
    target_link_libraries(multispectrum_sim PRIVATE CUDA::cufft)
    target_compile_definitions(multispectrum_sim PRIVATE
        YK_MULTISPECTRUM_CUDA_PRIMARY=1)
    add_custom_command(TARGET multispectrum_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_sim>)
else()
    target_sources(multispectrum_sim PRIVATE src/LibraryReconstructorStub.cpp)
endif()
if(MSVC)
    target_compile_options(nist_xcom_core PRIVATE /utf-8)
    target_compile_options(multispectrum_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

include(CTest)
if(BUILD_TESTING)
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
        foreach(module IN ITEMS geometry paths noise response transmission focal)
            add_test(NAME multispectrum_unit_${module} COMMAND multispectrum_cuda_unit_test ${module})
            set_tests_properties(multispectrum_unit_${module} PROPERTIES LABELS "unit;gpu;multispectrum")
        endforeach()
        add_executable(multispectrum_pipeline_test tests/test_projection_pipeline.cpp
            src/primary/CudaProjectionBackend.cu src/LibraryGeometry.cpp
            src/SpectralProjectionPipeline.cpp src/SpectralModel.cpp src/SimLogger.cpp
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
