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
add_executable(multispectrum_sim app/main.cpp src/SimLogger.cpp src/config/SimConfig.cpp src/SpectralModel.cpp src/XcomAttenuationProvider.cpp src/ProjectionSimulator.cpp)
target_include_directories(multispectrum_sim PRIVATE
    include "${SPECTRAL_XCOM_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/util")
target_link_libraries(multispectrum_sim PRIVATE nist_xcom_core)

# 在线多能谱前投由示例自身的 CUDA kernel 完成；YKCBCT DLL 只提供重建后端。
if(TARGET YKCBCT)
    target_sources(multispectrum_sim PRIVATE src/LibraryGeometry.cpp src/LibraryReconstructor.cpp src/CudaPrimaryProjector.cu)
    target_include_directories(multispectrum_sim PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
    target_link_libraries(multispectrum_sim PRIVATE YKCBCT)
    target_compile_definitions(multispectrum_sim PRIVATE
        YK_MULTISPECTRUM_CUDA_PRIMARY=1)
    add_custom_command(TARGET multispectrum_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_sim>)
else()
    target_sources(multispectrum_sim PRIVATE src/LibraryReconstructorStub.cpp)
endif()
if(MSVC)
    target_compile_options(multispectrum_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

include(CTest)
if(BUILD_TESTING)
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
    target_compile_definitions(multispectrum_xcom_test PRIVATE
        XCOM_TEST_DATA_DIRECTORY="${SPECTRAL_XCOM_ROOT}/data")
    if(MSVC)
        target_compile_options(multispectrum_xcom_test PRIVATE /utf-8)
    endif()
    add_test(NAME multispectrum_xcom_water COMMAND multispectrum_xcom_test)

endif()
