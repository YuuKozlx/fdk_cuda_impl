add_executable(YKCBCT_test
    ${CMAKE_CURRENT_SOURCE_DIR}/testmain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp1.cu
)

target_compile_definitions(YKCBCT_test PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _WINDOWS
)

target_compile_options(YKCBCT_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
        /Zc:__cplusplus
        /utf-8
    >
)

target_link_libraries(YKCBCT_test PRIVATE
    YKCBCT
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

target_include_directories(YKCBCT_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/global
    ${CMAKE_SOURCE_DIR}/src/interface
    ${CMAKE_SOURCE_DIR}/src/util
    ${CMAKE_SOURCE_DIR}/src/Filter
    ${CMAKE_SOURCE_DIR}/src/FDK
    ${CMAKE_SOURCE_DIR}/src/FDK/cuh
    ${CMAKE_SOURCE_DIR}/src/BP
    ${CMAKE_SOURCE_DIR}/src/FP
    ${CMAKE_SOURCE_DIR}/src/FP/kernels
    ${CMAKE_SOURCE_DIR}/src/CVP
    ${CMAKE_CURRENT_SOURCE_DIR}/Yktest
)

set_target_properties(YKCBCT_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 20
)
