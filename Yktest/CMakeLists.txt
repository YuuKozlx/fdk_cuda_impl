add_executable(YKCBCT_test
    ${CMAKE_CURRENT_SOURCE_DIR}/testmain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fdk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp1.cu
    $<TARGET_OBJECTS:YKCBCT_obj>
)

target_link_libraries(YKCBCT_test PRIVATE
    YKCBCT_obj
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

target_compile_options(YKCBCT_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8>
)

target_include_directories(YKCBCT_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(YKCBCT_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 20
)
