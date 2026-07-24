# Manual integration tests are intentionally separate from CTest: most cases
# consume large local raw-data files and are selected by a command-line group.
set(YKCBCT_MANUAL_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/testmain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pipeline_smoke.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reconstructors_smoke.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_operator_properties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_helical_algorithms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sart.cpp
)


add_executable(ykcbct_manual_tests
    ${YKCBCT_MANUAL_TEST_SOURCES}
    ${YKCBCT_MANUAL_SUPPORT_SOURCES_ABS}
)
target_link_libraries(ykcbct_manual_tests PRIVATE
    CUDA::cudart CUDA::cufft OpenMP::OpenMP_CXX)
target_include_directories(ykcbct_manual_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(ykcbct_manual_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3 /utf-8 /Zc:__cplusplus>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/openmp:experimental>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
set_target_properties(ykcbct_manual_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
