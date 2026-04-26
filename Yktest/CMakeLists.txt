add_executable(YKCBCT_test
    ${CMAKE_CURRENT_SOURCE_DIR}/testmain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fdk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_heli.cpp
    $<TARGET_OBJECTS:YKCBCT_obj>
 )

target_link_libraries(YKCBCT_test PRIVATE
    CUDA::cudart
    CUDA::cufft
    spdlog::spdlog
)

target_compile_options(YKCBCT_test PRIVATE
        # MSVC 编译器选项
        $<$<CXX_COMPILER_ID:MSVC>:/W3 /utf-8 /Zc:__cplusplus>

        # GCC 编译器选项
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>

        # Clang 编译器选项
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)
target_include_directories(YKCBCT_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(YKCBCT_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
)