cmake_minimum_required(VERSION 3.26)
project(YKCBCTDllTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------
# Dependencies
# -------------------------------------------------------
find_package(CUDAToolkit REQUIRED)


# -------------------------------------------------------
# Executable
# -------------------------------------------------------
add_executable(dlltest
    test_Task_dll.cpp
)

# -------------------------------------------------------
# Compile definitions
# -------------------------------------------------------
target_compile_definitions(dlltest PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# -------------------------------------------------------
# Compile options
# -------------------------------------------------------
target_compile_options(dlltest PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /Zc:__cplusplus /utf-8>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# -------------------------------------------------------
# Include
# -------------------------------------------------------
target_include_directories(dlltest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# -------------------------------------------------------
# Link
# -------------------------------------------------------
target_link_libraries(dlltest PRIVATE
    YKCBCT
    CUDA::cudart
)

# -------------------------------------------------------
# Properties
# -------------------------------------------------------
set_target_properties(dlltest PROPERTIES
    CXX_STANDARD 20
)

# -------------------------------------------------------
# Copy DLL/SO after build
# -------------------------------------------------------
add_custom_command(TARGET dlltest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:YKCBCT>
        $<TARGET_FILE_DIR:dlltest>
)