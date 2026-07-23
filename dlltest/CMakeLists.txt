add_executable(dlltest test_Task_dll.cpp)

target_link_libraries(dlltest PRIVATE
    YKCBCT
    CUDA::cudart
)

target_include_directories(dlltest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(dlltest PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /Zc:__cplusplus /utf-8>
)

set_target_properties(dlltest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_custom_command(TARGET dlltest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:dlltest>
)
