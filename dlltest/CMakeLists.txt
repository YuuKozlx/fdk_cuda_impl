# DLL 接口测试只使用公共头文件和导出目标，不访问 src 内部实现。
add_executable(dlltest test_Task_dll.cpp)

target_link_libraries(dlltest PRIVATE
    YKCBCT::YKCBCT
)

target_compile_options(dlltest PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /Zc:__cplusplus /utf-8>
)

set_target_properties(dlltest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_custom_command(TARGET dlltest POST_BUILD
    # Windows 下将运行时 DLL 放到测试程序同目录；其他平台同样可安全执行。
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:dlltest>
)
