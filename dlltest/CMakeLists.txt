# DLL 接口测试只使用公共头文件和导出目标，不访问 src 内部实现。
add_executable(dlltest test_Task_dll.cpp)

add_executable(dlltest_shepp_logan test_shepp_logan.cpp)
target_link_libraries(dlltest_shepp_logan PRIVATE YKCBCT::YKCBCT)
target_compile_options(dlltest_shepp_logan PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /Zc:__cplusplus /utf-8>)
set_target_properties(dlltest_shepp_logan PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_custom_command(TARGET dlltest_shepp_logan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:dlltest_shepp_logan>)
if(BUILD_TESTING)
    add_test(NAME dll_shepp_logan COMMAND dlltest_shepp_logan
        "${CMAKE_BINARY_DIR}/test-artifacts/dll-shepp-logan")
endif()

target_link_libraries(dlltest PRIVATE
    YKCBCT::YKCBCT
)

target_compile_options(dlltest PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /Zc:__cplusplus /utf-8>
)

# 测试需要区分“后端未编译进 DLL”和“wFBP 执行失败”两种情况。
# 该宏只描述构建能力，不向公共 API 暴露内部实现细节。
target_compile_definitions(dlltest PRIVATE
    YKCBCT_HAS_HELICAL=$<BOOL:${YKCBCT_BUILD_HELICAL}>
)

set_target_properties(dlltest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_custom_command(TARGET dlltest POST_BUILD
    # Windows 下将运行时 DLL 放到测试程序同目录；其他平台同样可安全执行。
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:dlltest>
)
