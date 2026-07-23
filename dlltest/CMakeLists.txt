cmake_minimum_required(VERSION 3.20)
project(YKCBCTDllTest LANGUAGES CXX)

# ----------------------------------------------------------------
# 标准
# ----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------
# 依赖
# 这里用 find_package 找已安装/已构建的 YKCBCT 包
# 外部调用方只需要知道 YKCBCT::YKCBCT 这个 imported target
# ----------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(spdlog      CONFIG REQUIRED)

# 方式一：如果 YKCBCT 已经 install 到系统或 CMAKE_PREFIX_PATH
# find_package(YKCBCT CONFIG REQUIRED)
# target_link_libraries(dlltest PRIVATE YKCBCT::YKCBCT)

# 方式二：直接引用上层构建目录（开发期间更方便，不需要 install）
# 在 VS 的 CMake 工程里，两个子项目可以直接共享 target
# ----------------------------------------------------------------
add_executable(dlltest
    ${CMAKE_CURRENT_SOURCE_DIR}/test_Task_dll.cpp
)

target_compile_definitions(dlltest PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _WINDOWS
    # FMT_UNICODE=0 避免 fmt 在 MSVC 下的 unicode 警告
    FMT_UNICODE=0
)

target_compile_options(dlltest PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
        /Zc:__cplusplus
        /utf-8
    >
)

# ----------------------------------------------------------------
# 链接
# 只链接 YKCBCT 的导入库（.lib），不链接内部 .obj
# spdlog / cudart 是测试代码自己用到的，不是通过 DLL 传来的
# ----------------------------------------------------------------
target_link_libraries(dlltest PRIVATE
    YKCBCT          # 链接导入库 YKCBCT.lib
    CUDA::cudart    # test_Task_dll.cpp 里直接调了 cudaMalloc 等
    spdlog::spdlog  # test_Task_dll.cpp 里直接用了 spdlog::info
)

# ----------------------------------------------------------------
# 头文件路径
# 只需要 include/YKCBCT/，不需要 src/ 下的内部头文件
# 这模拟了真实的外部使用场景
# ----------------------------------------------------------------
target_include_directories(dlltest PRIVATE
    ${CMAKE_SOURCE_DIR}/include   # YKCBCT/interface/*.hpp
)

set_target_properties(dlltest PROPERTIES
    CXX_STANDARD 20
)

add_custom_command(TARGET dlltest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:YKCBCT>          # YKCBCT.dll 的路径
        $<TARGET_FILE_DIR:dlltest>     # dlltest.exe 所在目录
    COMMENT "Copying YKCBCT.dll to dlltest output dir"
)