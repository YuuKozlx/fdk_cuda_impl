# 运行时工具实现；CLI11 为单头文件，不需要加入 target_sources。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/YkIoDump.cpp
)
