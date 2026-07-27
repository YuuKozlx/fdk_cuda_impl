# 全局基础设施：日志实现。头文件通过主目标的私有 include 路径使用。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/YkLog.cpp
)
