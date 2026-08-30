# 实验性螺旋重建实现，仅由顶层 YKCBCT_BUILD_HELICAL 控制加入。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/analytic/wfbp/kernels/YkWfbpKernels.cu
)
