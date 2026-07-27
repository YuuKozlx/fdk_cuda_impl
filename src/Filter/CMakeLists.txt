# 滤波卷积与频域滤波核构建 CUDA 实现。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/YkConvKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/YkCreateFilterKernelKernels.cu
)
