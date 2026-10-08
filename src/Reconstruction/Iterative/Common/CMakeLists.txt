# 迭代重建共享 CUDA kernel。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkIterKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkParallelPwlsKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkTvRegularizationKernels.cu
)
