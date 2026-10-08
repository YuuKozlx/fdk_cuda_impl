# 通用反投影 CUDA 实现：Siddon、Joseph 和 FDK 反投影。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkBPSiddonKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkBPJosephKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkBpFdkKernel.cu
)
