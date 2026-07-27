# 圆柱探测器正反投影 CUDA 实现。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylFpBpKernels.cu
)
