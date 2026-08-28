# 圆柱探测器正反投影 CUDA 实现。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylFpBpLegacyKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylFpBpKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylFpBpJosephKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylFpBpVoxelDrivenKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylFdkBackprojectorKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylSiddonForwardKernels.cu
)
