# 圆柱探测器正反投影 CUDA 实现。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/analytic/kernels/YkCylAnalyticProjectionMapKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/legacy/YkCylFpBpLegacyKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/legacy/YkCylFpBpKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/joseph/YkCylFpBpJosephKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/joseph/YkCylJosephBackprojectV3Kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fdk/YkCylFdkBackprojectorKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/iter/kernel/YkCylIterFdkBackprojectorKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/siddon/YkCylSiddonForwardKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/siddon/YkCylSiddonBackprojectKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/siddon/YkCylSiddonVoxelBackprojectKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkCylPwlsKernels.cu
)
