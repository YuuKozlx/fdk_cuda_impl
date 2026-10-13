# 通用正投影 CUDA 实现：Joseph 系列和 Siddon。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPSiddonKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFlatFootprintKernels.cu
)
