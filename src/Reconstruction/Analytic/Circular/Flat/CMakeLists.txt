# FDK 预加权、滤波、Parker 权重和反投影 CUDA 实现。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CFDK/kernels/YkCurveFilteredFdkKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/XFDK/kernels/YkXfdkKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/FDK/kernels/YkFDKBpKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/FDK/kernels/YkFDKBpPrecompute.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/FDK/kernels/YkFDKFilterKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/FDK/kernels/YkFDKParkerKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/FDK/kernels/YkFDKPreWeightKernels.cu
)
