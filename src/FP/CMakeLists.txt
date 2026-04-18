target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPCVPKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPSiddonKernel.cu
)
