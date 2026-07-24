target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFDKBpKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFDKBpPrecompute.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFDKFilterKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFDKParkerKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFDKPreWeightKernels.cu
)
