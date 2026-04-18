target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuh/YkFDKBpKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuh/YkFDKBpPrecompute.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuh/YkFDKFilterKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuh/YkFDKParkerKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuh/YkFDKPreWeightKernels.cu
)
