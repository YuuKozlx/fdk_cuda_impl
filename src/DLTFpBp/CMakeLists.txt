# DLT matrix driven ray geometry and its matched Siddon FP/BP kernels.
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkDltAlgebraicKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkDltSiddonKernels.cu
)
