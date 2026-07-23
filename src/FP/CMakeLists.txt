target_sources(YKCBCT_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPSiddonKernel.cu
)

# CVP is retained as an experimental implementation, but is not part of the
# regular forward-projection library unless it is explicitly requested.
option(YKCBCT_BUILD_CVP "Build experimental CVP forward projector" OFF)
if(YKCBCT_BUILD_CVP)
    target_sources(YKCBCT_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPCVPKernels.cu
    )
endif()
