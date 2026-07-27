# 通用正投影 CUDA 实现：Joseph 系列和 Siddon。
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPSiddonKernel.cu
)

# CVP 仍是实验实现，仅在顶层 YKCBCT_BUILD_CVP 开启时加入主库。
if(YKCBCT_BUILD_CVP)
    target_sources(YKCBCT PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkFPCVPKernels.cu
    )
endif()
