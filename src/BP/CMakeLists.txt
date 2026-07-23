# BP 模块目前全是头文件，无需添加源文件
# 如后续有 .cu 或 .cpp 文件，在此添加：
# target_sources(YKCBCT PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/YkBPKernels.cu
# )
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkBPSiddonKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkBPJosephKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/YkBpFdkKernel.cu
)
