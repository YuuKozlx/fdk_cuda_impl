#pragma once

#include <cuda_runtime.h>

#include "FDK/kernels/YkCurveFilteredFdkTypes.cuh"
#include "global/YkKernelLaunchPolicy.hpp"

namespace YK::Fdk::detail {

// 将真实平板投影 P(beta,a,b) 重排到论文的 cone-parallel (theta,t,c)
// 网格，并完成式 (31) 中 ramp 滤波之前的预加权。
void launchCurveFilteredFdkRebinPreweight(
    cudaTextureObject_t input_projection,
    float* output,
    const SCurveFilteredFdkGeometry& geometry,
    const SKernelLaunchPolicy& policy,
    cudaStream_t stream);

// 按论文式 (32)-(36) 对滤波后的 (theta,t,c) 数据反投影。该公式没有
// 普通 FDK 的距离平方权重，离散积分系数为 0.5*dtheta。
void launchCurveFilteredFdkBackprojection(
    cudaTextureObject_t filtered_projection,
    float* volume,
    const SCurveFilteredFdkGeometry& geometry,
    const SVolGeom& volume_geometry,
    bool accumulate,
    const SKernelLaunchPolicy& policy,
    cudaStream_t stream);

} // namespace YK::Fdk::detail
