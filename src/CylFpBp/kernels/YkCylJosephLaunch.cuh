#pragma once
#include <cuda_runtime.h>
#include "CylFpBp/kernels/YkCylKernelTypes.cuh"

namespace YK::CylFpBp::detail {
void launch_main_axis_forward(const float*, float*, const SKernelView*, int, int,
    int, const SVolGeom&, const Config&, cudaStream_t, bool);
void launch_main_axis_forward_texture(cudaTextureObject_t, float*,
    const SKernelView*, int, int, int, const SVolGeom&, const Config&,
    cudaStream_t, bool);
// 严格伴随对照路径，不由默认 Operator 调用。
void launch_main_axis_forward_texture_matched(cudaTextureObject_t, float*,
    const SKernelView*, int, int, int, const SVolGeom&, const Config&,
    cudaStream_t, bool);
void launch_main_axis_backproject(const float*, float*, const SKernelView*, int,
    int, int, const SVolGeom&, const Config&, cudaStream_t, bool);
void launch_main_axis_backproject_texture(cudaTextureObject_t, float*,
    const SKernelView*, int, int, int, const SVolGeom&, const Config&,
    cudaStream_t, bool);
} // namespace YK::CylFpBp::detail
