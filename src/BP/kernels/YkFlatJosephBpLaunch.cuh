#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "BP/YkBpCommon.cuh"
#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "global/YkFdkKernelTypes.hpp"

namespace YK::Bp {
void joseph_bp_launch(const float* projection,
    const std::vector<SConeProjGeomVec>& h_views,
    const SConeProjGeomVec* d_views_vox, float* volume,
    const SVolGeom& volume_geometry, int views, int channels, int rows,
    bool accumulate, cudaStream_t stream,
    BpStepSuperSample step = BpStepSuperSample::x1);
void joseph_bp_launch(cudaTextureObject_t projection_texture,
    const std::vector<SConeProjGeomVec>& h_views,
    const SConeProjGeomVec* d_views_vox, float* volume,
    const SVolGeom& volume_geometry, int views, int channels, int rows,
    bool accumulate, cudaStream_t stream,
    BpStepSuperSample step = BpStepSuperSample::x1);
// 兼容名称：v2 是体素驱动 Joseph，v3 是预计算仿射体素 BP。
void joseph_bp_v2_launch(cudaTextureObject_t projection_texture,
    const SConeProjGeomVec* d_views_world, float* volume,
    const SVolGeom& volume_geometry, int views, int channels, int rows,
    bool accumulate, cudaStream_t stream);
void joseph_bp_v3_launch(cudaTextureObject_t projection_texture,
    const SConeProjGeomVec* d_views, const FdkAffineCoeff* coefficients,
    float* volume, const SVolGeom& volume_geometry, int views,
    bool accumulate, cudaStream_t stream);
} // namespace YK::Bp
