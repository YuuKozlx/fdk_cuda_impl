#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "FlatFpBp/FP/YkFPCommon.cuh"
#include "global/YkGlobals.h"

namespace YK::Fp {
// h_views 仅用于从最终 geometry 派生主轴，d_views_vox 是 kernel 几何真源。
void fp_joseph_launch(cudaTextureObject_t volume_texture,
    const std::vector<SConeProjGeomVec>& h_views,
    const SConeProjGeomVec* d_views_vox, float* projection,
    const SVolGeom& volume_geometry, int views, int channels, int rows,
    bool accumulate, cudaStream_t stream,
    FpStepSuperSample step = FpStepSuperSample::x1);
void fp_joseph_ss_launch(cudaTextureObject_t volume_texture,
    const std::vector<SConeProjGeomVec>& h_views,
    const SConeProjGeomVec* d_views_vox, float* projection,
    const SVolGeom& volume_geometry, int views, int channels, int rows,
    bool accumulate, cudaStream_t stream,
    FpStepSuperSample step = FpStepSuperSample::x1,
    FpDetSuperSample detector = FpDetSuperSample::x1);
} // namespace YK::Fp
