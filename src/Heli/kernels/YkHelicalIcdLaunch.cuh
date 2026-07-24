#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

struct CUstream_st;

namespace YK {
namespace Helical {

// Parallel ICD update with diagonal data curvature and a 6-neighbor prior.
// residual_bp follows A^T(b-Ax), so a positive value increases the voxel.
void helical_icd_update_launch(float* volume, const float* residual_bp,
    const float* data_curvature, int nx, int ny, int nz,
    float relaxation, float regularization, float epsilon,
    float lower_bound, float upper_bound, ::CUstream_st* stream);

} // namespace Helical
} // namespace YK
