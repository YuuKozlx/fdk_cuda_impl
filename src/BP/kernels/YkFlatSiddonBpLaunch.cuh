#pragma once
#include <cuda_runtime.h>
#include "global/YkGlobals.h"

namespace YK::Bp {
// Ray-driven Siddon BP 用 atomicAdd 累加，因此入口没有覆盖模式。
void bp_siddon_launch(const float* projection, float* volume,
    const SConeProjGeomVec* views, const SVolGeom& volume_geometry,
    int channels, int rows, int views_count, cudaStream_t stream);
void bp_siddon_launch(cudaTextureObject_t projection_texture, float* volume,
    const SConeProjGeomVec* views, const SVolGeom& volume_geometry,
    int channels, int rows, int views_count, cudaStream_t stream);
void bp_siddon_voxel_launch(const float* projection, float* volume,
    const SConeProjGeomVec* views, const SVolGeom& volume_geometry,
    int channels, int rows, int views_count, bool accumulate,
    cudaStream_t stream);
void bp_siddon_voxel_v2_launch(cudaTextureObject_t projection_texture,
    float* volume, const SConeProjGeomVec* views,
    const SVolGeom& volume_geometry, int channels, int rows, int views_count,
    bool accumulate, cudaStream_t stream);
} // namespace YK::Bp
