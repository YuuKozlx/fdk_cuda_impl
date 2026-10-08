#pragma once

#include <cuda_runtime.h>

#include "Reconstruction/Analytic/Circular/Flat/XFDK/kernels/YkXfdkTypes.cuh"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKVecGeoDerived.hpp"
#include "global/YkKernelLaunchPolicy.hpp"

namespace YK::Fdk::detail {

void launchXfdkRebin(cudaTextureObject_t input, float* output,
    const SXfdkGeometry& geometry, const SXfdkChunk& chunk,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launchXfdkBackprojection(cudaTextureObject_t filtered, float* volume,
    const SXfdkGeometry& geometry, const SVolGeom& volume_geometry,
    const SXfdkChunk& chunk, bool accumulate,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

} // namespace YK::Fdk::detail
