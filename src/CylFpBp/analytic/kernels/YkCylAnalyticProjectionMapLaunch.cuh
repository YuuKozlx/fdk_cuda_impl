#pragma once

#include <cuda_runtime.h>

namespace YK::CylFpBp::Analytic {
struct ProjectionMapConfig;
namespace detail {

void launch_cylindrical_to_equiangular_map(const float* physical_projection,
    float* equiangular_projection, const ProjectionMapConfig& config,
    cudaStream_t stream);

} // namespace detail
} // namespace YK::CylFpBp::Analytic
