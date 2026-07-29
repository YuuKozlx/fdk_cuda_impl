#pragma once

#include <cufft.h>
#include <cuda_runtime.h>

#include "Heli/wfbp/YkWfbpTypes.hpp"

namespace YK { namespace Helical { namespace Wfbp { namespace detail {

void launch_flat_to_equiangular_arc(const float* flat, float* arc,
    const Geometry& geometry, float flat_du_mm,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_cylindrical_to_equiangular_arc(const float* cylindrical, float* arc,
    const Geometry& geometry, float curvature_radius_mm, float arc_du_mm,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_rebin(const float* input, float* output, const Geometry& geometry,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_pad(const float* input, float* padded, int channels, int padded_channels,
    int rows, int views, const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_crop(const float* padded, float* output, int channels, int padded_channels,
    int rows, int views, const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_build_freect_filter(float* spatial_filter, int length,
    float sample_spacing, const FreeCtFilterConfig& config,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_multiply_filter(cufftComplex* data, const cufftComplex* filter,
    int complex_channels, int batch, float scale,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_backproject(const float* filtered, float* volume,
    int nx, int ny, int nz, float dx, float dy, float dz,
    float ox, float oy, float oz, const Geometry& geometry,
    float redundancy_flat, const SKernelLaunchPolicy& policy, cudaStream_t stream);

} } } } // namespace YK::Helical::Wfbp::detail
