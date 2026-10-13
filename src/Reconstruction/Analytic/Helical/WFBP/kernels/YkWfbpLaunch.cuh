#pragma once

#include <cufft.h>
#include <cuda_runtime.h>

#include "Reconstruction/Analytic/Helical/WFBP/YkWfbpTypes.hpp"

namespace YK { namespace Helical { namespace Wfbp { namespace detail {

void launch_flat_to_equiangular_arc(const float* flat, float* arc,
    const Geometry& geometry, float flat_du_mm,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_rebin(const float* input, float* output, const Geometry& geometry,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_pad(const float* input, float* padded, int channels, int padded_channels,
    int rows, int views, const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_crop(const float* padded, float* output, int channels, int padded_channels,
    int rows, int views, const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_backproject(const float* filtered, float* volume,
    int nx, int ny, int nz, float dx, float dy, float dz,
    float ox, float oy, float oz, const Geometry& geometry,
    float redundancy_flat, const SKernelLaunchPolicy& policy, cudaStream_t stream);

void launch_rebin_range(const float* input, float* output, const Geometry& geometry,
    int first_output, int output_views, int first_raw, int raw_views,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);
void launch_backproject_phases(const float* filtered, float* volume,
    int nx, int ny, int nz, float dx, float dy, float dz,
    float ox, float oy, float oz, const Geometry& geometry,
    float redundancy_flat, int first_phase, int phases, bool clear,
    const SKernelLaunchPolicy& policy, cudaStream_t stream);

} } } } // namespace YK::Helical::Wfbp::detail
