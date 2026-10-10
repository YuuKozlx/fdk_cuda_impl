#pragma once

#include <cuda_runtime.h>

namespace YK::NonStandardFpBp {

// Projective map from a local detector pixel to fixed raw detector coordinates.
// Each coefficient triplet is ordered as constant, u, v.
struct SourceOrbitResampleMap {
    float3 numerator_u{};
    float3 numerator_v{};
    float3 denominator{};
};

bool launchSourceOrbitResample(const float* d_raw, float* d_window,
    int raw_u, int raw_v, int window_u, int window_v, int first_view, int views,
    const SourceOrbitResampleMap* d_maps, cudaStream_t stream);

} // namespace YK::NonStandardFpBp
