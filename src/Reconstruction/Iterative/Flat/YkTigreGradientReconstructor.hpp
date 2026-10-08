#pragma once

#include "Reconstruction/Iterative/Flat/YkTigreGradientReconstructorEx.hpp"

namespace YK::Iter {

class TigreGradientReconstructor {
public:
    using Config = TigreGradientConfig;

    bool prepare(const SReconstructionParams& params, const Config& config,
        cudaStream_t stream, int device_id = 0)
    {
        std::vector<SConeProjGeomVec> geometry;
        detail::buildCircularViews(params, geometry);
        return implementation_.prepare(params, geometry, config, stream, device_id);
    }

    bool reconstruct(const float* d_projection, float* d_volume)
    { return implementation_.reconstruct(d_projection, d_volume); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }
    int actualSubsetCount() const { return implementation_.actualSubsetCount(); }
    const TigreGradientStatistics& statistics() const
    { return implementation_.statistics(); }

private:
    TigreGradientReconstructorEx implementation_{};
};

} // namespace YK::Iter
