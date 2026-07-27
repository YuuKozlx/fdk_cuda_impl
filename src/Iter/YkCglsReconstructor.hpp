#pragma once

#include "Iter/YkCglsReconstructorEx.hpp"

namespace YK::Iter {

// 标准圆轨迹入口，仅负责构造 geometry；数值实现与 Ex 完全共用。
class CglsReconstructor {
public:
    using Config = CglsReconstructionConfig;

    bool prepare(const SCBCTParams& params, const Config& config,
        cudaStream_t stream, int device_id = 0)
    {
        std::vector<SConeProjGeomVec> geometry;
        detail::buildCircularViews(params, geometry);
        return implementation_.prepare(params, geometry, config, stream, device_id);
    }

    bool reconstruct(const float* measured_projection, float* volume)
    { return implementation_.reconstruct(measured_projection, volume); }

    void release() { implementation_.release(); }
    bool isPrepared() const { return implementation_.isPrepared(); }

private:
    CglsReconstructorEx implementation_{};
};

} // namespace YK::Iter
