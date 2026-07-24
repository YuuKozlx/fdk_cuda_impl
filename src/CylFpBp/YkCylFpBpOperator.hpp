#pragma once

#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/kernels/YkCylFpBpLaunch.cuh"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp {

// 圆柱探测器匹配算子。FP/BP 使用完全相同的离散射线采样和三线性权重；
// BP 是 FP 的转置散射形式，适合共轭性验证和迭代算法，而不是 FDK 权重 BP。
class Operator {
public:
    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const Config& config = {}, int device_id = 0)
    {
        release();
        if (volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0 || channels <= 0 || rows <= 0 ||
            geometry.empty() || config.samples_per_voxel <= 0.f) return false;
        for (const auto& view : geometry) {
            if (!(view.radius_mm > 0.f) ||
                !(hypotf(view.detector_u_tangent.x,
                    hypotf(view.detector_u_tangent.y,
                        view.detector_u_tangent.z)) > 0.f) ||
                !(hypotf(view.detector_v.x,
                    hypotf(view.detector_v.y, view.detector_v.z)) > 0.f))
                return false;
        }
        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        config_ = config;
        device_id_ = device_id;
        Mem::PodDataController pod;
        d_geometry_ = pod.allocateAndUpload(geometry, device_id_);
        prepared_ = true;
        return true;
    }

    bool forward(const float* volume, float* projection, cudaStream_t stream,
        bool accumulate = false) const
    {
        if (!prepared_ || !volume || !projection || !stream) return false;
        detail::launch_forward(volume, projection, d_geometry_.data(), views_, rows_,
            channels_, volume_geometry_, config_, stream, accumulate);
        return true;
    }

    bool backproject(const float* projection, float* volume, cudaStream_t stream,
        bool accumulate = false) const
    {
        if (!prepared_ || !volume || !projection || !stream) return false;
        detail::launch_backproject(projection, volume, d_geometry_.data(), views_,
            rows_, channels_, volume_geometry_, config_, stream, accumulate);
        return true;
    }

    void release()
    {
        d_geometry_ = {};
        volume_geometry_ = {};
        channels_ = rows_ = views_ = 0;
        prepared_ = false;
    }

private:
    SVolGeom volume_geometry_{};
    Config config_{};
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
    bool prepared_ = false;
    Mem::DeviceLinearBuffer<SCylConeProjGeomVec> d_geometry_{};
};

} // namespace YK::CylFpBp
