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
    ~Operator() { release(); }

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
        waitPrevious_(stream);
        detail::launch_forward(volume, projection, d_geometry_.data(), views_, rows_,
            channels_, volume_geometry_, config_, stream, accumulate);
        record_(stream);
        return true;
    }

    bool backproject(const float* projection, float* volume, cudaStream_t stream,
        bool accumulate = false) const
    {
        if (!prepared_ || !volume || !projection || !stream) return false;
        waitPrevious_(stream);
        detail::launch_backproject(projection, volume, d_geometry_.data(), views_,
            rows_, channels_, volume_geometry_, config_, stream, accumulate);
        record_(stream);
        return true;
    }

    void release()
    {
        if (completion_) {
            YK_CUDA_CHECK(cudaSetDevice(device_id_));
            if (completion_recorded_) YK_CUDA_CHECK(cudaEventSynchronize(completion_));
            YK_CUDA_CHECK(cudaEventDestroy(completion_));
        }
        completion_ = nullptr;
        completion_recorded_ = false;
        last_stream_ = nullptr;
        d_geometry_ = {};
        volume_geometry_ = {};
        channels_ = rows_ = views_ = 0;
        prepared_ = false;
    }

private:
    void waitPrevious_(cudaStream_t stream) const
    {
        if (completion_recorded_ && stream != last_stream_)
            YK_CUDA_CHECK(cudaEventSynchronize(completion_));
    }

    void record_(cudaStream_t stream) const
    {
        if (!completion_) {
            YK_CUDA_CHECK(cudaSetDevice(device_id_));
            YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
        }
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream));
        last_stream_ = stream;
        completion_recorded_ = true;
    }

    SVolGeom volume_geometry_{};
    Config config_{};
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
    bool prepared_ = false;
    Mem::DeviceLinearBuffer<SCylConeProjGeomVec> d_geometry_{};
    mutable cudaEvent_t completion_ = nullptr;
    mutable cudaStream_t last_stream_ = nullptr;
    mutable bool completion_recorded_ = false;
};

} // namespace YK::CylFpBp
