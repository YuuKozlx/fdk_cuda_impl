#pragma once

#include <memory>
#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/kernels/YkCylFpBpLaunch.cuh"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp::detail {

// Joseph FP/BP 共用的不可变预计算几何。正投和反投只保存共享句柄，
// 最后一个算子释放后设备几何才会销毁。
class PreparedJosephGeometry {
public:
    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry, int device_id)
    {
        if (volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0 || channels <= 0 || rows <= 0 ||
            geometry.empty()) return false;

        std::vector<SKernelView> packed_views;
        packed_views.reserve(geometry.size());
        for (const auto& view : geometry) {
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(view, channels, rows, frame)) return false;
            SKernelView packed{};
            packed.source = view.source;
            packed.radial_unit = make_float4(frame.radialUnit.x,
                frame.radialUnit.y, frame.radialUnit.z, 0.f);
            packed.tangent_unit = make_float4(frame.tangentUnit.x,
                frame.tangentUnit.y, frame.tangentUnit.z, 0.f);
            packed.cylinder_center = make_float4(frame.cylinderCenter.x,
                frame.cylinderCenter.y, frame.cylinderCenter.z, 0.f);
            packed.detector_v = view.detectorV;
            packed.radius_mm = frame.radius_mm;
            packed.principal_u = frame.principalU;
            packed.principal_v = frame.principalV;
            packed.channel_angle_step_rad = frame.channelStepRad;
            packed_views.push_back(packed);
        }

        YK_CUDA_CHECK(cudaSetDevice(device_id));
        Mem::PodDataController pod;
        device_views_ = pod.allocateAndUpload(packed_views, device_id);
        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        device_id_ = device_id;
        return true;
    }

    const SKernelView* data() const { return device_views_.data(); }
    const SVolGeom& volumeGeometry() const { return volume_geometry_; }
    int channels() const { return channels_; }
    int rows() const { return rows_; }
    int views() const { return views_; }
    int device() const { return device_id_; }

private:
    Mem::DeviceLinearBuffer<SKernelView> device_views_{};
    SVolGeom volume_geometry_{};
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
};

using JosephGeometryHandle = std::shared_ptr<const PreparedJosephGeometry>;

inline JosephGeometryHandle prepareJosephGeometry(
    const SVolGeom& volume_geometry, int channels, int rows,
    const std::vector<SCylConeProjGeomVec>& geometry, int device_id = 0)
{
    auto prepared = std::make_shared<PreparedJosephGeometry>();
    if (!prepared->prepare(volume_geometry, channels, rows, geometry, device_id))
        return {};
    return prepared;
}

// 每个计算算子拥有独立完成事件；共享几何本身是只读的，不需要在 FP/BP
// 之间建立串行依赖。事件只保护该算子跨 stream 重用和析构时的生命周期。
class OperatorCompletion {
public:
    ~OperatorCompletion() { release(); }

    bool prepare(int device_id)
    {
        release();
        device_id_ = device_id;
        YK_CUDA_CHECK(cudaSetDevice(device_id_));
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        return true;
    }

    void waitBefore(cudaStream_t stream) const
    {
        if (recorded_ && stream != last_stream_)
            YK_CUDA_CHECK(cudaEventSynchronize(event_));
    }

    void record(cudaStream_t stream) const
    {
        YK_CUDA_CHECK(cudaEventRecord(event_, stream));
        last_stream_ = stream;
        recorded_ = true;
    }

    void release()
    {
        if (event_) {
            YK_CUDA_CHECK(cudaSetDevice(device_id_));
            if (recorded_) YK_CUDA_CHECK(cudaEventSynchronize(event_));
            YK_CUDA_CHECK(cudaEventDestroy(event_));
        }
        event_ = nullptr;
        last_stream_ = nullptr;
        recorded_ = false;
    }

private:
    cudaEvent_t event_ = nullptr;
    int device_id_ = 0;
    mutable cudaStream_t last_stream_ = nullptr;
    mutable bool recorded_ = false;
};

} // namespace YK::CylFpBp::detail
