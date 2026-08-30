#pragma once

#include <cmath>
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
        std::vector<SSiddonChannelRay> siddon_channel_rays;
        packed_views.reserve(geometry.size());
        siddon_channel_rays.reserve(geometry.size() *
            static_cast<size_t>(channels));
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

            // 圆柱像素可分解为“通道基点 + 行方向偏移”。预计算通道基点
            // 可以从每次迭代的 FP/BP 中移除 sin/cos，同时 FP 与 BP 使用
            // 完全相同的射线数据，保持离散 Siddon 伴随关系。
            for (int channel = 0; channel < channels; ++channel) {
                const float delta = (channel - frame.principalU) *
                    frame.channelStepRad;
                const float sine = std::sin(delta);
                const float cosine = std::cos(delta);
                const float3 detector = make_float3(
                    frame.cylinderCenter.x + frame.radius_mm *
                        (frame.radialUnit.x * cosine +
                         frame.tangentUnit.x * sine),
                    frame.cylinderCenter.y + frame.radius_mm *
                        (frame.radialUnit.y * cosine +
                         frame.tangentUnit.y * sine),
                    frame.cylinderCenter.z + frame.radius_mm *
                        (frame.radialUnit.z * cosine +
                         frame.tangentUnit.z * sine));
                SSiddonChannelRay ray{};
                ray.ray_at_principal_row = make_float4(
                    detector.x - view.source.x,
                    detector.y - view.source.y,
                    detector.z - view.source.z, 0.f);
                siddon_channel_rays.push_back(ray);
            }
        }

        YK_CUDA_CHECK(cudaSetDevice(device_id));
        Mem::PodDataController pod;
        device_views_ = pod.allocateAndUpload(packed_views, device_id);
        device_siddon_channel_rays_ = pod.allocateAndUpload(
            siddon_channel_rays, device_id);
        if (!device_views_ || !device_siddon_channel_rays_) return false;
        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        device_id_ = device_id;
        return true;
    }

    const SKernelView* data() const { return device_views_.data(); }
    const SSiddonChannelRay* siddonChannelRays() const
    { return device_siddon_channel_rays_.data(); }
    const SVolGeom& volumeGeometry() const { return volume_geometry_; }
    int channels() const { return channels_; }
    int rows() const { return rows_; }
    int views() const { return views_; }
    int device() const { return device_id_; }

private:
    Mem::DeviceLinearBuffer<SKernelView> device_views_{};
    Mem::DeviceLinearBuffer<SSiddonChannelRay> device_siddon_channel_rays_{};
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
