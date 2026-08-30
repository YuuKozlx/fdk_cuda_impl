#pragma once

#include <vector>
#include <cmath>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/kernels/siddon/YkCylSiddonLaunch.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp::detail {

enum class ESiddonBackprojectorVersion {
    V2,
    V3
};

// 圆柱 Siddon 的体素驱动优化实现。V2 使用连续探测器坐标和 Linear
// 插值；V3 使用最近像素射线并按 Z 分块累加。二者都计算射线与体素盒的
// 实际交长，但不等于 ray-driven FP 的离散转置；严格伴随测试应继续使用
// SiddonRayDriven。
class SiddonVoxelStrategy {
public:
    ~SiddonVoxelStrategy() { release(); }
    SiddonVoxelStrategy() = default;
    SiddonVoxelStrategy(const SiddonVoxelStrategy&) = delete;
    SiddonVoxelStrategy& operator=(const SiddonVoxelStrategy&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        ESiddonBackprojectorVersion version, cudaStream_t stream,
        int device_id = 0)
    {
        release();
        if (!stream || channels <= 0 || rows <= 0 || geometry.empty() ||
            volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0) return false;
        std::vector<SVoxelDrivenView> packed;
        std::vector<SCylVoxelChannelRay> channel_rays;
        packed.reserve(geometry.size());
        channel_rays.reserve(geometry.size() * static_cast<size_t>(channels));
        for (const auto& view : geometry) {
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(view, channels, rows, frame)) return false;
            SVoxelDrivenView item{};
            item.source = view.source;
            item.radial_unit = make_float4(frame.radialUnit.x,
                frame.radialUnit.y, frame.radialUnit.z, 0.f);
            item.tangent_unit = make_float4(frame.tangentUnit.x,
                frame.tangentUnit.y, frame.tangentUnit.z, 0.f);
            item.axis_unit = make_float4(frame.axisUnit.x, frame.axisUnit.y,
                frame.axisUnit.z, 0.f);
            item.cylinder_center = make_float4(frame.cylinderCenter.x,
                frame.cylinderCenter.y, frame.cylinderCenter.z, 0.f);
            item.radius_mm = frame.radius_mm;
            item.principal_u = frame.principalU;
            item.principal_v = frame.principalV;
            item.inv_channel_angle_step_rad = 1.f / frame.channelStepRad;
            item.inv_row_step_mm = 1.f / frame.rowStepMm;
            const float3 source = make_float3(view.source.x, view.source.y,
                view.source.z);
            const float3 center = frame.cylinderCenter;
            const float3 delta = make_float3(source.x - center.x,
                source.y - center.y, source.z - center.z);
            const float axial = delta.x * frame.axisUnit.x +
                delta.y * frame.axisUnit.y + delta.z * frame.axisUnit.z;
            const float3 perp = make_float3(delta.x - axial * frame.axisUnit.x,
                delta.y - axial * frame.axisUnit.y,
                delta.z - axial * frame.axisUnit.z);
            item.source_perp = make_float4(perp.x, perp.y, perp.z, 0.f);
            item.source_perp_norm_sq = perp.x * perp.x + perp.y * perp.y +
                perp.z * perp.z;
            item.source_on_axis = item.source_perp_norm_sq < 1e-8f ? 1u : 0u;
            packed.push_back(item);
            for (int channel = 0; channel < channels; ++channel) {
                const float angle = (channel - frame.principalU) *
                    frame.channelStepRad;
                const float sine = std::sin(angle), cosine = std::cos(angle);
                const float3 point = make_float3(
                    frame.cylinderCenter.x + frame.radius_mm *
                        (frame.radialUnit.x * cosine + frame.tangentUnit.x * sine),
                    frame.cylinderCenter.y + frame.radius_mm *
                        (frame.radialUnit.y * cosine + frame.tangentUnit.y * sine),
                    frame.cylinderCenter.z + frame.radius_mm *
                        (frame.radialUnit.z * cosine + frame.tangentUnit.z * sine));
                channel_rays.push_back({make_float4(point.x, point.y, point.z, 0.f)});
            }
        }
        standard_geometry_ = true;
        for (const auto& view : packed) {
            const float3 axis = make_float3(view.axis_unit.x, view.axis_unit.y,
                view.axis_unit.z);
            if (!view.source_on_axis || std::fabs(axis.x) > 1e-6f ||
                std::fabs(axis.y) > 1e-6f) { standard_geometry_ = false; break; }
        }
        YK_CUDA_CHECK(cudaSetDevice(device_id));
        Mem::PodDataController pod;
        d_geometry_ = pod.allocateAndUpload(packed, device_id);
        d_channel_rays_ = pod.allocateAndUpload(channel_rays, device_id);
        const auto filter = version == ESiddonBackprojectorVersion::V2
            ? cudaFilterModeLinear : cudaFilterModePoint;
        projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels, rows, static_cast<int>(geometry.size()), filter,
            cudaAddressModeBorder);
        if (!d_geometry_ || !d_channel_rays_ || !projection_texture_.valid()) return false;
        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        version_ = version;
        stream_ = stream;
        device_id_ = device_id;
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
        prepared_ = true;
        return true;
    }

    bool uploadProjection(const float* projection)
    {
        if (!prepared_ || !projection) return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(projection_texture_,
            projection, channels_, rows_, views_, stream_);
        record_();
        return true;
    }

    bool backproject(float* volume, bool accumulate = false)
    {
        if (!prepared_ || !volume) return false;
        if (version_ == ESiddonBackprojectorVersion::V2)
            launch_cyl_siddon_backproject_v2(projection_texture_.tex,
                volume, d_geometry_.data(), views_, volume_geometry_, accumulate,
                stream_);
        else
            (standard_geometry_ ? launch_cyl_siddon_backproject_v3_standard
                : launch_cyl_siddon_backproject_v3)(projection_texture_.tex,
                volume, d_geometry_.data(), d_channel_rays_.data(), views_, channels_,
                volume_geometry_, accumulate,
                stream_);
        record_();
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
        projection_texture_ = {};
        d_geometry_ = {};
        d_channel_rays_ = {};
        stream_ = nullptr;
        channels_ = rows_ = views_ = 0;
        prepared_ = false;
        standard_geometry_ = false;
    }

private:
    void record_()
    {
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
    }

    SVolGeom volume_geometry_{};
    Mem::DeviceLinearBuffer<SVoxelDrivenView> d_geometry_{};
    Mem::DeviceLinearBuffer<SCylVoxelChannelRay> d_channel_rays_{};
    Mem::Tex3DHandle projection_texture_{};
    ESiddonBackprojectorVersion version_ = ESiddonBackprojectorVersion::V3;
    cudaEvent_t completion_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int channels_ = 0, rows_ = 0, views_ = 0, device_id_ = 0;
    bool completion_recorded_ = false;
    bool prepared_ = false;
    bool standard_geometry_ = false;
};

} // namespace YK::CylFpBp::detail
