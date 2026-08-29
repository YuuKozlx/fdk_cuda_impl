#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/kernels/YkCylFpBpLaunch.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp {
namespace detail {

template<bool MatchedWeight>
class CylFdkBackprojectorImpl {
public:
    ~CylFdkBackprojectorImpl() { release(); }
    CylFdkBackprojectorImpl() = default;
    CylFdkBackprojectorImpl(const CylFdkBackprojectorImpl&) = delete;
    CylFdkBackprojectorImpl& operator=(const CylFdkBackprojectorImpl&) = delete;

    // 仅接受源中心等角弧面：每个 view 必须满足 R=SDD，且径向、切向、
    // 圆柱轴三者正交。一般 R!=SDD 投影必须在进入本算子前完成重采样。
    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || channels <= 0 || rows <= 0 || geometry.empty() ||
            volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0) return false;

        std::vector<SCylFdkView> packed;
        packed.reserve(geometry.size());
        for (const auto& view : geometry) {
            const float3 source = make_float3(view.source.x, view.source.y,
                view.source.z);
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(view, channels, rows, frame)) return false;
            const float3 axis_offset = make_float3(
                frame.cylinderCenter.x - source.x,
                frame.cylinderCenter.y - source.y,
                frame.cylinderCenter.z - source.z);
            const float axial = dot(axis_offset, frame.axisUnit);
            const float source_to_axis = sqrtf(std::max(0.f,
                dot(axis_offset, axis_offset) - axial * axial));
            const float radius_tolerance = std::max(1e-3f,
                1e-5f * frame.radius_mm);
            if (source_to_axis > radius_tolerance) return false;
            const auto dot = [](float3 a, float3 b) {
                return a.x * b.x + a.y * b.y + a.z * b.z;
            };

            // 当前几何约定的旋转中心是世界原点。中央射线方向 radial 指向
            // 等中心，因此 -source 在 radial 上的投影就是 SID。
            const float source_axial = dot(source, frame.axisUnit);
            const float3 source_to_center = make_float3(
                -source.x + source_axial * frame.axisUnit.x,
                -source.y + source_axial * frame.axisUnit.y,
                -source.z + source_axial * frame.axisUnit.z);
            const float sid = sqrtf(dot(source_to_center, source_to_center));
            if (!(sid > 0.f)) return false;

            SCylFdkView item{};
            item.source = view.source;
            item.radial_unit = make_float4(frame.radialUnit.x,
                frame.radialUnit.y, frame.radialUnit.z, 0.f);
            item.tangent_unit = make_float4(frame.tangentUnit.x,
                frame.tangentUnit.y, frame.tangentUnit.z, 0.f);
            item.axis_unit = make_float4(frame.axisUnit.x,
                frame.axisUnit.y, frame.axisUnit.z, 0.f);
            item.depth_unit = make_float4(source_to_center.x / sid,
                source_to_center.y / sid, source_to_center.z / sid, 0.f);
            item.radius_mm = frame.radius_mm;
            item.sid_mm = sid;
            item.principal_u = frame.principalU;
            item.principal_v = frame.principalV;
            item.inv_channel_angle_step_rad = 1.f / frame.channelStepRad;
            item.inv_row_step_mm = 1.f / frame.rowStepMm;
            item.inverse_pixel_area = 1.f /
                (frame.channelStepRad * frame.radius_mm * frame.rowStepMm);
            item.detector_center_angle_rad = atan2f(
                -dot(make_float3(item.depth_unit.x, item.depth_unit.y,
                    item.depth_unit.z), frame.tangentUnit),
                dot(make_float3(item.depth_unit.x, item.depth_unit.y,
                    item.depth_unit.z), frame.radialUnit));
            item.detector_axial_offset_mm = dot(make_float3(
                frame.detectorCenter.x - source.x,
                frame.detectorCenter.y - source.y,
                frame.detectorCenter.z - source.z), frame.axisUnit);
            packed.push_back(item);
        }

        YK_CUDA_CHECK(cudaSetDevice(device_id));
        Mem::PodDataController pod;
        d_geometry_ = pod.allocateAndUpload(packed, device_id);
        projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels, rows, static_cast<int>(geometry.size()),
            cudaFilterModeLinear, cudaAddressModeBorder);
        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        stream_ = stream;
        device_id_ = device_id;
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_,
            cudaEventDisableTiming));
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
        if constexpr (MatchedWeight) {
            launch_cyl_fdk_matched_bp(projection_texture_.tex,
                d_geometry_.data(), volume, views_, volume_geometry_, stream_,
                accumulate);
        }
        else {
            launch_cyl_fdk_bp(projection_texture_.tex, d_geometry_.data(),
                volume, views_, volume_geometry_, stream_, accumulate);
        }
        record_();
        return true;
    }

    void release()
    {
        if (completion_) {
            YK_CUDA_CHECK(cudaSetDevice(device_id_));
            if (completion_recorded_)
                YK_CUDA_CHECK(cudaEventSynchronize(completion_));
            YK_CUDA_CHECK(cudaEventDestroy(completion_));
        }
        completion_ = nullptr;
        completion_recorded_ = false;
        projection_texture_ = {};
        d_geometry_ = {};
        volume_geometry_ = {};
        stream_ = nullptr;
        views_ = rows_ = channels_ = 0;
        prepared_ = false;
    }

private:
    void record_()
    {
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
    }

    SVolGeom volume_geometry_{};
    Mem::DeviceLinearBuffer<SCylFdkView> d_geometry_{};
    Mem::Tex3DHandle projection_texture_{};
    cudaEvent_t completion_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
    bool completion_recorded_ = false;
    bool prepared_ = false;
};

} // namespace detail

using FdkBackprojector = detail::CylFdkBackprojectorImpl<false>;
using FdkMatchedBackprojector = detail::CylFdkBackprojectorImpl<true>;

} // namespace YK::CylFpBp
