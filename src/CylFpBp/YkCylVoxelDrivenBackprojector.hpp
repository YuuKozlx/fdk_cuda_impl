#pragma once

#include <cmath>
#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/kernels/YkCylFpBpLaunch.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp {

// 圆柱探测器体素驱动 V3 反投影。每个线程负责一个体素并遍历全部视图，
// 因而没有 atomicAdd；投影通过持久化 3D 纹理执行 u/v 双线性插值。
//
// 重要：该算法模仿平板 BP_Joseph_v3 的物理主轴长度权重，但不是
// MainAxisJoseph FP 的离散转置。它适用于允许近似 BP 的 SART/SIRT 类更新
// 和性能实验，不可用于要求严格伴随的 CGLS 或伴随误差验证。
class VoxelDrivenBackprojectorV3 {
public:
    ~VoxelDrivenBackprojectorV3() { release(); }
    VoxelDrivenBackprojectorV3() = default;
    VoxelDrivenBackprojectorV3(const VoxelDrivenBackprojectorV3&) = delete;
    VoxelDrivenBackprojectorV3& operator=(const VoxelDrivenBackprojectorV3&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || channels <= 0 || rows <= 0 || geometry.empty() ||
            volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0) return false;
        std::vector<detail::SVoxelDrivenView> packed;
        packed.reserve(geometry.size());
        for (const auto& view : geometry) {
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(view, channels, rows, frame)) return false;
            detail::SVoxelDrivenView item{};
            item.source = view.source;
            item.radial_unit = make_float4(frame.radialUnit.x,
                frame.radialUnit.y, frame.radialUnit.z, 0.f);
            item.tangent_unit = make_float4(frame.tangentUnit.x,
                frame.tangentUnit.y, frame.tangentUnit.z, 0.f);
            item.axis_unit = make_float4(frame.axisUnit.x,
                frame.axisUnit.y, frame.axisUnit.z, 0.f);
            item.cylinder_center = make_float4(frame.cylinderCenter.x,
                frame.cylinderCenter.y, frame.cylinderCenter.z, 0.f);
            item.radius_mm = frame.radius_mm;
            item.principal_u = frame.principalU;
            item.principal_v = frame.principalV;
            item.inv_channel_angle_step_rad = 1.f / frame.channelStepRad;
            item.inv_row_step_mm = 1.f / frame.rowStepMm;
            packed.push_back(item);
        }
        YK_CUDA_CHECK(cudaSetDevice(device_id));
        Mem::PodDataController pod;
        d_geometry_ = pod.allocateAndUpload(packed, device_id);
        projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels, rows, static_cast<int>(geometry.size()));
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
        detail::launch_voxel_driven_v3(projection_texture_.tex,
            d_geometry_.data(), volume, views_, volume_geometry_, stream_,
            accumulate);
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
    Mem::DeviceLinearBuffer<detail::SVoxelDrivenView> d_geometry_{};
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

} // namespace YK::CylFpBp
