#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "CylFpBp/bp/YkCylFdkBackprojector.hpp"
#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/analytic/kernels/YkCylAnalyticFdkLaunch.cuh"
#include "FDK/YkFDKFilterProcessor.hpp"
#include "FDK/kernels/YkFDKFilterHelpers.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp {

// 源中心等角柱面 FDK 近似管线。
//
// 数据流：原始柱面投影 [view,row,channel]
//   -> 柱面余弦/锥角预加权（同时乘 dtheta）
//   -> 按弧长 t=R*gamma 的一维 ramp 滤波
//   -> 柱面 FDK 风格反投影。
// 这是独立的解析重建管线，负责预加权、弧长域滤波和最终反投影；
// 不复用 Flat-FDK，也不包含螺旋冗余权重。该底层管线只接受 R=SDD
// 的虚拟等角柱面；R!=SDD 的物理投影必须先经过 Analytic::ProjectionMapper。
class CylFdkPipeline {
public:
    ~CylFdkPipeline() { release(); }
    CylFdkPipeline() = default;
    CylFdkPipeline(const CylFdkPipeline&) = delete;
    CylFdkPipeline& operator=(const CylFdkPipeline&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const SFilterKernelDesc& filter = SFilterKernelDesc::RamLak(),
        cudaStream_t stream = nullptr, int device_id = 0)
    {
        release();
        if (!stream || channels <= 0 || rows <= 0 || geometry.size() < 2 ||
            volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0)
            return false;

        std::vector<detail::SCylFdkView> packed;
        std::vector<SFDKGeoParamPerView> filter_geometry;
        packed.reserve(geometry.size());
        filter_geometry.resize(geometry.size());
        float arc_step = 0.f;
        float channel_angle_step = 0.f;
        std::vector<float> unwrapped_angles;
        unwrapped_angles.reserve(geometry.size());
        constexpr float pi = 3.14159265358979323846f;
        constexpr float two_pi = 2.f * pi;
        for (const auto& view : geometry) {
            float angle = cylViewAngle(view);
            if (!std::isfinite(angle)) return false;
            if (!unwrapped_angles.empty()) {
                // 将角度展开到相邻视图的最短连续差，允许输入跨越 -pi/pi。
                float delta = angle - unwrapped_angles.back();
                while (delta <= -pi) delta += two_pi;
                while (delta > pi) delta -= two_pi;
                if (std::fabs(delta) <= 1e-8f) return false;
                angle = unwrapped_angles.back() + delta;
            }
            unwrapped_angles.push_back(angle);
        }
        const float direction = unwrapped_angles[1] - unwrapped_angles[0];
        const float direction_sign = direction >= 0.f ? 1.f : -1.f;
        for (size_t i = 2; i < unwrapped_angles.size(); ++i) {
            const float delta = unwrapped_angles[i] - unwrapped_angles[i - 1];
            if (delta * direction <= 0.f) return false;
        }
        // Parker 的扫描范围按“首个视图前半步 + 视图中心跨度 +
        // 最后一个视图后半步”定义。只使用 last-first 会让最后一帧
        // 的 beta 超出 scan_range，导致 fall 段被错误钳成零。
        const float first_step = std::fabs(unwrapped_angles[1] -
            unwrapped_angles[0]);
        const float last_step = std::fabs(unwrapped_angles.back() -
            unwrapped_angles[unwrapped_angles.size() - 2]);
        const float angular_coverage = std::fabs(unwrapped_angles.back() -
            unwrapped_angles.front()) + 0.5f * (first_step + last_step);
        // Cyl Parker 仅对规则扇角圆扫有定义。完整圆扫不加权；短扫必须
        // 覆盖 π+2Γ，否则边界数据没有完整冗余，继续执行会产生明显截断。
        if (angular_coverage > two_pi + 0.02f * two_pi)
            return false;
        const bool parker_enabled = angular_coverage < two_pi - 0.02f * two_pi;
        float fan_half_angle = 0.f;
        for (size_t i = 0; i < geometry.size(); ++i) {
            const auto& input = geometry[i];
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(input, channels, rows, frame))
                return false;
            const float3 source = make_float3(input.source.x, input.source.y,
                input.source.z);
            const float3 center_to_source = make_float3(
                frame.cylinderCenter.x - source.x,
                frame.cylinderCenter.y - source.y,
                frame.cylinderCenter.z - source.z);
            const float axial = center_to_source.x * frame.axisUnit.x +
                center_to_source.y * frame.axisUnit.y +
                center_to_source.z * frame.axisUnit.z;
            const float source_axis_sq = center_to_source.x * center_to_source.x +
                center_to_source.y * center_to_source.y +
                center_to_source.z * center_to_source.z - axial * axial;
            const float source_axis_distance = std::sqrt(std::max(0.f, source_axis_sq));
            const float source_axial = source.x * frame.axisUnit.x +
                source.y * frame.axisUnit.y + source.z * frame.axisUnit.z;
            const float3 source_to_center = make_float3(
                -source.x + source_axial * frame.axisUnit.x,
                -source.y + source_axial * frame.axisUnit.y,
                -source.z + source_axial * frame.axisUnit.z);
            const float sid = std::sqrt(source_to_center.x * source_to_center.x +
                source_to_center.y * source_to_center.y +
                source_to_center.z * source_to_center.z);
            const float tolerance = std::max(1e-3f, 1e-5f * frame.radius_mm);
            if (source_axis_distance > tolerance || sid <= 0.f) {
                // 一般曲率由上层 Analytic::Reconstruction 先 map 到 R=SDD。
                return false;
            }

            const float dtheta = i == 0
                ? std::fabs(unwrapped_angles[1] - unwrapped_angles[0])
                : (i + 1 == geometry.size()
                    ? std::fabs(unwrapped_angles[i] - unwrapped_angles[i - 1])
                    : 0.5f * std::fabs(unwrapped_angles[i + 1] - unwrapped_angles[i - 1]));
            if (!(dtheta > 1e-8f) || !std::isfinite(dtheta)) return false;
            if (i == 0) {
                arc_step = frame.radius_mm * frame.channelStepRad;
                channel_angle_step = frame.channelStepRad;
                fan_half_angle = std::max(frame.principalU,
                    static_cast<float>(channels - 1) - frame.principalU) *
                    frame.channelStepRad;
            }
            if (std::fabs(frame.radius_mm * frame.channelStepRad - arc_step) >
                std::max(1e-4f, 1e-4f * arc_step)) return false;

            detail::SCylFdkView item{};
            item.source = input.source;
            item.radial_unit = make_float4(frame.radialUnit.x, frame.radialUnit.y,
                frame.radialUnit.z, 0.f);
            item.tangent_unit = make_float4(frame.tangentUnit.x, frame.tangentUnit.y,
                frame.tangentUnit.z, 0.f);
            item.axis_unit = make_float4(frame.axisUnit.x, frame.axisUnit.y,
                frame.axisUnit.z, 0.f);
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
            item.detector_center_angle_rad = std::atan2(
                -(item.depth_unit.x * frame.tangentUnit.x +
                  item.depth_unit.y * frame.tangentUnit.y +
                  item.depth_unit.z * frame.tangentUnit.z),
                item.depth_unit.x * frame.radialUnit.x +
                item.depth_unit.y * frame.radialUnit.y +
                item.depth_unit.z * frame.radialUnit.z);
            item.detector_axial_offset_mm =
                (frame.detectorCenter.x - source.x) * frame.axisUnit.x +
                (frame.detectorCenter.y - source.y) * frame.axisUnit.y +
                (frame.detectorCenter.z - source.z) * frame.axisUnit.z;
            item.dtheta = dtheta;
            // Parker 分段只依赖“沿扫描方向的累计角”。反向采集时也
            // 映射到同一正向参数，避免把反向扫描的起止过渡区颠倒。
            item.parker_beta_rad = 0.5f * dtheta +
                direction_sign * (unwrapped_angles[i] - unwrapped_angles.front());
            item.parker_scan_range_rad = angular_coverage;
            item.parker_redundancy_half_rad =
                0.5f * (angular_coverage - static_cast<float>(CUDA_PI));
            item.parker_enabled = parker_enabled ? 1u : 0u;
            packed.push_back(item);

            // FilterProcessor 只需该视图的物理采样间隔和中心偏移。
            filter_geometry[i].du_mm = arc_step;
            filter_geometry[i].offsetU_pix = 0.f;
        }
        if (parker_enabled && angular_coverage + 1e-5f <
            static_cast<float>(CUDA_PI) + 2.f * fan_half_angle) {
            release();
            return false;
        }

        YK_CUDA_CHECK(cudaSetDevice(device_id));
        Mem::PodDataController pod;
        d_geometry_ = pod.allocateAndUpload(packed, device_id);
        d_preweighted_ = memory_.allocateDevice3D<float>(channels, rows,
            static_cast<int>(geometry.size()), device_id);
        d_filtered_ = memory_.allocateDevice3D<float>(channels, rows,
            static_cast<int>(geometry.size()), device_id);

        // 等角扇束的相邻射线距离是 R*sin(delta_gamma)，并非弧长
        // R*delta_gamma。用该距离修正离散 Ram-Lak 的非零抽样点；小扇角
        // 下修正趋近 1，从而连续退化为普通平板 ramp。
        if (filter.kind == EFilterKernel::None ||
            filter.kind == EFilterKernel::Custom ||
            filter.kind == EFilterKernel::Butterworth ||
            filter.kind == EFilterKernel::Kaiser ||
            filter.kind == EFilterKernel::Tukey) {
            // 自定义频谱/参数窗尚无柱面专用解析推导；None 也不能跳过
            // 柱面 ramp。明确拒绝，避免把 Flat 的频率核误用于角度域。
            release();
            return false;
        }
        const int padded = Fdk::detail::fp_computePaddedN(channels);
        const int half = padded / 2 - 1;
        std::vector<float> cylindrical_ramp(static_cast<size_t>(2 * half + 1), 0.f);
        cylindrical_ramp[half] = 0.25f;
        constexpr float pi_squared = 9.86960440108935861883f;
        for (int n = 1; n <= half; n += 2) {
            const float angular_distance = n * channel_angle_step;
            const float sine = std::sin(angular_distance);
            if (std::fabs(sine) <= 1e-7f) {
                release();
                return false;
            }
            const float ratio = angular_distance / sine;
            const float value = -ratio * ratio /
                (pi_squared * static_cast<float>(n * n));
            cylindrical_ramp[half - n] = value;
            cylindrical_ramp[half + n] = value;
        }
        SFilterKernelDesc cylindrical_filter = SFilterKernelDesc::SpatialRamp(
            std::move(cylindrical_ramp), filter.gain);
        // 保留柱面空域 ramp 的 source，同时把请求的有限带宽窗传给
        // SpatialRampFFT 分支；该分支现在会实际执行窗函数。
        cylindrical_filter.kind = filter.kind;
        cylindrical_filter.cutoff = filter.cutoff;

        FdkFilterConfig filter_config{};
        filter_config.dims = { channels, rows, static_cast<int>(geometry.size()) };
        filter_config.desc = std::move(cylindrical_filter);
        filter_config.stream = stream;
        if (!filter_.prepare(filter_config) ||
            !backprojector_.prepare(volume_geometry, channels, rows, geometry,
                stream, device_id)) {
            release();
            return false;
        }
        volume_geometry_ = volume_geometry;
        filter_geometry_ = std::move(filter_geometry);
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        stream_ = stream;
        device_id_ = device_id;
        prepared_ = true;
        return true;
    }

    bool reconstruct(const float* d_projection, float* d_volume,
        bool clear_output = true)
    {
        if (!prepared_ || !d_projection || !d_volume) return false;
        if (clear_output) {
            const size_t bytes = static_cast<size_t>(volume_geometry_.Nx) *
                volume_geometry_.Ny * volume_geometry_.Nz * sizeof(float);
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0, bytes, stream_));
        }
        detail::launch_cyl_fdk_preweight(d_projection, d_preweighted_.data(),
            d_geometry_.data(), channels_, rows_, views_, stream_);
        if (!filter_.apply(d_preweighted_.data(), d_filtered_.data(),
                { filter_geometry_.data(), views_ }, stream_)) return false;
        return backprojector_.uploadProjection(d_filtered_.data()) &&
            backprojector_.backprojectAnalytic(d_volume, !clear_output);
    }

    void release()
    {
        backprojector_.release();
        filter_.release();
        d_geometry_ = {};
        d_preweighted_ = {};
        d_filtered_ = {};
        filter_geometry_.clear();
        volume_geometry_ = {};
        channels_ = rows_ = views_ = 0;
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    bool prepared_ = false;
    int channels_ = 0, rows_ = 0, views_ = 0, device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    SVolGeom volume_geometry_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer<detail::SCylFdkView> d_geometry_{};
    Mem::DeviceLinearBuffer3D<float> d_preweighted_{};
    Mem::DeviceLinearBuffer3D<float> d_filtered_{};
    std::vector<SFDKGeoParamPerView> filter_geometry_{};
    Fdk::FilterProcessor filter_{};
    detail::CylFdkBackprojectorImpl<false> backprojector_{};
};

// 兼容旧调用方；新代码使用语义更明确的 CylFdkPipeline。
using FdkPipeline = CylFdkPipeline;

} // namespace YK::CylFpBp
