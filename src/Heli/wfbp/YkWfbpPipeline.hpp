#pragma once

#include <algorithm>
#include <cmath>

#include "Heli/wfbp/YkWfbpProcessors.hpp"
#include "global/YkLog.h"

namespace YK { namespace Helical { namespace Wfbp {

// FreeCT_wFBP 主链：探测器适配（可选） -> fan-to-parallel 重排 ->
// FreeCT ramp 滤波 -> W(q) 螺旋加权反投。
//
// 这里保留工程统一的线性显存、stream 与 launch policy，但算法坐标、
// FFS 分支和归一化均对应 FreeCT_wFBP，而不是普通 FDK 的包装。
class Pipeline {
public:
    ~Pipeline() { release(); }
    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    bool prepare(const SHeliCTParam& params, const Config& config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(params, config, stream)) return false;
        params_ = params;
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;
        geometry_ = makeGeometry_(params, config);
        if (geometry_.views <= 0) {
            YK_LOGE("[wFBP] insufficient boundary views for FreeCT add_projections");
            release();
            return false;
        }

        if (config_.input_detector == EInputDetector::FlatPanel) {
            d_arc_projection_ = memory_.allocateDevice3D<float>(
                geometry_.input_channels, geometry_.input_rows,
                geometry_.raw_views, device_id_);
            if (!flat_to_arc_.prepare(geometry_, params_.du_mm, config_)) {
                release();
                return false;
            }
        }
        d_rebinned_ = memory_.allocateDevice3D<float>(geometry_.output_channels,
            geometry_.rows, geometry_.views, device_id_);
        d_filtered_ = memory_.allocateDevice3D<float>(geometry_.output_channels,
            geometry_.rows, geometry_.views, device_id_);
        if (!rebin_.prepare(geometry_, config_) ||
            !filter_.prepare(geometry_, config_, stream_, device_id_) ||
            !backproject_.prepare(params_, geometry_, config_)) {
            release();
            return false;
        }
        prepared_ = true;
        return true;
    }

    bool reconstruct(const float* projection, float* volume)
    {
        if (!prepared_ || !projection || !volume) return false;
        const float* arc_projection = projection;
        if (config_.input_detector == EInputDetector::FlatPanel) {
            if (!flat_to_arc_.apply(projection, d_arc_projection_.data(), stream_))
                return false;
            arc_projection = d_arc_projection_.data();
        }
        return rebin_.apply(arc_projection, d_rebinned_.data(), stream_) &&
            filter_.apply(d_rebinned_.data(), d_filtered_.data()) &&
            backproject_.apply(d_filtered_.data(), volume, stream_);
    }

    void release()
    {
        filter_.release();
        d_arc_projection_ = {};
        d_rebinned_ = {};
        d_filtered_ = {};
        params_ = {};
        geometry_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

    const Geometry& geometry() const { return geometry_; }
    const float* arcProjectionData() const
    {
        return config_.input_detector == EInputDetector::FlatPanel
            ? d_arc_projection_.data() : nullptr;
    }
    const float* rebinnedData() const { return d_rebinned_.data(); }
    const float* filteredData() const { return d_filtered_.data(); }

private:
    static int phiSpotCount_(EFocalSpotMode mode)
    {
        return mode == EFocalSpotMode::Phi || mode == EFocalSpotMode::PhiAndZ ? 2 : 1;
    }

    static int zSpotCount_(EFocalSpotMode mode)
    {
        return mode == EFocalSpotMode::Z || mode == EFocalSpotMode::PhiAndZ ? 2 : 1;
    }

    static bool validate_(const SHeliCTParam& p, const Config& c, cudaStream_t stream)
    {
        const int focal_spots = phiSpotCount_(c.focal_spot_mode) *
            zSpotCount_(c.focal_spot_mode);
        if (!stream || p.iPU < 2 || p.iPV < 2 || p.iVX <= 0 || p.iVY <= 0 ||
            p.iVZ <= 0 || p.SID <= 0.f || p.SDD <= p.SID || p.du_mm <= 0.f ||
            p.dv_mm <= 0.f || p.pitch_mm <= 0.f || p.views_per_rot < 4 ||
            c.channel_oversampling != 2 || c.redundancy_flat < 0.f ||
            c.redundancy_flat >= 1.f || c.filter.cutoff_c <= 0.f ||
            c.filter.cutoff_c > 1.f || c.filter.apodization_a < 0.f ||
            c.filter.apodization_a > 1.f ||
            p.views_per_rot % focal_spots != 0 ||
            (p.views_per_rot / focal_spots) % 2 != 0 ||
            p.angle_list.size() < static_cast<size_t>(p.views_per_rot) ||
            p.angle_list.size() % static_cast<size_t>(focal_spots) != 0) {
            YK_LOGE("[wFBP] invalid FreeCT configuration");
            return false;
        }
        if (c.input_detector == EInputDetector::EquiangularArc &&
            !(c.arc_channel_angle_step_rad > 0.f)) {
            YK_LOGE("[wFBP] arc input requires arc_channel_angle_step_rad");
            return false;
        }
        if (zSpotCount_(c.focal_spot_mode) == 2 &&
            !(c.anode_angle_rad > 0.f && c.anode_angle_rad < 0.5f * CUDA_PI)) {
            YK_LOGE("[wFBP] z-FFS requires a valid anode_angle_rad");
            return false;
        }
        if (fabsf(p.tiltu_angle_rad) > 1e-6f || fabsf(p.tiltv_angle_rad) > 1e-6f ||
            fabsf(p.tiltn_angle_rad) > 1e-6f) {
            YK_LOGE("[wFBP] FreeCT_wFBP does not define a general tilted-detector model");
            return false;
        }

        const float raw_step = p.angle_list[1] - p.angle_list[0];
        if (!(raw_step > 0.f)) return false;
        for (size_t i = 1; i < p.angle_list.size(); ++i) {
            const float current = p.angle_list[i] - p.angle_list[i - 1];
            if (fabsf(current - raw_step) > c.angle_tolerance * fabsf(raw_step)) {
                YK_LOGE("[wFBP] FreeCT requires uniformly spaced raw views");
                return false;
            }
        }
        const float expected = 2.f * CUDA_PI / p.views_per_rot;
        if (fabsf(raw_step - expected) > c.angle_tolerance * expected) {
            YK_LOGE("[wFBP] views_per_rot does not match raw angle_list");
            return false;
        }

        const float principal_u = c.input_detector == EInputDetector::EquiangularArc
            ? (c.arc_principal_channel >= 0.f ? c.arc_principal_channel
                                               : 0.5f * (p.iPU - 1))
            : 0.5f * (p.iPU - 1) - p.offsetU_mm / p.du_mm;
        const float principal_v = 0.5f * (p.iPV - 1) - p.offsetV_mm / p.dv_mm;
        if (!(principal_u > 0.f && principal_u < p.iPU - 1.f) ||
            !(principal_v > 0.f && principal_v < p.iPV - 1.f)) {
            YK_LOGE("[wFBP] principal channel/row must lie inside the detector");
            return false;
        }
        return true;
    }

    static Geometry makeGeometry_(const SHeliCTParam& p, const Config& c)
    {
        Geometry g{};
        g.input_channels = p.iPU;
        g.output_channels = 2 * p.iPU; // FreeCT 固定二倍通道过采样
        g.input_rows = p.iPV;
        g.raw_views = static_cast<int>(p.angle_list.size());
        g.phi_spot_count = phiSpotCount_(c.focal_spot_mode);
        g.z_spot_count = zSpotCount_(c.focal_spot_mode);
        g.focal_spot_count = g.phi_spot_count * g.z_spot_count;
        g.rows = p.iPV * g.z_spot_count;
        g.sequence_views = g.raw_views / g.focal_spot_count;
        g.raw_views_per_turn = p.views_per_rot;
        g.views_per_turn = p.views_per_rot / g.focal_spot_count;
        g.raw_angle_step = p.angle_list[1] - p.angle_list[0];
        g.angle_step = g.raw_angle_step * g.focal_spot_count;
        g.focal_spot_mode = c.focal_spot_mode;
        g.reverse_row_interleave = c.reverse_row_interleave ? 1 : 0;
        g.central_channel = c.input_detector == EInputDetector::EquiangularArc
            ? (c.arc_principal_channel >= 0.f ? c.arc_principal_channel
                                               : 0.5f * (p.iPU - 1))
            : 0.5f * (p.iPU - 1) - p.offsetU_mm / p.du_mm;
        g.parallel_center = 2.f * g.central_channel;

        if (c.input_detector == EInputDetector::EquiangularArc) {
            g.fan_angle_step = c.arc_channel_angle_step_rad;
        }
        else {
            // 平板只作为输入适配：选择能完整落在物理探测器两端内的均匀弧网格。
            const float left = atanf(g.central_channel * p.du_mm / p.SDD);
            const float right_span = p.iPU - 1.f - g.central_channel;
            const float right = atanf(right_span * p.du_mm / p.SDD);
            g.fan_angle_step = std::min(left / g.central_channel, right / right_span);
        }
        g.parallel_spacing = p.SID * g.fan_angle_step * 0.5f;

        // FreeCT setup.cu: 扇角重排在两端各多拉 add_projections 帧，重排后
        // reshape_out 再将其裁掉。这既避免边界纹理夹取，也保证 BP 的首帧
        // 对应真实可插值的平行束角度。
        g.add_projections = static_cast<int>(
            (g.fan_angle_step * g.input_channels * 0.5f) / g.angle_step) + 10;
        const int available_views = g.sequence_views - 2 * g.add_projections;
        const int half_turn_views = g.views_per_turn / 2;
        // FreeCT backproject.cu 以整数 n_half_turns 工作，尾部不足半圈的
        // 重排帧不会进入 BP。这里在分配前即裁齐，避免滤波无效尾段。
        g.views = available_views > 0
            ? (available_views / half_turn_views) * half_turn_views : 0;
        g.first_angle = p.angle_list.front() + g.add_projections * g.angle_step;

        g.raw_central_row = 0.5f * (p.iPV - 1) - p.offsetV_mm / p.dv_mm;
        if (g.z_spot_count == 2) {
            // FreeCT 将每个原始 row 展成两个相隔半行距的虚拟 row：原始
            // principal row 因而映射到 2*row+0.5。交换两焦点的奇偶顺序只
            // 改变数据归属，不能改变探测器主射线的位置。
            g.central_row = 2.f * g.raw_central_row + 0.5f;
        }
        else {
            g.central_row = g.raw_central_row;
        }
        // FreeCT 的 W(q) 使用对称 [-1,1] 探测器范围。V offset 后取主射线
        // 两侧共同可用的对称范围，避免在较宽一侧引入无对应冗余的射线。
        const float row_half_span = std::min(g.central_row,
            static_cast<float>(g.rows - 1) - g.central_row);
        const float virtual_dv = p.dv_mm / static_cast<float>(g.z_spot_count);
        g.cone_half_angle = atanf(row_half_span * virtual_dv / p.SDD);

        g.sid = p.SID;
        g.sdd = p.SDD;
        g.pitch = p.pitch_mm;
        g.start_z = p.start_z_mm;

        if (g.phi_spot_count == 2) {
            // FreeCT: da = SDD * SID * fan_increment / (4 * (SDD-SID)).
            g.phi_ffs_shift = p.SDD * p.SID * g.fan_angle_step /
                (4.f * (p.SDD - p.SID));
        }
        if (g.z_spot_count == 2) {
            const float collimated_slice = p.dv_mm * p.SID / p.SDD;
            // FreeCT: dr = SDD * slice_width /
            //               (4 * (SDD-SID) * tan(anode_angle)).
            g.z_ffs_shift = p.SDD * collimated_slice /
                (4.f * (p.SDD - p.SID) * tanf(c.anode_angle_rad));
        }
        return g;
    }

    SHeliCTParam params_{};
    Config config_{};
    Geometry geometry_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_arc_projection_{};
    Mem::DeviceLinearBuffer3D<float> d_rebinned_{};
    Mem::DeviceLinearBuffer3D<float> d_filtered_{};
    RebinProcessor rebin_{};
    FlatToEquiangularArcProcessor flat_to_arc_{};
    FilterProcessor filter_{};
    BackProjectProcessor backproject_{};
};

} } } // namespace YK::Helical::Wfbp
