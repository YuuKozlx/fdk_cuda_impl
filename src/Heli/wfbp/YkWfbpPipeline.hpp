#pragma once

#include <cmath>

#include "Heli/wfbp/YkWfbpProcessors.hpp"
#include "global/YkLog.h"

namespace YK { namespace Helical { namespace Wfbp {

// 无 FFS wFBP 主链路：扇束重排 -> 平行通道滤波 -> 螺旋加权 BP。
// 算法结构和核心坐标变换移植自 FreeCT_wFBP (GPL-2.0-or-later)。
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

        if (config_.input_detector == EInputDetector::FlatPanel) {
            d_arc_projection_ = memory_.allocateDevice3D<float>(geometry_.input_channels,
                geometry_.rows, geometry_.views, device_id_);
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
    static bool validate_(const SHeliCTParam& p, const Config& c, cudaStream_t stream)
    {
        if (!stream || p.iPU < 2 || p.iPV < 2 || p.iVX <= 0 || p.iVY <= 0 ||
            p.iVZ <= 0 || p.SID <= 0.f || p.SDD <= p.SID || p.du_mm <= 0.f ||
            p.dv_mm <= 0.f || p.pitch_mm <= 0.f || p.views_per_rot < 4 ||
            static_cast<int>(p.angle_list.size()) < p.views_per_rot ||
            (c.channel_oversampling != 1 && c.channel_oversampling != 2 &&
             c.channel_oversampling != 4) || c.redundancy_flat < 0.f ||
            c.redundancy_flat >= 1.f) {
            YK_LOGE("[wFBP] invalid configuration");
            return false;
        }
        if (c.input_detector == EInputDetector::EquiangularArc &&
            !(c.arc_channel_angle_step_rad > 0.f)) {
            YK_LOGE("[wFBP] equiangular arc input requires arc_channel_angle_step_rad");
            return false;
        }
        const float principal_u = c.input_detector == EInputDetector::EquiangularArc
            ? (c.arc_principal_channel >= 0.f ? c.arc_principal_channel
                                               : 0.5f * (p.iPU - 1))
            : 0.5f * (p.iPU - 1) - p.offsetU_mm / p.du_mm;
        if (!(principal_u > 0.f && principal_u < p.iPU - 1.f)) {
            YK_LOGE("[wFBP] principal U must lie inside the detector");
            return false;
        }
        if (fabsf(p.tiltu_angle_rad) > 1e-6f || fabsf(p.tiltv_angle_rad) > 1e-6f ||
            fabsf(p.tiltn_angle_rad) > 1e-6f || fabsf(p.offsetV_mm) > 1e-6f) {
            YK_LOGE("[wFBP] detector tilt and V offset are not supported by the FreeCT geometry path");
            return false;
        }
        const float step = p.angle_list[1] - p.angle_list[0];
        if (!(step > 0.f)) return false;
        for (size_t i = 1; i < p.angle_list.size(); ++i) {
            const float current = p.angle_list[i] - p.angle_list[i - 1];
            if (fabsf(current - step) > c.angle_tolerance * fabsf(step)) {
                YK_LOGE("[wFBP] angles must be uniformly spaced");
                return false;
            }
        }
        const float expected = 2.f * CUDA_PI / p.views_per_rot;
        if (fabsf(step - expected) > c.angle_tolerance * expected) {
            YK_LOGE("[wFBP] views_per_rot does not match angle_list");
            return false;
        }
        return true;
    }

    static Geometry makeGeometry_(const SHeliCTParam& p, const Config& c)
    {
        Geometry g{};
        g.input_channels = p.iPU;
        g.output_channels = p.iPU * c.channel_oversampling;
        g.rows = p.iPV;
        g.views = static_cast<int>(p.angle_list.size());
        g.views_per_turn = p.views_per_rot;
        g.first_angle = p.angle_list.front();
        g.angle_step = p.angle_list[1] - p.angle_list[0];
        g.central_channel = c.input_detector == EInputDetector::EquiangularArc
            ? (c.arc_principal_channel >= 0.f ? c.arc_principal_channel
                                               : 0.5f * (p.iPU - 1))
            : 0.5f * (p.iPU - 1) - p.offsetU_mm / p.du_mm;
        g.parallel_center = c.channel_oversampling * g.central_channel;
        if (c.input_detector == EInputDetector::EquiangularArc) {
            g.fan_angle_step = c.arc_channel_angle_step_rad;
        }
        else {
            // Use the largest uniform angular step whose virtual arc remains
            // completely inside both edges of the physical flat detector.
            const float left_angle = atanf(g.central_channel * p.du_mm / p.SDD);
            const float right_span = p.iPU - 1.f - g.central_channel;
            const float right_angle = atanf(right_span * p.du_mm / p.SDD);
            g.fan_angle_step = std::min(left_angle / g.central_channel,
                right_angle / right_span);
        }
        g.parallel_spacing = p.SID * g.fan_angle_step / c.channel_oversampling;
        const float half_height = 0.5f * (p.iPV - 1) * p.dv_mm;
        g.cone_half_angle = atanf(half_height / p.SDD);
        g.sid = p.SID;
        g.sdd = p.SDD;
        g.pitch = p.pitch_mm;
        g.start_z = p.start_z_mm;
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
