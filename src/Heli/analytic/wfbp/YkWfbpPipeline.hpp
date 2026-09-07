#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "Heli/analytic/wfbp/YkWfbpProcessors.hpp"
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

    // 公共逐视图几何入口。wFBP 的重排仍由 FreeCT 负责，但角度、源点
    // 高度、SID/SDD 和探测器采样间隔全部从 geometry 派生，调用方不再
    // 维护第二份 angles/FreeCT 几何。该入口只接受规则螺旋；不规则校准
    // 几何会在 validate_ 中明确拒绝。
    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SConeProjGeomVec>& geometry,
        const Config& config, cudaStream_t stream, int device_id = 0)
    {
        InputGeometry input{};
        float unused_radius = 0.f;
        if (!derivePublicGeometry_(geometry, channels, rows, input,
                unused_radius)) return false;
        return prepare(input, volume, config, stream, device_id);
    }

    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        Config config, cudaStream_t stream, int device_id = 0)
    {
        InputGeometry input{};
        float radius = 0.f;
        if (!derivePublicGeometry_(geometry, channels, rows, input, radius))
            return false;
        config.input_detector = EInputDetector::CylindricalArc;
        config.arc_curvature_radius_mm = radius;
        return prepare(input, volume, config, stream, device_id);
    }

    bool prepare(const InputGeometry& input, const SVolGeom& volume,
        const Config& config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(input, volume, config, stream)) return false;
        input_ = input;
        volume_ = volume;
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;
        geometry_ = makeGeometry_(input, config);
        if (geometry_.views <= 0) {
            YK_LOGE("[wFBP] insufficient boundary views for FreeCT add_projections");
            release();
            return false;
        }

        if (config_.input_detector != EInputDetector::EquiangularArc) {
            d_arc_projection_ = memory_.allocateDevice3D<float>(
                geometry_.input_channels, geometry_.input_rows,
                geometry_.raw_views, device_id_);
        }
        if (config_.input_detector == EInputDetector::FlatPanel) {
            if (!flat_to_arc_.prepare(geometry_, input_.channel_spacing_mm, config_)) {
                release();
                return false;
            }
        }
        else if (config_.input_detector == EInputDetector::CylindricalArc) {
            if (!cylindrical_to_arc_.prepare(geometry_,
                config_.arc_curvature_radius_mm, input_.channel_spacing_mm, config_)) {
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
            !backproject_.prepare(volume_, geometry_, config_)) {
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
        else if (config_.input_detector == EInputDetector::CylindricalArc) {
            if (!cylindrical_to_arc_.apply(projection, d_arc_projection_.data(),
                stream_)) return false;
            arc_projection = d_arc_projection_.data();
        }
        return rebin_.apply(arc_projection, d_rebinned_.data(), stream_) &&
            filter_.apply(d_rebinned_.data(), d_filtered_.data()) &&
            backproject_.apply(d_filtered_.data(), volume, stream_);
    }

    void release()
    {
        // reconstruct() 有意只负责入队。释放或重新 prepare 会覆盖内部工作区，
        // 因此必须先等待绑定 stream，不能依赖 cudaFree 的隐式同步行为。
        if (stream_)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        filter_.release();
        d_arc_projection_ = {};
        d_rebinned_ = {};
        d_filtered_ = {};
        input_ = {};
        volume_ = {};
        geometry_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

    const Geometry& geometry() const { return geometry_; }
    const float* arcProjectionData() const
    {
        return config_.input_detector != EInputDetector::EquiangularArc
            ? d_arc_projection_.data() : nullptr;
    }
    const float* rebinnedData() const { return d_rebinned_.data(); }
    const float* filteredData() const { return d_filtered_.data(); }

private:
    template <typename GeometryView>
    static bool derivePublicGeometry_(const std::vector<GeometryView>& views,
        int channels, int rows, InputGeometry& out, float& curvature_radius)
    {
        if (views.size() < 4 || channels < 2 || rows < 2) return false;
        const auto finite = [](float v) { return std::isfinite(v); };
        out.channels = channels;
        out.rows = rows;
        out.trajectory.angles_rad.reserve(views.size());
        const auto xyz = [](const float4& v) { return make_float3(v.x, v.y, v.z); };
        const auto norm = [](float3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
        const auto sub = [](float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); };
        for (size_t i = 0; i < views.size(); ++i) {
            const auto& g = views[i];
            float4 source4{}, center4{}, u4{}, v4{};
            float angle = 0.f;
            if constexpr (std::is_same_v<GeometryView, SCylConeProjGeomVec>) {
                source4 = g.source; center4 = g.detectorCenter;
                u4 = g.detectorU; v4 = g.detectorV; angle = cylViewAngle(g);
                SCylProjectionFrame frame{};
                if (!deriveCylProjectionFrame(g, channels, rows, frame)) return false;
                center4 = make_float4(frame.detectorCenter.x, frame.detectorCenter.y,
                    frame.detectorCenter.z, 0.f);
            } else {
                source4 = g.src; center4 = g.detS;
                u4 = g.detU; v4 = g.detV; angle = g.angle.x;
                SProjectionFrame frame{};
                if (!deriveProjectionFrame(g, channels, rows, frame)) return false;
                center4 = make_float4(frame.detectorCenter.x, frame.detectorCenter.y,
                    frame.detectorCenter.z, 0.f);
            }
            if (!finite(angle) || !finite(source4.x) || !finite(center4.x)) return false;
            out.trajectory.angles_rad.push_back(angle);
            const float sid = norm(xyz(source4));
            const float sdd = norm(sub(xyz(center4), xyz(source4)));
            if (i == 0) {
                if (!(sid > 0.f && sdd > sid)) return false;
                out.trajectory.sid_mm = sid;
                out.trajectory.sdd_mm = sdd;
                out.trajectory.start_z_mm = source4.z;
                out.channel_spacing_mm = norm(xyz(u4));
                out.row_spacing_mm = norm(xyz(v4));
                if (!(out.channel_spacing_mm > 0.f && out.row_spacing_mm > 0.f)) return false;
                if constexpr (std::is_same_v<GeometryView, SCylConeProjGeomVec>)
                    curvature_radius = cylDetectorRadius(g);
            }
            else {
                if (std::fabs(sid - out.trajectory.sid_mm) > 1e-3f ||
                    std::fabs(sdd - out.trajectory.sdd_mm) > 1e-3f) return false;
            }
        }
        // 视图数量和角步长是逐视图 geometry 的唯一真源；views_per_turn
        // 仅作为 FreeCT 的规则采样元数据派生出来。
        const float step = out.trajectory.angles_rad[1] - out.trajectory.angles_rad[0];
        if (!(step > 0.f) || !std::isfinite(step)) return false;
        const int vpt = static_cast<int>(std::llround(2.0 * CUDA_PI / step));
        if (vpt < 4 || std::fabs(step - 2.f * CUDA_PI / vpt) > 1e-4f) return false;
        out.views_per_turn = vpt;
        const auto sourceZ = [](const GeometryView& g) {
            if constexpr (std::is_same_v<GeometryView, SCylConeProjGeomVec>)
                return g.source.z;
            else return g.src.z;
        };
        out.trajectory.pitch_mm_per_turn = (sourceZ(views.back()) -
            sourceZ(views.front())) * static_cast<float>(vpt) /
            static_cast<float>(views.size() - 1);
        out.detector_pose = {};
        return out.trajectory.pitch_mm_per_turn > 0.f;
    }

    static int phiSpotCount_(EFocalSpotMode mode)
    {
        return mode == EFocalSpotMode::Phi || mode == EFocalSpotMode::PhiAndZ ? 2 : 1;
    }

    static int zSpotCount_(EFocalSpotMode mode)
    {
        return mode == EFocalSpotMode::Z || mode == EFocalSpotMode::PhiAndZ ? 2 : 1;
    }

    static bool validate_(const InputGeometry& p, const SVolGeom& volume,
        const Config& c, cudaStream_t stream)
    {
        const int focal_spots = phiSpotCount_(c.focal_spot_mode) *
            zSpotCount_(c.focal_spot_mode);
        const auto& trajectory = p.trajectory;
        const auto& pose = p.detector_pose;
        if (!stream || p.channels < 2 || p.rows < 2 || volume.Nx <= 0 ||
            volume.Ny <= 0 || volume.Nz <= 0 || trajectory.sid_mm <= 0.f ||
            trajectory.sdd_mm <= trajectory.sid_mm ||
            p.channel_spacing_mm <= 0.f || p.row_spacing_mm <= 0.f ||
            trajectory.pitch_mm_per_turn <= 0.f || p.views_per_turn < 4 ||
            c.channel_oversampling != 2 || c.redundancy_flat < 0.f ||
            c.redundancy_flat >= 1.f || c.filter.cutoff_c <= 0.f ||
            c.filter.cutoff_c > 1.f || c.filter.apodization_a < 0.f ||
            c.filter.apodization_a > 1.f ||
            p.views_per_turn % focal_spots != 0 ||
            (p.views_per_turn / focal_spots) % 2 != 0 ||
            trajectory.angles_rad.size() < static_cast<size_t>(p.views_per_turn) ||
            trajectory.angles_rad.size() % static_cast<size_t>(focal_spots) != 0) {
            YK_LOGE("[wFBP] invalid FreeCT configuration");
            return false;
        }
        if (c.input_detector == EInputDetector::EquiangularArc &&
            !(c.arc_channel_angle_step_rad > 0.f)) {
            YK_LOGE("[wFBP] arc input requires arc_channel_angle_step_rad");
            return false;
        }
        if (c.input_detector == EInputDetector::CylindricalArc) {
            if (!(c.arc_curvature_radius_mm > 0.f)) {
                YK_LOGE("[wFBP] cylindrical arc input requires a positive curvature radius");
                return false;
            }
            if (c.focal_spot_mode != EFocalSpotMode::None) {
                YK_LOGE("[wFBP] non-concentric cylindrical arc input does not yet support FFS");
                return false;
            }
            const float principal = principalChannel_(p, c);
            const float radius = c.arc_curvature_radius_mm;
            const float center_distance = trajectory.sdd_mm - radius;
            const float left_alpha = -principal * p.channel_spacing_mm / radius;
            const float right_alpha = (p.channels - 1.f - principal) *
                p.channel_spacing_mm / radius;
            const float left_radial = center_distance + radius * cosf(left_alpha);
            const float right_radial = center_distance + radius * cosf(right_alpha);
            const float left_monotonic = radius + center_distance * cosf(left_alpha);
            const float right_monotonic = radius + center_distance * cosf(right_alpha);
            if (!(left_radial > 0.f && right_radial > 0.f &&
                left_monotonic > 0.f && right_monotonic > 0.f)) {
                YK_LOGE("[wFBP] cylindrical detector fan range is not a visible monotonic branch");
                return false;
            }
        }
        if (zSpotCount_(c.focal_spot_mode) == 2 &&
            !(c.anode_angle_rad > 0.f && c.anode_angle_rad < 0.5f * CUDA_PI)) {
            YK_LOGE("[wFBP] z-FFS requires a valid anode_angle_rad");
            return false;
        }
        if (fabsf(pose.tilt_u_rad) > 1e-6f ||
            fabsf(pose.tilt_v_rad) > 1e-6f ||
            fabsf(pose.tilt_n_rad) > 1e-6f) {
            YK_LOGE("[wFBP] FreeCT_wFBP does not define a general tilted-detector model");
            return false;
        }

        const float raw_step = trajectory.angles_rad[1] - trajectory.angles_rad[0];
        if (!(raw_step > 0.f)) return false;
        for (size_t i = 1; i < trajectory.angles_rad.size(); ++i) {
            const float current = trajectory.angles_rad[i] -
                trajectory.angles_rad[i - 1];
            if (fabsf(current - raw_step) > c.angle_tolerance * fabsf(raw_step)) {
                YK_LOGE("[wFBP] FreeCT requires uniformly spaced raw views");
                return false;
            }
        }
        const float expected = 2.f * CUDA_PI / p.views_per_turn;
        if (fabsf(raw_step - expected) > c.angle_tolerance * expected) {
            YK_LOGE("[wFBP] views_per_rot does not match raw angle_list");
            return false;
        }

        const float principal_u = principalChannel_(p, c);
        const float principal_v = 0.5f * (p.rows - 1) -
            pose.offset_unv_mm.z / p.row_spacing_mm;
        if (!(principal_u > 0.f && principal_u < p.channels - 1.f) ||
            !(principal_v > 0.f && principal_v < p.rows - 1.f)) {
            YK_LOGE("[wFBP] principal channel/row must lie inside the detector");
            return false;
        }
        return true;
    }

    static Geometry makeGeometry_(const InputGeometry& p, const Config& c)
    {
        Geometry g{};
        g.input_channels = p.channels;
        g.output_channels = 2 * p.channels; // FreeCT 固定二倍通道过采样
        g.input_rows = p.rows;
        g.raw_views = static_cast<int>(p.trajectory.angles_rad.size());
        g.phi_spot_count = phiSpotCount_(c.focal_spot_mode);
        g.z_spot_count = zSpotCount_(c.focal_spot_mode);
        g.focal_spot_count = g.phi_spot_count * g.z_spot_count;
        g.rows = p.rows * g.z_spot_count;
        g.sequence_views = g.raw_views / g.focal_spot_count;
        g.raw_views_per_turn = p.views_per_turn;
        g.views_per_turn = p.views_per_turn / g.focal_spot_count;
        g.raw_angle_step = p.trajectory.angles_rad[1] - p.trajectory.angles_rad[0];
        g.angle_step = g.raw_angle_step * g.focal_spot_count;
        g.focal_spot_mode = c.focal_spot_mode;
        g.reverse_row_interleave = c.reverse_row_interleave ? 1 : 0;
        g.central_channel = principalChannel_(p, c);
        g.parallel_center = 2.f * g.central_channel;

        if (c.input_detector == EInputDetector::EquiangularArc) {
            g.fan_angle_step = c.arc_channel_angle_step_rad;
        }
        else if (c.input_detector == EInputDetector::CylindricalArc) {
            // 将真实圆柱两端的射线方向换成焦点扇角，并选择左右两侧都
            // 完整落在原始探测器内的最大均匀扇角步长。
            const float radius = c.arc_curvature_radius_mm;
            const float center_distance = p.trajectory.sdd_mm - radius;
            const float left_alpha = -g.central_channel * p.channel_spacing_mm / radius;
            const float right_span = p.channels - 1.f - g.central_channel;
            const float right_alpha = right_span * p.channel_spacing_mm / radius;
            const float left_gamma = atan2f(radius * sinf(-left_alpha),
                center_distance + radius * cosf(left_alpha));
            const float right_gamma = atan2f(radius * sinf(right_alpha),
                center_distance + radius * cosf(right_alpha));
            g.fan_angle_step = std::min(left_gamma / g.central_channel,
                right_gamma / right_span);
        }
        else {
            // 平板只作为输入适配：选择能完整落在物理探测器两端内的均匀弧网格。
            const float left = atanf(g.central_channel * p.channel_spacing_mm / p.trajectory.sdd_mm);
            const float right_span = p.channels - 1.f - g.central_channel;
            const float right = atanf(right_span * p.channel_spacing_mm / p.trajectory.sdd_mm);
            g.fan_angle_step = std::min(left / g.central_channel, right / right_span);
        }
        g.parallel_spacing = p.trajectory.sid_mm * g.fan_angle_step * 0.5f;

        // FreeCT setup.cu: 扇角重排在两端各多拉 add_projections 帧，重排后
        // reshape_out 再将其裁掉。这既避免边界纹理夹取，也保证 BP 的首帧
        // 对应真实可插值的平行束角度。
        const int boundary_views = static_cast<int>(
            (g.fan_angle_step * g.input_channels * 0.5f) / g.angle_step) + 10;
        const int available_views = g.sequence_views - 2 * boundary_views;
        const int half_turn_views = g.views_per_turn / 2;
        // FreeCT backproject.cu 以整数 n_half_turns 工作。可用帧数不是半圈
        // 整数倍时，必须从两端近似对称地裁掉余量；若总是只裁尾部，有效
        // 螺旋轨迹会相对目标体积发生 Z 偏移，材料值也会被错误层面混合。
        g.views = available_views > 0
            ? (available_views / half_turn_views) * half_turn_views : 0;
        const int unused_views = available_views - g.views;
        g.add_projections = boundary_views + unused_views / 2;
        g.first_angle = p.trajectory.angles_rad.front() + g.add_projections * g.angle_step;

        g.raw_central_row = 0.5f * (p.rows - 1) - p.detector_pose.offset_unv_mm.z / p.row_spacing_mm;
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
        const float virtual_dv = p.row_spacing_mm / static_cast<float>(g.z_spot_count);
        g.cone_half_angle = atanf(row_half_span * virtual_dv / p.trajectory.sdd_mm);

        g.sid = p.trajectory.sid_mm;
        g.sdd = p.trajectory.sdd_mm;
        g.pitch = p.trajectory.pitch_mm_per_turn;
        g.start_z = p.trajectory.start_z_mm;

        if (g.phi_spot_count == 2) {
            // FreeCT: da = SDD * SID * fan_increment / (4 * (SDD-SID)).
            g.phi_ffs_shift = p.trajectory.sdd_mm * p.trajectory.sid_mm * g.fan_angle_step /
                (4.f * (p.trajectory.sdd_mm - p.trajectory.sid_mm));
        }
        if (g.z_spot_count == 2) {
            const float collimated_slice = p.row_spacing_mm * p.trajectory.sid_mm / p.trajectory.sdd_mm;
            // FreeCT: dr = SDD * slice_width /
            //               (4 * (SDD-SID) * tan(anode_angle)).
            g.z_ffs_shift = p.trajectory.sdd_mm * collimated_slice /
                (4.f * (p.trajectory.sdd_mm - p.trajectory.sid_mm) * tanf(c.anode_angle_rad));
        }
        return g;
    }

    static float principalChannel_(const InputGeometry& p, const Config& c)
    {
        if (c.arc_principal_channel >= 0.f &&
            c.input_detector != EInputDetector::FlatPanel)
            return c.arc_principal_channel;
        if (c.input_detector == EInputDetector::EquiangularArc)
            return 0.5f * (p.channels - 1);
        return 0.5f * (p.channels - 1) - p.detector_pose.offset_unv_mm.x / p.channel_spacing_mm;
    }

    InputGeometry input_{};
    SVolGeom volume_{};
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
    CylindricalToEquiangularArcProcessor cylindrical_to_arc_{};
    FilterProcessor filter_{};
    BackProjectProcessor backproject_{};
};

} } } // namespace YK::Helical::Wfbp
