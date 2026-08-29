#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "FDK/YkFDKFilterProcessor.hpp"
#include "FDK/kernels/YkCurveFilteredFdkLaunch.cuh"
#include "global/YkCBCTParams.h"
#include "global/YkCudaTextureController.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace YK::Fdk {

// Li 等人在 2011 年提出的 curve-filtered FDK（C-FDK）。它是平板圆轨迹
// FDK 的重排改进，不是柱面探测器 FDK。算法先将 P(beta,a,b) 重排成
// cone-parallel 数据 p(theta,t,c)，沿 t 滤波，再用论文式 (32)-(36) 反投影。
//
// 第一版有意只接受理想、等角、完整 2pi 圆扫描：论文重排依赖这些条件，
// 对任意逐视图校准几何做静默近似会比明确拒绝更危险。重排跨相邻角度取样，
// 因此当前只提供全量设备投影入口，不提供普通 FDK 的在线 chunk 接口。
class CurveFilteredFdkPipeline {
public:
    CurveFilteredFdkPipeline() = default;
    ~CurveFilteredFdkPipeline() { release(); }
    CurveFilteredFdkPipeline(const CurveFilteredFdkPipeline&) = delete;
    CurveFilteredFdkPipeline& operator=(const CurveFilteredFdkPipeline&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || params.scan.Nu < 2 || params.scan.Nv < 2 ||
            params.volume.Nx <= 0 || params.volume.Ny <= 0 ||
            params.volume.Nz <= 0 || geometry.size() < 3 ||
            static_cast<int>(geometry.size()) != params.scan.NAng) {
            YK_LOGE("[CurveFilteredFdkPipeline] 参数尺寸、stream 或 geometry 数量无效。");
            return false;
        }

        detail::SCurveFilteredFdkGeometry derived{};
        if (!deriveAndValidateGeometry_(params, geometry, derived)) return false;

        YK_CUDA_CHECK(cudaSetDevice(device_id));
        const int nt = params.scan.Nu;
        const int nc = params.scan.Nv;
        const int views = static_cast<int>(geometry.size());
        d_rebinned_ = memory_.allocateDevice3D<float>(nt, nc, views, device_id);
        d_filtered_ = memory_.allocateDevice3D<float>(nt, nc, views, device_id);
        input_texture_ = Mem::TextureController::createEmptyTex3D(
            params.scan.Nu, params.scan.Nv, views,
            cudaFilterModeLinear, cudaAddressModeBorder);
        filtered_texture_ = Mem::TextureController::createEmptyTex3D(
            nt, nc, views, cudaFilterModeLinear, cudaAddressModeBorder);

        // FilterProcessor 只需要每行的 t 采样间隔和零中心偏移；C-FDK 的
        // 其余几何全部由专用重排与反投影 kernel 使用。
        filter_geometry_.resize(views);
        for (auto& item : filter_geometry_) {
            item.du_mm = derived.dt_mm;
            item.inv_du_mm = 1.f / derived.dt_mm;
            item.offsetU_pix = 0.f;
        }
        FdkFilterConfig filter_config{};
        filter_config.dims = {nt, nc, views};
        filter_config.desc = params.reconstruction.filter;
        filter_config.stream = stream;
        if (!filter_.prepare(filter_config)) {
            release();
            return false;
        }

        volume_geometry_ = SVolGeom::make_centered(params.volume.Nx,
            params.volume.Ny, params.volume.Nz, params.volume.voxelX_mm,
            params.volume.voxelY_mm, params.volume.voxelZ_mm);
        volume_geometry_.center = make_float3(params.volume.centerX_mm,
            params.volume.centerY_mm, params.volume.centerZ_mm);
        geometry_ = derived;
        stream_ = stream;
        device_id_ = device_id;
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
        prepared_ = true;
        return true;
    }

    // d_projection 布局为 [view][v][u]，并且包含 prepare() 时给出的全部视图。
    // 返回 true 表示工作已提交到 stream；调用者读取输出前仍需同步该 stream。
    bool reconstruct(const float* d_projection, float* d_volume,
        bool clear_output = true)
    {
        if (!prepared_ || !d_projection || !d_volume) return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(input_texture_,
            d_projection, geometry_.input_u, geometry_.input_v,
            geometry_.views, stream_);
        detail::launchCurveFilteredFdkRebinPreweight(input_texture_.tex,
            d_rebinned_.data(), geometry_, launch_policy_, stream_);
        if (!filter_.apply(d_rebinned_.data(), d_filtered_.data(),
                {filter_geometry_.data(), geometry_.views}, stream_))
            return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(filtered_texture_,
            d_filtered_.data(), geometry_.output_t, geometry_.output_c,
            geometry_.views, stream_);
        detail::launchCurveFilteredFdkBackprojection(filtered_texture_.tex,
            d_volume, geometry_, volume_geometry_, !clear_output,
            launch_policy_, stream_);
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
        return true;
    }

    bool wait() const
    {
        if (!completion_recorded_) return true;
        return cudaEventSynchronize(completion_) == cudaSuccess;
    }

    void release()
    {
        if (completion_recorded_ && completion_) cudaEventSynchronize(completion_);
        filter_.release();
        input_texture_ = {};
        filtered_texture_ = {};
        d_rebinned_ = {};
        d_filtered_ = {};
        filter_geometry_.clear();
        if (completion_) cudaEventDestroy(completion_);
        completion_ = nullptr;
        completion_recorded_ = false;
        geometry_ = {};
        volume_geometry_ = {};
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }
    const detail::SCurveFilteredFdkGeometry& derivedGeometry() const
    { return geometry_; }

private:
    static float dot_(const float3& a, const float3& b)
    { return a.x * b.x + a.y * b.y + a.z * b.z; }
    static float length_(const float3& a) { return std::sqrt(dot_(a, a)); }
    static float3 sub_(const float3& a, const float3& b)
    { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
    static float3 xyz_(const float4& a) { return make_float3(a.x, a.y, a.z); }

    static bool deriveAndValidateGeometry_(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        detail::SCurveFilteredFdkGeometry& result)
    {
        constexpr float pi = 3.14159265358979323846f;
        constexpr float two_pi = 2.f * pi;
        const int nu = params.scan.Nu;
        const int nv = params.scan.Nv;
        const float center_u = 0.5f * static_cast<float>(nu - 1);
        const float center_v = 0.5f * static_cast<float>(nv - 1);
        std::vector<float> source_angles;
        source_angles.reserve(geometry.size());
        float sid_sum = 0.f, sdd_sum = 0.f, du_sum = 0.f, dv_sum = 0.f;

        for (size_t i = 0; i < geometry.size(); ++i) {
            const auto& view = geometry[i];
            const float3 source = xyz_(view.src);
            const float3 du = xyz_(view.detU);
            const float3 dv = xyz_(view.detV);
            const float3 detector_center = make_float3(
                view.detS.x + center_u * du.x + center_v * dv.x,
                view.detS.y + center_u * du.y + center_v * dv.y,
                view.detS.z + center_u * du.z + center_v * dv.z);
            const float sid = std::hypot(source.x, source.y);
            const float sdd = length_(sub_(detector_center, source));
            const float du_mm = length_(du);
            const float dv_mm = length_(dv);
            if (!(sid > 0.f && sdd > sid && du_mm > 0.f && dv_mm > 0.f))
                return geometryError_("存在退化的 SID/SDD 或探测器像素向量");

            const float scale = std::max({sid, sdd, 1.f});
            const float tolerance = 2e-4f * scale;
            const float3 expected_u = make_float3(-source.y / sid,
                source.x / sid, 0.f);
            const float3 expected_center = make_float3(
                -source.x * (sdd - sid) / sid,
                -source.y * (sdd - sid) / sid, 0.f);
            const float radial_u = (du.x * source.x + du.y * source.y) /
                (du_mm * sid);
            if (std::fabs(source.z) > tolerance ||
                length_(sub_(detector_center, expected_center)) > tolerance ||
                std::fabs(du.z) > 2e-4f * du_mm ||
                std::fabs(dv.x) > 2e-4f * dv_mm ||
                std::fabs(dv.y) > 2e-4f * dv_mm ||
                std::fabs(dot_(du, dv)) > 2e-4f * du_mm * dv_mm ||
                std::fabs(radial_u) > 2e-4f ||
                dot_(du, expected_u) < (1.f - 2e-4f) * du_mm ||
                dv.z < (1.f - 2e-4f) * dv_mm) {
                return geometryError_(
                    "论文 C-FDK 不支持源偏移、探测器 offset/tilt/skew 或非水平圆轨迹");
            }

            float angle = std::atan2(source.y, source.x);
            if (!source_angles.empty()) {
                float delta = angle - source_angles.back();
                while (delta <= -pi) delta += two_pi;
                while (delta > pi) delta -= two_pi;
                angle = source_angles.back() + delta;
            }
            source_angles.push_back(angle);
            sid_sum += sid;
            sdd_sum += sdd;
            du_sum += du_mm;
            dv_sum += dv_mm;
        }

        const float sid = sid_sum / geometry.size();
        const float sdd = sdd_sum / geometry.size();
        const float du_mm = du_sum / geometry.size();
        const float dv_mm = dv_sum / geometry.size();
        const float signed_step = (source_angles.back() - source_angles.front()) /
            static_cast<float>(geometry.size() - 1);
        if (std::fabs(signed_step) <= 1e-8f)
            return geometryError_("视图角度没有形成有效圆扫描");
        const float expected_step = two_pi / static_cast<float>(geometry.size());
        if (std::fabs(std::fabs(signed_step) - expected_step) > 2e-4f ||
            params.scan.short_scan)
            return geometryError_("当前仅支持不重复终点的等角完整 2pi 扫描");

        for (size_t i = 0; i < geometry.size(); ++i) {
            const float expected_angle = source_angles.front() + signed_step * i;
            const float3 source = xyz_(geometry[i].src);
            const float3 du = xyz_(geometry[i].detU);
            const float3 dv = xyz_(geometry[i].detV);
            const float center_sid = std::hypot(source.x, source.y);
            const float center_sdd = length_(sub_(make_float3(
                geometry[i].detS.x + center_u * du.x + center_v * dv.x,
                geometry[i].detS.y + center_u * du.y + center_v * dv.y,
                geometry[i].detS.z + center_u * du.z + center_v * dv.z), source));
            if (std::fabs(source_angles[i] - expected_angle) > 2e-4f ||
                std::fabs(center_sid - sid) > 2e-4f * sid ||
                std::fabs(center_sdd - sdd) > 2e-4f * sdd ||
                std::fabs(length_(du) - du_mm) > 2e-4f * du_mm ||
                std::fabs(length_(dv) - dv_mm) > 2e-4f * dv_mm)
                return geometryError_("轨迹半径、探测器距离、采样或角度步长不恒定");
        }

        // 论文为简化推导把探测器放在中心虚拟平面；实际设备允许任意 SDD。
        // 因而先把实际探测器半尺寸缩放到虚拟平面，重排采样时再映回实际平板。
        const float virtual_scale = sid / sdd;
        const float am = center_u * du_mm * virtual_scale;
        const float bm = center_v * dv_mm * virtual_scale;
        const float am2 = am * am;
        const float sid2 = sid * sid;
        if (am >= sid || bm <= 0.f)
            return geometryError_("探测器范围超出 C-FDK 重排定义域");
        const float q_edge = std::sqrt(sid2 - am2);
        const float t_max = sid * am / std::sqrt(sid2 + am2);

        result.input_u = nu;
        result.input_v = nv;
        result.views = static_cast<int>(geometry.size());
        result.output_t = nu;
        result.output_c = nv;
        result.sid_mm = sid;
        result.sdd_mm = sdd;
        result.input_du_mm = du_mm;
        result.input_dv_mm = dv_mm;
        result.beta0_rad = source_angles.front();
        result.signed_dtheta_rad = signed_step;
        result.dtheta_rad = std::fabs(signed_step);
        result.t_min_mm = -t_max;
        result.dt_mm = 2.f * t_max / static_cast<float>(nu - 1);
        result.c_min_mm = -bm;
        result.dc_mm = 2.f * bm / static_cast<float>(nv - 1);
        result.bm_mm = bm;
        result.c0_mm = bm * (2.f * (sid2 - am2) - sid * q_edge) / sid2;
        result.s0_mm = bm * (sid2 - am2) / sid2;
        if (!(result.c0_mm > 0.f && result.c0_mm <= result.s0_mm &&
            result.s0_mm <= result.bm_mm))
            return geometryError_("C-FDK 分段边界 c0/s0/bm 无效");
        return true;
    }

    static bool geometryError_(const char* reason)
    {
        YK_LOGE("[CurveFilteredFdkPipeline] 几何不满足论文适用条件：{}。", reason);
        return false;
    }

    bool prepared_ = false;
    bool completion_recorded_ = false;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t completion_ = nullptr;
    detail::SCurveFilteredFdkGeometry geometry_{};
    SVolGeom volume_geometry_{};
    SKernelLaunchPolicy launch_policy_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_rebinned_{};
    Mem::DeviceLinearBuffer3D<float> d_filtered_{};
    Mem::Tex3DHandle input_texture_{};
    Mem::Tex3DHandle filtered_texture_{};
    std::vector<SFDKGeoParamPerView> filter_geometry_{};
    FilterProcessor filter_{};
};

} // namespace YK::Fdk
