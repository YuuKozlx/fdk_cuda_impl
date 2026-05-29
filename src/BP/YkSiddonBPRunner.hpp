// YkConeBackprojector.hpp
#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"

#include "YkBPGpuContext_Siddon.hpp"
#include "BP/kernels/YkBPSiddonLaunch.cuh"
#include "BP/kernels/YkBpJosephLaunch.cuh"
#include "BP/kernels/YkBpFdkLaunch.cuh"

namespace YK {

    // ====================================================================
    // ConeBackprojector
    //
    // 支持的 ETask：
    //   BP_Siddon_RayDriven —— 射线驱动，严格伴随，慢
    //   BP_Siddon_VoxDriven —— 体素驱动，近似伴随，快
    //
    // 其他 ETask 直接报错返回 false。
    // ====================================================================
    class ConeBackprojector {
    public:
        bool isInitialized() const { return is_initialized_; }

        void release()
        {
            is_initialized_ = false;
            bp_type_ = ETask::BP_Siddon_RayDriven;
            YK_LOGD("[ConeBackprojector] release\n");
        }

        void reset()
        {
            YK_LOGD("[ConeBackprojector] reset\n");
        }

        bool init(const SCBCTParams& params,
            ETask task = ETask::BP_Siddon_RayDriven,
            int   deviceId = 0)
        {
            if (params.iPU <= 0 || params.iPV <= 0
                || params.iVX <= 0 || params.iVY <= 0 || params.iVZ <= 0)
            {
                YK_LOGE("[ConeBackprojector] init: invalid geometry params\n");
                return false;
            }
            if (task != ETask::BP_Siddon_RayDriven &&
                task != ETask::BP_Siddon_VoxDriven &&
                task != ETask::Bp_Joseph && task != ETask::BP_FDK && task != ETask::Bp_Joseph_v2 && task != ETask::Bp_Joseph_v3)
            {
                YK_LOGE("[ConeBackprojector] init: task %d is not a BP task\n",
                    static_cast<int>(task));
                return false;
            }
            bp_type_ = task;
            is_initialized_ = true;
            return true;
        }

        bool run(
            const float* d_sino,
            const SCBCTParams& params,
            float* d_vol_out,
            cudaStream_t       stream,
            bool               clear_vol = true,
            int                deviceId = 0,
            TaskDumpCallback   onDump = nullptr,
            void* userdata = nullptr)
        {
            if (!is_initialized_ || !d_sino || !d_vol_out) {
                YK_LOGE("[ConeBackprojector] run: not ready\n");
                return false;
            }
            if (params.iPAng <= 0
                || (int)params.angle_list.size() != params.iPAng)
            {
                YK_LOGE("[ConeBackprojector] run: invalid angle params\n");
                return false;
            }

            // ── 构建几何 ─────────────────────────────────────────────────
            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            SVolGeom vol_geom = SVolGeom::make_centered(
                params.iVX, params.iVY, params.iVZ,
                params.vox_x_mm, params.vox_z_mm);
            vol_geom.center = make_float3(
                params.vol_offset_x_mm,
                params.vol_offset_y_mm,
                params.vol_offset_z_mm);

            const int Na = params.iPAng;
            std::vector<SConeProjGeomVec> h_views(Na);
            std::vector<SFDKGeoParamPerView> h_gv(Na);
            build_circular_vec_geometry_from_theta(
                h_views, params.angle_list, Na,
                params.iPU, params.iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));



            GeoDerivedManagerVec{}.build_geo_params(
                params.iPU, params.iPV, params.scan_range_rad, h_views, h_gv);

            // ── GPU context（两种 BP 都不建纹理）────────────────────────
            Bp::BpSiddonGpuContext gpuctx;
            gpuctx.initNoTex(d_sino, vol_geom, h_views, h_gv, stream, deviceId);

            // ── 清零 ─────────────────────────────────────────────────────
            if (clear_vol) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0,
                    (size_t)params.iVX * params.iVY * params.iVZ * sizeof(float),
                    stream));
            }

            // ── dump 输入正弦图 ───────────────────────────────────────────
            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                const size_t view_elems = (size_t)params.iPU * params.iPV;
                for (int i = 0; i < Na; ++i) {
                    DumpPayload payload{
                        i, "sino",
                        static_cast<void*>(const_cast<float*>(d_sino) + i * view_elems),
                        view_elems, stream, userdata
                    };
                    onDump(&payload);
                }
            }

            // ── Dispatch ─────────────────────────────────────────────────
            switch (bp_type_)
            {
            case ETask::BP_Siddon_RayDriven:
                Bp::bp_siddon_launch(
                    gpuctx.d_sino_raw, d_vol_out,
                    gpuctx.geo.d_views_world(),
                    vol_geom,
                    params.iPU, params.iPV, Na,
                    stream);
                break;

            case ETask::BP_Siddon_VoxDriven: {
                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::bp_siddon_voxel_v2_launch(
                    sinoTex.tex, d_vol_out,
                    gpuctx.geo.d_views_world(),
                    vol_geom,
                    params.iPU, params.iPV, Na,
                    false, stream);
                break;
            }

            case ETask::Bp_Joseph:
            {

                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::joseph_bp_launch(
                    sinoTex.tex,
                    gpuctx.geo.h_views_world_vec(),
                    gpuctx.geo.d_views_vox(),   // 世界坐标
                    d_vol_out, vol_geom,
                    Na, params.iPU, params.iPV,
                    false, stream);
                break;
            }
            case ETask::BP_FDK:
            {

                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);

                Bp::fdk_bp_launch(sinoTex.tex, gpuctx.geo.d_views_world(),
                    gpuctx.geo.d_coeffs_data(), d_vol_out, vol_geom,
                    Na, false, stream);

                break;
            }
            case ETask::Bp_Joseph_v2: {
                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::joseph_bp_v2_launch(
                    sinoTex.tex,
                    gpuctx.geo.d_views_vox(),   // 世界坐标
                    d_vol_out, vol_geom,
                    Na, params.iPU, params.iPV,
                    false, stream);
                break;
            }
            case ETask::Bp_Joseph_v3: {
                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::joseph_bp_v3_launch(sinoTex.tex,
                    gpuctx.geo.d_views_world(),   // 世界坐标
                    gpuctx.geo.d_coeffs_data(),
                    d_vol_out, vol_geom,
                    Na,
                    false, stream);
            }

            default:
                return false;
            }

            // ── dump 体数据 ───────────────────────────────────────────────
            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                DumpPayload payload{
                    Na, "vol",
                    static_cast<void*>(d_vol_out),
                    (size_t)params.iVX * params.iVY * params.iVZ,
                    stream, userdata
                };
                onDump(&payload);
            }

            return true;
        }

    private:
        bool  is_initialized_ = false;
        ETask bp_type_ = ETask::BP_Siddon_RayDriven;
    };

    // ====================================================================
    // 便捷函数
    // ====================================================================
    YK_INLINE bool bp_backproject(
        const float* d_sino,
        float* d_vol_out,
        const SCBCTParams& params,
        ETask              task,
        cudaStream_t       stream,
        bool               clear_vol = true,
        int                deviceId = 0,
        TaskDumpCallback   onDump = nullptr,
        void* userdata = nullptr)
    {
        ConeBackprojector bp;
        if (!bp.init(params, task, deviceId)) return false;
        return bp.run(d_sino, params, d_vol_out, stream,
            clear_vol, deviceId, onDump, userdata);
    }




    // ====================================================================
// SiddonBpReconstructor  →  整合后用 ConeBackprojector
// SiddonBpReconstructorEx →  ConeBackprojectorEx（外部几何）
// ====================================================================
    class ConeBackprojectorEx {
    public:
        bool isInitialized() const { return is_initialized_; }
        int  totalReceived() const { return total_received_; }

        void reset() { total_received_ = 0; }
        void release() { reset(); is_initialized_ = false; }

        bool init(const SCBCTParams& params,
            ETask task = ETask::BP_Siddon_RayDriven,
            int   deviceId = 0)
        {
            if (params.iPU <= 0 || params.iPV <= 0
                || params.iVX <= 0 || params.iVY <= 0 || params.iVZ <= 0)
            {
                YK_LOGE("[ConeBackprojectorEx] init: invalid geometry params\n");
                return false;
            }
            if (task != ETask::BP_Siddon_RayDriven &&
                task != ETask::BP_Siddon_VoxDriven &&
                task != ETask::Bp_Joseph && task != ETask::BP_FDK && task != ETask::Bp_Joseph_v2 && task != ETask::Bp_Joseph_v3)
            {
                YK_LOGE("[ConeBackprojector] init: task %d is not a BP task\n",
                    static_cast<int>(task));
                return false;
            }
            bp_type_ = task;
            is_initialized_ = true;
            return true;
        }

        bool run(
            const float* d_sino,
            const SCBCTParams& params,
            const std::vector<SConeProjGeomVec>& h_views_ext,
            cudaStream_t                          stream,
            float* d_vol_out,
            bool                                  clear_vol = true,
            int                deviceId = 0,
            TaskDumpCallback                      onDump = nullptr,
            void* userdata = nullptr)
        {
            if (!is_initialized_ || !d_sino || !d_vol_out) {
                YK_LOGE("[ConeBackprojectorEx] run: not ready\n"); return false;
            }
            if (params.iPAng <= 0
                || (int)params.angle_list.size() != params.iPAng)
            {
                YK_LOGE("[ConeBackprojectorEx] run: invalid angle params\n"); return false;
            }
            if ((int)h_views_ext.size() != params.iPAng) {
                YK_LOGE("[ConeBackprojectorEx] run: h_views_ext size mismatch\n"); return false;
            }
            deviceId = 0;

            const int Na = params.iPAng;

            SVolGeom vol_geom = SVolGeom::make_centered(
                params.iVX, params.iVY, params.iVZ,
                params.vox_x_mm, params.vox_z_mm);
            vol_geom.center = make_float3(
                params.vol_offset_x_mm,
                params.vol_offset_y_mm,
                params.vol_offset_z_mm);


            std::vector<SFDKGeoParamPerView> h_gv(h_views_ext.size());

            GeoDerivedManagerVec{}.build_geo_params(
                params.iPU, params.iPV, params.scan_range_rad, h_views_ext, h_gv);


            Bp::BpSiddonGpuContext gpuctx;
            gpuctx.initNoTex(d_sino, vol_geom, h_views_ext, h_gv, stream, deviceId);

            if (clear_vol) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0,
                    (size_t)params.iVX * params.iVY * params.iVZ * sizeof(float),
                    stream));
            }

            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                const size_t view_elems = (size_t)params.iPU * params.iPV;
                for (int i = 0; i < Na; ++i) {
                    DumpPayload payload{
                        total_received_ + i, "sino",
                        static_cast<void*>(const_cast<float*>(d_sino) + i * view_elems),
                        view_elems, stream, userdata
                    };
                    onDump(&payload);
                }
            }

            // ── Dispatch ─────────────────────────────────────────────────
            switch (bp_type_)
            {
            case ETask::BP_Siddon_RayDriven:
                Bp::bp_siddon_launch(
                    gpuctx.d_sino_raw, d_vol_out,
                    gpuctx.geo.d_views_world(),
                    vol_geom,
                    params.iPU, params.iPV, Na,
                    stream);
                break;

            case ETask::BP_Siddon_VoxDriven: {
                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::bp_siddon_voxel_v2_launch(
                    sinoTex.tex, d_vol_out,
                    gpuctx.geo.d_views_world(),
                    vol_geom,
                    params.iPU, params.iPV, Na,
                    false, stream);
                break;
            }

            case ETask::Bp_Joseph:
            {

                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::joseph_bp_launch(
                    sinoTex.tex,
                    gpuctx.geo.h_views_world_vec(),
                    gpuctx.geo.d_views_vox(),   // 世界坐标
                    d_vol_out, vol_geom,
                    Na, params.iPU, params.iPV,
                    false, stream);
                break;
            }
            case ETask::BP_FDK:
            {

                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);

                Bp::fdk_bp_launch(sinoTex.tex, gpuctx.geo.d_views_world(),
                    gpuctx.geo.d_coeffs_data(), d_vol_out, vol_geom,
                    Na, false, stream);

                break;
            }
            case ETask::Bp_Joseph_v2: {
                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::joseph_bp_v2_launch(
                    sinoTex.tex,
                    gpuctx.geo.d_views_vox(),   // 世界坐标
                    d_vol_out, vol_geom,
                    Na, params.iPU, params.iPV,
                    false, stream);
                break;
            }
            case ETask::Bp_Joseph_v3: {
                auto sinoTex = Mem::TextureController::createTex3DFromDevice(
                    gpuctx.d_sino_raw,
                    params.iPU, params.iPV, Na);
                Bp::joseph_bp_v3_launch(sinoTex.tex,
                    gpuctx.geo.d_views_world(),   // 世界坐标
                    gpuctx.geo.d_coeffs_data(),
                    d_vol_out, vol_geom,
                    Na,
                    false, stream);
            }

            default:
                return false;
            }

            total_received_ += Na;

            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                DumpPayload payload{
                    total_received_, "vol",
                    static_cast<void*>(d_vol_out),
                    (size_t)params.iVX * params.iVY * params.iVZ,
                    stream, userdata
                };
                onDump(&payload);
            }

            return true;
        }

    private:
        bool  is_initialized_ = false;
        int   total_received_ = 0;
        ETask bp_type_ = ETask::BP_Siddon_RayDriven;
    };

    // 便捷函数
    YK_INLINE bool bp_backproject_ex(
        const float* d_sino,
        float* d_vol_out,
        const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& h_views_ext,
        ETask                                 task,
        cudaStream_t                          stream,
        bool                                  clear_vol = true,
        int                                   deviceId = 0,
        TaskDumpCallback                      onDump = nullptr,
        void* userdata = nullptr)
    {
        ConeBackprojectorEx bp;
        if (!bp.init(params, task)) return false;
        return bp.run(d_sino, params, h_views_ext, stream,
            d_vol_out, clear_vol, deviceId, onDump, userdata);
    }
} // namespace YK