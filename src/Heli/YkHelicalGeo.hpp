#pragma once
#include <cmath>
#include <limits>
#include <vector>
#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "util/YkVecOperation.hpp"
#include "Heli/YkHeliCTParams.h"  // SHeliCTParam 定义在这里


namespace YK {

    // ================================================================
    // HelicalViewSelection
    // ================================================================
    struct HelicalViewSelection {
        std::vector<int>   indices;
        std::vector<float> angles;
        float              z_center = 0.f;
        float              theta_center = 0.f;
    };

    // ================================================================
    // HelicalSlabConfig
    // ================================================================
    struct HelicalSlabConfig {
        float                z_center = 0.f;
        int                  z_start_vox = 0;
        int                  z_count_vox = 0;
        HelicalViewSelection views;
    };

    // ----------------------------------------------------------------
    // build_helical_vec_geometry
    // 从 SHeliCTParam 构建螺旋几何，angle_list 外部传入
    // ----------------------------------------------------------------
    YK_INLINE void build_helical_vec_geometry(
        std::vector<SConeProjGeomVec>& geo,
        const SHeliCTParam& param)
    {
        const auto& theta = param.angle_list;
        const int   Ang = (int)theta.size();

        geo.resize(Ang);

        auto deg2rad = [](float deg) { return deg * CUDA_PI / 180.f; };
        auto rad2deg = [](float rad) { return rad * 180.f / CUDA_PI; };

        const float tiltu_deg = rad2deg(param.tiltu_angle_rad);
        const float tiltv_deg = rad2deg(param.tiltv_angle_rad);
        const float tiltn_deg = rad2deg(param.tiltn_angle_rad);

        const float IDD = param.SDD - param.SID;

        // ---- 探测器 U/V 方向 ----
        float3 detU_dir = make_float3(1.f, 0.f, 0.f);
        float3 detV_dir = make_float3(0.f, 0.f, 1.f);
        {
            const float3 axisU = detU_dir;
            detU_dir = f3_rot_axis(detU_dir, axisU, deg2rad(tiltu_deg));
            detV_dir = f3_rot_axis(detV_dir, axisU, deg2rad(tiltu_deg));

            const float3 axisV = detV_dir;
            detU_dir = f3_rot_axis(detU_dir, axisV, deg2rad(tiltv_deg));
            detV_dir = f3_rot_axis(detV_dir, axisV, deg2rad(tiltv_deg));

            const float3 normal = f3_normalize(cross(detU_dir, detV_dir));
            detU_dir = f3_rot_axis(detU_dir, normal, deg2rad(tiltn_deg));
            detV_dir = f3_rot_axis(detV_dir, normal, deg2rad(tiltn_deg));
        }

        // ---- 主射线方向 ----
        float3 srcCR_dir = f3_normalize(make_float3(0.f, 1.f, 0.f));

        const float3 src0 = make_float3(0.f, -param.SID, 0.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);
        const float  cu = 0.5f * (param.iPU - 1);
        const float  cv = 0.5f * (param.iPV - 1);

        // det_offset 包含 U/V 偏移
        const float3 det_offset = make_float3(
            param.offsetU_mm, param.offsetV_mm, 0.f);

        for (int a = 0; a < Ang; ++a) {
            const float  t = theta[a];
            const float  src_z = param.start_z_mm
                + param.pitch_mm * t / (2.f * CUDA_PI);
            const float3 helical_z = make_float3(0.f, 0.f, src_z);

            float3 src = f3_rotz(src0, t) + helical_z;
            float3 detC = f3_rotz(detC0 + det_offset, t) + helical_z;
            float3 srcCR = f3_rotz(srcCR_dir, t);
            float3 U = f3_rotz(detU_dir, t) * param.du_mm;
            float3 V = f3_rotz(detV_dir, t) * param.dv_mm;
            float3 detS = detC - U * cu - V * cv;

            geo[a] = SConeProjGeomVec{
                f3_to_f4(src),
                f3_to_f4(srcCR),
                f3_to_f4(detS),
                f3_to_f4(U),
                f3_to_f4(V),
                make_float4(t, 0.f, 0.f, 0.f)
            };
        }
    }

    // ----------------------------------------------------------------
    // selectHelicalViews
    // ----------------------------------------------------------------
    YK_INLINE HelicalViewSelection selectHelicalViews(
        float z_center,
        float view_half,
        const std::vector<SConeProjGeomVec>& geo)
    {
        int   center_idx = 0;
        float min_dz = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)geo.size(); ++i) {
            const float dz = std::fabs(geo[i].src.z - z_center);
            if (dz < min_dz) {
                min_dz = dz;
                center_idx = i;
            }
        }

        const float theta_center = geo[center_idx].angle.x;
        const float theta_lo = theta_center - view_half;
        const float theta_hi = theta_center + view_half;

        HelicalViewSelection sel{};
        sel.z_center = z_center;
        sel.theta_center = theta_center;

        for (int i = 0; i < (int)geo.size(); ++i) {
            const float theta = geo[i].angle.x;
            if (theta >= theta_lo && theta <= theta_hi) {
                sel.indices.push_back(i);
                sel.angles.push_back(theta);
            }
        }

        return sel;
    }

    // ----------------------------------------------------------------
    // buildHelicalSlabs
    // ----------------------------------------------------------------
    YK_INLINE std::vector<HelicalSlabConfig> buildHelicalSlabs(
        const SHeliCTParam& param,
        float                                view_half,
        const std::vector<SConeProjGeomVec>& geo)
    {
        const float z_vol_half =
            param.iVZ * param.vox_z_mm * 0.5f;
        const float z_vol_start =
            param.vol_offset_z_mm - z_vol_half;
        const float z_vol_end =
            param.vol_offset_z_mm + z_vol_half;

        std::vector<HelicalSlabConfig> slabs;

        for (float z0 = z_vol_start + param.z_block_mm * 0.5f;
            z0 <= z_vol_end - param.z_block_mm * 0.5f + 1e-4f;
            z0 += param.z_step_mm)
        {
            HelicalSlabConfig cfg{};
            cfg.z_center = z0;
            cfg.z_start_vox = (int)roundf(
                (z0 - param.z_block_mm * 0.5f - z_vol_start)
                / param.vox_z_mm);
            cfg.z_count_vox = (int)roundf(
                param.z_block_mm / param.vox_z_mm);
            cfg.views = selectHelicalViews(z0, view_half, geo);

            if (cfg.views.indices.empty()) {
                YK_LOGW("[helical] no views for z0={:.2f}, skip", z0);
                continue;
            }

            const float angle_range =
                cfg.views.angles.back() - cfg.views.angles.front();
            YK_LOGI("[helical slab] z0={:.2f}mm  views={}  "
                "angle=[{:.1f},{:.1f}]deg  range={:.1f}deg",
                z0,
                (int)cfg.views.indices.size(),
                cfg.views.angles.front() * 180.f / CUDA_PI,
                cfg.views.angles.back() * 180.f / CUDA_PI,
                angle_range * 180.f / CUDA_PI);

            slabs.push_back(cfg);
        }

        YK_LOGI("[helical] total {} slabs", (int)slabs.size());
        return slabs;
    }

} // namespace YK