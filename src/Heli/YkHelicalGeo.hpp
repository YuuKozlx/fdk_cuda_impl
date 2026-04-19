#pragma once
#include <cmath>
#include <limits>
#include <vector>
#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "util/YkVecOperation.hpp"

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
    // build_helical_vec_geometry_from_theta
    // ----------------------------------------------------------------
    YK_INLINE void build_helical_vec_geometry_from_theta(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& theta,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float pitch_mm,
        float start_z_mm = 0.f,
        float3 det_offset = make_float3(0.f, 0.f, 0.f),
        float3 detTilt_deg = make_float3(0.f, 0.f, 0.f),
        float3 src_offset = make_float3(0.f, 0.f, 0.f),
        float3 srcCRTilt_deg = make_float3(0.f, 0.f, 0.f))
    {
        geo.resize(Ang);

        auto deg2rad = [](float deg) { return deg * CUDA_PI / 180.f; };

        const float tiltu_deg = detTilt_deg.x;
        const float tiltv_deg = detTilt_deg.z;
        const float tiltn_deg = detTilt_deg.y;

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

        float3 srcCR_dir = make_float3(0.f, 1.f, 0.f);
        {
            srcCR_dir = f3_rot_axis(srcCR_dir, make_float3(1.f, 0.f, 0.f), deg2rad(srcCRTilt_deg.x));
            srcCR_dir = f3_rot_axis(srcCR_dir, make_float3(0.f, 0.f, 1.f), deg2rad(srcCRTilt_deg.y));
            srcCR_dir = f3_rot_axis(srcCR_dir, f3_normalize(srcCR_dir), deg2rad(srcCRTilt_deg.z));
            srcCR_dir = f3_normalize(srcCR_dir);
        }

        const float3 src0 = make_float3(0.f, -SID, 0.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);
        const float  cu = 0.5f * (Nu - 1);
        const float  cv = 0.5f * (Nv - 1);

        for (int a = 0; a < Ang; ++a) {
            const float t = theta[a];
            const float src_z = start_z_mm
                + pitch_mm * t / (2.f * CUDA_PI);
            const float3 helical_z = make_float3(0.f, 0.f, src_z);

            float3 src = f3_rotz(src0 + src_offset, t) + helical_z;
            float3 detC = f3_rotz(detC0 + det_offset, t) + helical_z;
            float3 srcCR = f3_rotz(srcCR_dir, t);
            float3 U = f3_rotz(detU_dir, t) * du;
            float3 V = f3_rotz(detV_dir, t) * dv;
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
    // selectHelicalViews：从 geo 直接读 src.z 和 angle.x
    // 不依赖 pitch/start_z，支持真实扫描数据
    // ----------------------------------------------------------------
    YK_INLINE HelicalViewSelection selectHelicalViews(
        float z_center,
        float view_half,
        const std::vector<SConeProjGeomVec>& geo)
    {
        // 找离 z_center 最近的投影作为角度中心
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
        float z_vol_start, float z_vol_end,
        float z_step_mm, float z_block_mm,
        float vox_z_mm,
        float view_half,   // 投影选取的角度半宽，与 Parker 无关
        const std::vector<SConeProjGeomVec>& geo)
    {
        std::vector<HelicalSlabConfig> slabs;

        for (float z0 = z_vol_start + z_block_mm * 0.5f;
            z0 <= z_vol_end - z_block_mm * 0.5f + 1e-4f;
            z0 += z_step_mm)
        {
            HelicalSlabConfig cfg{};
            cfg.z_center = z0;
            cfg.z_start_vox = (int)roundf(
                (z0 - z_block_mm * 0.5f - z_vol_start) / vox_z_mm);
            cfg.z_count_vox = (int)roundf(z_block_mm / vox_z_mm);
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