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
            param.offsetU_mm, 0.f, param.offsetV_mm);

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


    /**
 * @brief 为螺旋CT重建中的某一z层切片筛选有效投影视角集合。
 *
 * 算法分两步：
 *  1. 遍历全部投影几何，找到球管源点 src.z 最接近 z_center 的投影，
 *     取其旋转角 theta_center 作为该层的"中心角"。
 *  2. 以 [theta_center - view_half, theta_center + view_half] 为角度窗口，
 *     收集落在窗口内的全部投影索引和角度值。
 *
 * @note 角度比较为线性比较，要求 geo[i].angle.x 为单调递增的累积角度（rad），
 *       不得做模 2π 归一化，否则跨越 0 的切片会漏选投影。
 *
 * @note view_half 的理论下限为 π/2 + γ_max（γ_max 为探测器最大扇角），
 *       传入值若低于此下限将导致重建截断伪影。
 *
 * @param z_center   当前 slab 的 z 中心坐标（mm，世界坐标系）。
 * @param view_half  角度窗口的半宽（rad），对应短扫描所需的角度覆盖范围。
 * @param geo        全部投影的锥束几何参数列表，每个元素包含 src（源点坐标）
 *                   和 angle.x（累积旋转角，rad）。
 *
 * @return HelicalViewSelection  包含：
 *           - z_center      : 输入的 z 中心（透传）
 *           - theta_center  : 匹配到的中心旋转角（rad）
 *           - indices       : 落在角度窗口内的投影下标列表
 *           - angles        : 对应的旋转角列表（rad，与 indices 一一对应）
 */

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



    /**
 * @brief 将螺旋CT体积按 z 方向切分为一组重叠 slab，并为每个 slab 关联重建所需的投影视角。
 *
 * Slab 划分策略（滑动窗口）：
 * @code
 *   z_vol_start ──────────────────────────── z_vol_end
 *         [  slab0  ]
 *             [  slab1  ]        ← 步进 z_step_mm，相邻 slab 有重叠
 *                 [  slab2  ]
 * @endcode
 *
 *  - slab 宽度  = param.z_block_mm
 *  - slab 步进  = param.z_step_mm（通常 < z_block_mm，重叠区需外层调度加权融合）
 *  - 首个 slab 中心 = z_vol_start + z_block_mm / 2
 *  - 末尾判断加 1e-4f 容差，防止浮点误差漏掉最后一个 slab
 *
 * 每个 HelicalSlabConfig 记录：
 *  - z_center     : slab 中心的世界坐标（mm）
 *  - z_start_vox  : slab 底边在全局体素网格中的起始索引（z 轴）
 *  - z_count_vox  : slab 的体素层数
 *  - views        : 由 selectHelicalViews 筛选出的对应投影子集
 *
 * @note 若某 slab 在 geo 中找不到任何有效投影（views.indices 为空），
 *       该 slab 将被跳过并输出警告日志，不计入返回结果。
 *
 * @note 重叠 slab 之间的体素加权融合（避免 HU 叠加）由外层重建调度负责，
 *       本函数仅完成 slab 的几何划分与视角分配。
 *
 * @param param      螺旋CT扫描与重建参数，包含：
 *                     - iVZ           : 体积 z 方向体素总数
 *                     - vox_z_mm      : 体素 z 方向尺寸（mm）
 *                     - vol_offset_z_mm : 体积中心的 z 偏移（mm）
 *                     - z_block_mm    : 单个 slab 的 z 厚度（mm）
 *                     - z_step_mm     : slab 滑动步进（mm）
 * @param view_half  传递给 selectHelicalViews 的角度半窗宽（rad）。
 * @param geo        全部投影的锥束几何参数列表（同 selectHelicalViews）。
 *
 * @return std::vector<HelicalSlabConfig>  按 z 顺序排列的有效 slab 配置列表，
 *         空视角的 slab 已被过滤。列表长度由体积范围与步进共同决定。
 */
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

    //这两个函数是螺旋CT重建中分块（Slab）管理的核心调度逻辑。

    //    selectHelicalViews — 为某一z层选取投影视角
    //    输入
    //    参数含义z_center当前slab的z中心坐标（mm）view_half角度半窗宽（rad），决定选多少圈投影geo全部投影的几何参数列表
    //    逻辑
    //    第一步：找最近视角
    //    cppfor(int i = 0; i < geo.size(); ++i) {
    //    const float dz = fabs(geo[i].src.z - z_center);
    //    // 找源点z坐标最接近z_center的投影
    //}
    //在螺旋扫描中，球管边旋转边平移，每个投影的 src.z 各不相同。找到 src.z ≈ z_center 的那个视角，取其旋转角 theta_center。
    //    第二步：按角度窗口过滤
    //    cpptheta_lo = theta_center - view_half
    //    theta_hi = theta_center + view_half
    //    // 保留 [theta_lo, theta_hi] 范围内的所有投影
    //    这是 * *短扫描（Short - scan） * *的核心思想：围绕中心角度取 π + 扇角 左右的投影，够FDK重建用即可。

    //    buildHelicalSlabs — 把整个体积切成一批Slab
    //    Slab的几何意义
    //    z_vol_start ──────────────────────── z_vol_end
    //    [slab0]
    //    [slab1]          ← z_step_mm 步进，相邻slab有重叠
    //    [slab2]
    //    每个slab：

    //    宽度 = z_block_mm（重建z块厚度）
    //    步进 = z_step_mm（通常 < z_block_mm，即有overlap）
    //    中心 = z0，从 z_vol_start + z_block_mm / 2 开始

    //    循环逻辑
    //    cppfor(float z0 = z_vol_start + z_block_mm * 0.5f;
    //z0 <= z_vol_end - z_block_mm * 0.5f + 1e-4f;
    //z0 += z_step_mm)
    //    + 1e-4f 是浮点边界保护，防止最后一个slab因精度问题被漏掉。
    //    每个slab填写的字段
    //    cppcfg.z_center     // slab中心z（mm）
    //    cfg.z_start_vox  // 在全局体积中的起始voxel index（z轴）
    //    cfg.z_count_vox  // slab的voxel厚度
    //    cfg.views        // 调用selectHelicalViews拿到的投影集合
    //    z_start_vox 的计算：
    //    cpp(z0 - z_block_mm * 0.5f - z_vol_start) / vox_z_mm
    //    即 slab底边距体积原点的体素偏移。

    //    整体数据流
    //    SHeliCTParam（体积 / 螺旋参数）
    //    │
    //    ▼
    //    buildHelicalSlabs()
    //    │  按z_step_mm遍历所有slab中心z0
    //    │
    //    ├──► selectHelicalViews(z0, view_half, geo)
    //    │        找src.z最近的投影 → 取其theta_center
    //    │        按[theta_center±view_half] 过滤投影列表
    //    │        返回 indices + angles
    //    │
    //    └──► HelicalSlabConfig{ z_center, z_start_vox, z_count_vox, views }

    //返回: vector<HelicalSlabConfig>  →  逐slab送入FDK重建kernel

    //    潜在问题值得注意
    //    1. selectHelicalViews 按线性角度比较
    //    theta >= theta_lo && theta <= theta_hi 是线性比较，如果你的 angle.x 是累积角度（单调递增，不做模2π），这没问题。但如果角度做了归一化到[0, 2π)，跨越0°的slab会漏掉投影。需要确认 geo[i].angle.x 的存储约定。
    //    2. slab之间的overlap处理
    //    z_step_mm < z_block_mm 时相邻slab有重叠体素，最终合并时需要做加权融合（否则重叠区HU叠加）。当前代码只负责分配，融合逻辑应该在外层重建调度里。
    //    3. view_half 的选取
    //    理论上短扫描需要 π / 2 + γ_max（γ_max为最大扇角），view_half 如果传入固定值需要确保覆盖这个下限，否则重建会有truncation artifact

} // namespace YK