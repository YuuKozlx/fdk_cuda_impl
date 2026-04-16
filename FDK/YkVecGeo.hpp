#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include <fmt/format.h>
#include <stdexcept>
#include <string>
#include <vector_functions.hpp>
#include <vector_types.h>
#include "../global/YkGlobals.h"
#include "../global/YkLog.h"
#include "../global/YkMacro.hpp"
#include "../util/YkVecOperation.hpp"

namespace YK {

    /**
     * @brief Circular cone-beam vector geometry (vector form, Euler-enabled)
     *
     * World coordinate system (right-handed):
     *   X: right
     *   Y: forward (initial source-to-detector direction)
     *   Z: up
     *
     * Gantry rotation:
     *   theta increases CCW around +Z (right-hand rule)
     *
     * ----------------------------------------------------------------------------
     * Base geometry at theta = 0 (before any rotation / offset)
     * ----------------------------------------------------------------------------
     *   src0   = (0, -SID, 0)        // source position
     *   detC0  = (0,  IDD, 0)        // detector center
     *
     *   detU0  = (du, 0, 0)          // detector u-axis (pixel step)
     *   detV0  = (0, 0, dv)          // detector v-axis (pixel step)
     *
     *   srcCR0 = (0, 1, 0)           // center ray direction (source → detector)
     *
     * ----------------------------------------------------------------------------
     * Geometry construction per view (theta = theta[a])
     * ----------------------------------------------------------------------------
     * 1) Source position:
     *      src = src0 + src_offset
     *
     * 2) Detector center:
     *      detC = Rz(theta) * (detC0 + det_offset)
     *
     * 3) Center ray direction:
     *      srcCR is defined as:
     *
     *        a) base direction srcCR0
     *        b) apply source Euler rotation (pitch=X, yaw=Y, roll=Z)
     *        c) apply gantry rotation Rz(theta)
     *
     *      i.e.:
     *        srcCR = Rz(theta) * R_euler(srcEuler) * srcCR0
     *
     *      NOTE:
     *        Optionally, srcCR can also be derived from geometry:
     *          srcCR = normalize(detC - src)
     *        depending on whether explicit angular control or geometric consistency is preferred.
     *
     * ----------------------------------------------------------------------------
     * 4) Detector local axes (U, V)
     * ----------------------------------------------------------------------------
     *   Step 1: apply gantry rotation
     *      U = Rz(theta) * detU0
     *      V = Rz(theta) * detV0
     *
     *   Step 2: apply detector tilt (Euler angles)
     *
     *      detTiltEuler = (phi_x, phi_y, phi_z)
     *
     *      Rotation order: X → Y → Z (extrinsic, world axes)
     *
     *      U, V are rotated consistently:
     *        U = Rz * Ry * Rx * U
     *        V = Rz * Ry * Rx * V
     *
     * ----------------------------------------------------------------------------
     * 5) Detector pixel origin (top-left pixel center)
     * ----------------------------------------------------------------------------
     *   Let:
     *      cu = (Nu - 1) / 2
     *      cv = (Nv - 1) / 2
     *
     *   Then:
     *      detS = detC
     *           - cu * U
     *           - cv * V
     *
     * ----------------------------------------------------------------------------
     * Detector pixel offset semantics (IMPORTANT)
     * ----------------------------------------------------------------------------
     *   Pixel offsets (if used) represent ONLY in-plane detector shifts
     *   along the FINAL (already rotated + tilted) U/V axes.
     *
     *   They DO NOT represent:
     *     - source position correction
     *     - principal point / projection offset
     *     - calibration parameters
     *
     *   If enabled:
     *
     *     detS = detC
     *          - (cu + offsetU_pix) * U
     *          - (cv + offsetV_pix) * V
     *
     *   Offsets MUST be applied AFTER:
     *     - gantry rotation
     *     - detector tilt
     *
     * ----------------------------------------------------------------------------
     * Notes
     * ----------------------------------------------------------------------------
     * - All rotations are right-handed
     * - Euler angles are in radians
     * - Gantry rotation is always about global Z axis
     * - Detector tilt is applied after gantry rotation
     * - Source Euler rotation is applied before gantry rotation
     *
     * ----------------------------------------------------------------------------
     * Stored parameters
     * ----------------------------------------------------------------------------
     *   geo[a].src   = source position
     *   geo[a].srcCR = center ray direction (unit vector)
     *   geo[a].detS  = detector pixel (0,0) position
     *   geo[a].U     = detector u step vector
     *   geo[a].V     = detector v step vector
     *   geo[a].ang   = (theta, 0, 0)
     */


    YK_INLINE void build_circular_vec_geometry_from_theta(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& theta,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float3 det_offset = make_float3(0.f, 0.f, 0.f),
        float3 detTilt_deg = make_float3(0.f, 0.f, 0.f),  // outOfPlane(x), lateral(y), inPlane(z)
        float3 src_offset = make_float3(0.f, 0.f, 0.f),
        float3 srcCRTilt_deg = make_float3(0.f, 0.f, 0.f)   // pitch(x), yaw(y), roll(z) 单位:度
    )
    {
        geo.resize(Ang);

        auto deg2rad = [](float deg) { return deg * CUDA_PI / 180.f; };

        float tiltu_deg = detTilt_deg.x;
        float tiltv_deg = detTilt_deg.z;
        float tiltn_deg = detTilt_deg.y;

        // ---- 探测器 U/V 方向向量（局部系） ----
        float3 detU_dir = make_float3(1.f, 0.f, 0.f);
        float3 detV_dir = make_float3(0.f, 0.f, 1.f);
        {
            const float3 axisU = detU_dir;
            detU_dir = f3_rot_axis(detU_dir, axisU, deg2rad(tiltu_deg));
            detV_dir = f3_rot_axis(detV_dir, axisU, deg2rad(tiltu_deg));

            const float3 axisV = detV_dir;
            detU_dir = f3_rot_axis(detU_dir, axisV, deg2rad(tiltv_deg));
            detV_dir = f3_rot_axis(detV_dir, axisV, deg2rad(tiltv_deg));

            const float3 normal = f3_normalize(f3_cross(detU_dir, detV_dir));
            detU_dir = f3_rot_axis(detU_dir, normal, deg2rad(tiltn_deg));
            detV_dir = f3_rot_axis(detV_dir, normal, deg2rad(tiltn_deg));
        }

        // ---- 主射线方向（局部系） ----
        // 初始主射线沿 +Y，局部系轴：X=右, Y=前, Z=上
        float3 srcCR_dir = make_float3(0.f, 1.f, 0.f);
        {
            // pitch：绕局部 X 轴（上下俯仰）
            srcCR_dir = f3_rot_axis(srcCR_dir,
                make_float3(1.f, 0.f, 0.f), deg2rad(srcCRTilt_deg.x));

            // yaw：绕局部 Z 轴（左右偏转）
            srcCR_dir = f3_rot_axis(srcCR_dir,
                make_float3(0.f, 0.f, 1.f), deg2rad(srcCRTilt_deg.y));

            // roll：绕射线自身方向（面内旋转，通常为0）
            srcCR_dir = f3_rot_axis(srcCR_dir,
                f3_normalize(srcCR_dir), deg2rad(srcCRTilt_deg.z));

            srcCR_dir = f3_normalize(srcCR_dir);
        }

        const float3 src0 = make_float3(0.f, -SID, 0.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);

        for (int a = 0; a < Ang; ++a)
        {
            const float t = theta[a];

            // 1) 源位置
            float3 src = f3_rotz_p(f3_translate_point(src0, src_offset), t);

            // 2) 探测器中心
            float3 detC = f3_rotz_p(f3_translate_point(detC0, det_offset), t);

            // 3) 中心射线：局部系方向随机架旋转
            float3 srcCR = f3_rotz(srcCR_dir, t);

            // 4) 探测器 U/V：局部系方向随机架旋转，再乘间距
            float3 U = f3_scale(f3_rotz(detU_dir, t), du);
            float3 V = f3_scale(f3_rotz(detV_dir, t), dv);

            // 5) detS：像素 (0,0) 的世界坐标
            const float cu = 0.5f * (Nu - 1);
            const float cv = 0.5f * (Nv - 1);
            float3 detS = f3_sub(detC, f3_add(f3_scale(U, cu), f3_scale(V, cv)));

            // 6) 保存
            geo[a] = SConeProjGeomVec{
                src,
                srcCR,
                detS,
                U,
                V,
                make_float3(t, 0.f, 0.f)
            };
        }
    }

    YK_INLINE void build_circular_vec_geometry(
        std::vector<SConeProjGeomVec>& geo,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float3 det_offset = make_float3(0.0f, 0.0f, 0.0f),
        float3 detTiltEuler = make_float3(0.0f, 0.0f, 0.0f),
        float3 src_offset = make_float3(0.0f, 0.0f, 0.0f),
        float3 srcCRTiltEuler = make_float3(0.0f, 0.0f, 0.0f))
    {
        if (Ang <= 0) throw std::runtime_error("Ang must > 0");

        const float two_pi = 2.0f * CUDA_PI;

        std::vector<float> theta;
        theta.resize(Ang);

        for (int a = 0; a < Ang; ++a)
        {
            theta[a] = two_pi * float(a) / float(Ang);
        }

        // 🔥 直接调用你的“新统一接口”
        build_circular_vec_geometry_from_theta(
            geo,
            theta,        // 注意：你现在接口是 vector，不是 pointer
            Ang,
            Nu, Nv,
            du, dv,
            SID, IDD,
            det_offset,
            detTiltEuler,
            src_offset,
            srcCRTiltEuler
        );
    }




    YK_INLINE SConeProjGeomVec build_from_rtk_single(
        float gantryAngle,      // RTK: GantryAngle [rad]
        float outOfPlaneAngle,  // RTK: OutOfPlaneAngle [rad]
        float inPlaneAngle,     // RTK: InPlaneAngle [rad]
        float SAD,              // RTK: SourceToIsocenterDistance
        float SID,              // RTK: SourceToDetectorDistance
        float srcOffsetX,       // RTK: SourceOffsetX
        float srcOffsetY,       // RTK: SourceOffsetY
        float projOffsetX,      // RTK: ProjectionOffsetX
        float projOffsetY,      // RTK: ProjectionOffsetY
        int Nu, int Nv,
        float du, float dv
    )
    {
        // ----------------------------------------
        // 参数转换
        // RTK: 源在 +Z，旋转轴 Y
        // 你:  源在 -Y，旋转轴 Z
        // 但参数含义对齐，不改坐标，只映射数值
        // ----------------------------------------

        // SID → 你的 SID（源到等中心）
        // SID - SAD → 你的 IDD（等中心到探测器）
        float my_SID = SAD;
        float my_IDD = SID - SAD;

        // src_offset: RTK SourceOffsetX/Y 在旋转坐标系内
        // 你的接口也是旋转前施加，含义一致
        float3 src_offset = make_float3(srcOffsetX, srcOffsetY, 0.f);

        // det_offset: RTK ProjectionOffsetX/Y
        float3 det_offset = make_float3(projOffsetX, projOffsetY, 0.f);

        // detTiltEuler:
        //   x = OutOfPlaneAngle（绕局部X轴）
        //   y = 0（RTK无此项）
        //   z = InPlaneAngle（绕探测器法线）
        float3 detTiltEuler = make_float3(outOfPlaneAngle, 0.f, inPlaneAngle);

        // srcCRTiltEuler: RTK无对应，置零
        float3 srcCRTiltEuler = make_float3(0.f, 0.f, 0.f);

        // ----------------------------------------
        // 构造单个投影
        // ----------------------------------------
        std::vector<SConeProjGeomVec> geo;
        std::vector<float> theta = { gantryAngle };

        build_circular_vec_geometry_from_theta(
            geo,
            theta,
            1, Nu, Nv,
            du, dv,
            my_SID, my_IDD,
            det_offset,
            detTiltEuler,
            src_offset,
            srcCRTiltEuler
        );

        return geo[0];
    }

    // 批量版本：从 RTK XML 读出的 per-projection 参数数组转换
    YK_INLINE  void build_from_rtk_geometry(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& gantryAngles,
        float SAD, float SID,                    // 若全局相同
        int Nu, int Nv, float du, float dv,
        // per-projection 偏移，若无则传空vector
        const std::vector<float>& projOffsetX,
        const std::vector<float>& projOffsetY,
        float outOfPlaneAngle = 0.f,
        float inPlaneAngle = 0.f,
        float srcOffsetX = 0.f,
        float srcOffsetY = 0.f
    )
    {
        int Ang = (int)gantryAngles.size();
        geo.resize(Ang);

        for (int a = 0; a < Ang; ++a)
        {
            float px = projOffsetX.empty() ? 0.f : projOffsetX[a];
            float py = projOffsetY.empty() ? 0.f : projOffsetY[a];

            geo[a] = build_from_rtk_single(
                gantryAngles[a],
                outOfPlaneAngle,
                inPlaneAngle,
                SAD, SID,
                srcOffsetX, srcOffsetY,
                px, py,
                Nu, Nv, du, dv
            );
        }
    }



    inline std::string fmt_f3(const float3& v) {
        return fmt::format("[{:.4f}, {:.4f}, {:.4f}]", v.x, v.y, v.z);
    }

    YK_INLINE void print_proj_geom(const SConeProjGeomVec& pg) {
        YK_LOGI(
            "ProjGeom:\n"
            "  src   = {}\n"
            "  srcCR = {}\n"
            "  detS  = {}\n"
            "  detU  = {}\n"
            "  detV  = {}\n"
            "  angle = {:.4f}",
            fmt_f3(pg.src),
            fmt_f3(pg.srcCR),
            fmt_f3(pg.detS),
            fmt_f3(pg.detU),
            fmt_f3(pg.detV),
            pg.angle.x
        );
    }

    YK_INLINE void print_proj_geom(const std::vector<SConeProjGeomVec>& pgv) {
        for (int i = 0; i < pgv.size(); ++i) {
            print_proj_geom(pgv[i]);
        }
    }

} // namespace YK
