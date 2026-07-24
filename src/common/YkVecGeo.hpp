#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <vector>


#include <stdexcept>
#include <string>
#include <vector_functions.hpp>
#include <vector_types.h>
#include "../global/YkGlobals.h"
#include "../global/YkLog.h"
#include "../global/YkMacro.hpp"
#include "../util/YkVecOperation.hpp"
#include "YKCBCT/geometry/YkProjectionGeometryBuilders.hpp"

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
        const std::vector<float>& theta, // rad
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float3 det_offset = make_float3(0.f, 0.f, 0.f),   // u(x) offset ; idd(y) offset ; v(z) offset
        float3 detTilt_deg = make_float3(0.f, 0.f, 0.f),   // outOfPlane(x), lateral(y), inPlane(z)
        float3 src_offset = make_float3(0.f, 0.f, 0.f),   // x/y/z offset of source position
        float3 srcCRTilt_deg = make_float3(0.f, 0.f, 0.f)    // pitch(x), yaw(y), roll(z)
    )
    {
        geo.resize(Ang);

        auto deg2rad = [](float deg) { return deg * CUDA_PI / 180.f; };

        const float tiltu_deg = detTilt_deg.x;
        const float tiltv_deg = detTilt_deg.z;
        const float tiltn_deg = detTilt_deg.y;

        // ---- 探测器 U/V 方向向量（局部系）----
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

        // ---- 主射线方向（局部系）----
        float3 srcCR_dir = make_float3(0.f, 1.f, 0.f);
        {
            srcCR_dir = f3_rot_axis(srcCR_dir, make_float3(1.f, 0.f, 0.f), deg2rad(srcCRTilt_deg.x));
            srcCR_dir = f3_rot_axis(srcCR_dir, make_float3(0.f, 0.f, 1.f), deg2rad(srcCRTilt_deg.y));
            srcCR_dir = f3_rot_axis(srcCR_dir, f3_normalize(srcCR_dir), deg2rad(srcCRTilt_deg.z));
            srcCR_dir = f3_normalize(srcCR_dir);
        }

        const float3 src0 = make_float3(0.f, -SID, 0.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);

        const float cu = 0.5f * (Nu - 1);
        const float cv = 0.5f * (Nv - 1);

        for (int a = 0; a < Ang; ++a)
        {
            const float t = theta[a];

            // 1) 源位置
            float3 src = f3_rotz(src0 + src_offset, t);

            // 2) 探测器中心
            float3 detC = f3_rotz(detC0 + det_offset, t);

            // 3) 中心射线
            float3 srcCR = f3_rotz(srcCR_dir, t);

            // 4) 探测器 U/V，乘间距
            float3 U = f3_rotz(detU_dir, t) * du;
            float3 V = f3_rotz(detV_dir, t) * dv;

            // 5) detS：像素 (0,0) 世界坐标
            float3 detS = detC - U * cu - V * cv;

            // 6) 保存，float3 -> float4，w=0
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

    YK_INLINE void build_circular_vec_geometry(
        std::vector<SConeProjGeomVec>& geo,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float3 det_offset = make_float3(0.0f, 0.0f, 0.0f), // u(x) offset ; idd(y) offset ; v(z) offset
        float3 detTiltEuler = make_float3(0.0f, 0.0f, 0.0f), // outOfPlane(x), lateral(y), inPlane(z)
        float3 src_offset = make_float3(0.0f, 0.0f, 0.0f), // x/y/z offset of source position
        float3 srcCRTiltEuler = make_float3(0.0f, 0.0f, 0.0f) // pitch(x), yaw(y), roll(z)

    )
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


    // 为每视角设置几何参数
    YK_INLINE void build_circular_vec_geometry_perframe(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& theta,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        const std::vector<float3>& det_offsets, // u(x) offset ; idd(y) offset ; v(z) offset
        const std::vector<float3>& src_offsets, // x/y/z offset of source position
        const std::vector<float3>& detTiltEuler_degs, // outOfPlane(x), lateral(y), inPlane(z)
        const std::vector<float3>& srcCRTiltEuler_degs // pitch(x), yaw(y), roll(z)
    )
    {
        // size==1 退化为全局常量
        auto get = [](const std::vector<float3>& v, int i) -> float3 {
            return v.size() == 1 ? v[0] : v[i];
            };

        auto deg2rad = [](float deg) { return deg * CUDA_PI / 180.f; };

        geo.resize(Ang);

        const float3 src0 = make_float3(0.f, -SID, 0.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);
        const float  cu = 0.5f * (Nu - 1);
        const float  cv = 0.5f * (Nv - 1);

        for (int a = 0; a < Ang; ++a)
        {
            const float  t = theta[a];
            const float3 det_offset = get(det_offsets, a);
            const float3 src_offset = get(src_offsets, a);
            const float3 detTilt_deg = get(detTiltEuler_degs, a);
            const float3 srcCR_tilt = get(srcCRTiltEuler_degs, a);

            // ---- 探测器 U/V 方向向量（局部系，per-frame tilt）----
            float3 detU_dir = make_float3(1.f, 0.f, 0.f);
            float3 detV_dir = make_float3(0.f, 0.f, 1.f);
            {
                const float tiltu_deg = detTilt_deg.x;
                const float tiltv_deg = detTilt_deg.z;
                const float tiltn_deg = detTilt_deg.y;

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

            // ---- 主射线方向（局部系，per-frame tilt）----
            float3 srcCR_dir = make_float3(0.f, 1.f, 0.f);
            {
                srcCR_dir = f3_rot_axis(srcCR_dir, make_float3(1.f, 0.f, 0.f), deg2rad(srcCR_tilt.x));
                srcCR_dir = f3_rot_axis(srcCR_dir, make_float3(0.f, 0.f, 1.f), deg2rad(srcCR_tilt.y));
                srcCR_dir = f3_rot_axis(srcCR_dir, f3_normalize(srcCR_dir), deg2rad(srcCR_tilt.z));
                srcCR_dir = f3_normalize(srcCR_dir);
            }

            // ---- 旋转到当前角度 ----
            float3 src = f3_rotz(src0 + src_offset, t);
            float3 detC = f3_rotz(detC0 + det_offset, t);
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




#ifndef __CUDACC__
#include <util/fmt/format.h>

    inline std::string fmt_f4(const float4& v) {
        return fmt::format("[{:.4f}, {:.4f}, {:.4f},{:.4f}]", v.x, v.y, v.z, v.w);
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
            fmt_f3(f4_to_f3(pg.src)),
            fmt_f3(f4_to_f3(pg.srcCR)),
            fmt_f3(f4_to_f3(pg.detS)),
            fmt_f3(f4_to_f3(pg.detU)),
            fmt_f3(f4_to_f3(pg.detV)),
            pg.angle.x
        );
    }

    YK_INLINE void print_proj_geom(const std::vector<SConeProjGeomVec>& pgv) {
        for (size_t i = 0; i < pgv.size(); ++i) {
            print_proj_geom(pgv[i]);
        }
    }

#endif // __CUDACC__

} // namespace YK


/*namespace YK {
    // planar CT
    YK_INLINE void build_planar_ct_vec_geometry(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& theta,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float R)  // 源偏离Y轴的距离
    {
        geo.resize(Ang);

        const float3 detU_dir = make_float3(1.f, 0.f, 0.f);
        const float3 detV_dir = make_float3(0.f, 0.f, 1.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);
        const float3 U = detU_dir * du;
        const float3 V = detV_dir * dv;
        const float  cu = 0.5f * (Nu - 1);
        const float  cv = 0.5f * (Nv - 1);
        const float3 detS = detC0 - U * cu - V * cv;

        // 初始源位置 (R, -SID, 0)，绕Y轴旋转
        const float3 src0 = make_float3(R, -SID, 0.f);

        for (int a = 0; a < Ang; ++a)
        {
            const float t = theta[a];

            // 绕Y轴旋转 src0
            const float3 src = f3_roty(src0, t);

            // 主射线方向：从源指向探测器中心
            const float3 srcCR = f3(0, 1, 0);

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

}

namespace YK {

    // 通用平面CT几何构建函数，支持自定义源轨迹
    // src_pos_func : 给定角度（弧度），返回源点的世界坐标 (x, y, z)
    // 其他参数与原函数含义相同：
    //   theta : 各视角角度（弧度），长度 = Ang
    //   Ang, Nu, Nv, du, dv, SID, IDD
    // 探测器中心固定于 (0, IDD, 0)，探测器U、V方向沿世界坐标轴，不随角度旋转
    YK_INLINE void build_planar_ct_vec_geometry_custom(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& theta,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        const std::function<float3(float angle_rad)>& src_pos_func)
    {
        geo.resize(Ang);

        // 探测器固定参数
        const float3 detU_dir = make_float3(1.f, 0.f, 0.f);
        const float3 detV_dir = make_float3(0.f, 0.f, 1.f);
        const float3 detC0 = make_float3(0.f, IDD, 0.f);
        const float3 U = detU_dir * du;
        const float3 V = detV_dir * dv;
        const float  cu = 0.5f * (Nu - 1);
        const float  cv = 0.5f * (Nv - 1);
        const float3 detS = detC0 - U * cu - V * cv;

        for (int a = 0; a < Ang; ++a)
        {
            const float t = theta[a];
            const float3 src = src_pos_func(t);

            // 主射线方向：从源指向探测器中心
            // 主射线方向：从源指向探测器中心
            const float3 srcCR = f3(0, 1, 0);

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

    // 便捷版本：椭圆轨迹（参数与圆形兼容：theta=0时源在 (a, -SID, 0)）
    // a : X轴半长（对应 cos(theta) 系数）
    // b : Z轴半长（对应 -sin(theta) 系数，保持与原圆形轨迹相同的旋转方向）
    YK_INLINE void build_planar_ct_vec_geometry_ellipse(
        std::vector<SConeProjGeomVec>& geo,
        const std::vector<float>& theta,
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float a, float b)
    {
        auto src_pos = [=](float t) -> float3 {
            return make_float3(a * cos(t), -SID, -b * sin(t));
            };
        build_planar_ct_vec_geometry_custom(geo, theta, Ang, Nu, Nv, du, dv, SID, IDD, src_pos);
    }

} // namespace YK*/
