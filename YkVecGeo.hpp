#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include <stdexcept>
#include <vector_functions.hpp>
#include <vector_types.h>
#include "YkGlobals.h"
#include "YkVecOperation.hpp"

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


    inline void build_circular_vec_geometry_from_theta(
        std::vector<SConeProjectionVec>& geo,
        const std::vector<float>& theta,      // [Ang] radians
        int Ang, int Nu, int Nv,
        float du, float dv,
        float SID, float IDD,
        float3 det_offset = make_float3(0.0f, 0.0f, 0.0f),
        float3 detTiltEuler = make_float3(0.0f, 0.0f, 0.0f),  // OutOfPlane(x), reserved(y), InPlane(z)
        float3 src_offset = make_float3(0.0f, 0.0f, 0.0f),
        float3 srcCRTiltEuler = make_float3(0.0f, 0.0f, 0.0f) // pitch(x), yaw(y), roll(z) in local frame
    )
    {
        geo.resize(Ang);

        const float3 detU0 = make_float3(du, 0.f, 0.f);  // detector U local
        const float3 detV0 = make_float3(0.f, 0.f, dv);   // detector V local
        const float3 src0 = make_float3(0.f, -SID, 0.f); // source position
        const float3 detC0 = make_float3(0.f, IDD, 0.f); // detector center
        const float3 srcCR0 = make_float3(0.f, 1.f, 0.f); // initial center ray along Y

        for (int a = 0; a < Ang; ++a)
        {
            float t = theta[a];

            // --------------------------------------------------
            // 机架旋转后的局部坐标轴
            // --------------------------------------------------
            const float3 localX = f3_rotz(make_float3(1.f, 0.f, 0.f), t);
            const float3 localY = f3_rotz(make_float3(0.f, 1.f, 0.f), t);
            // localZ = (0,0,1) 不变，绕Z轴旋转后Z轴本身不动

            // --------------------------------------------------
            // 1) 源位置：先加局部偏移，再做机架旋转
            //    src_offset 在旋转前施加 = 在局部系内定义偏移
            // --------------------------------------------------
            float3 src = f3_translate_point(src0, src_offset);
            src = f3_rotz_p(src, t);

            // --------------------------------------------------
            // 2) 探测器中心：先加局部偏移，再做机架旋转
            // --------------------------------------------------
            float3 detC = f3_translate_point(detC0, det_offset);
            detC = f3_rotz_p(detC, t);

            // --------------------------------------------------
            // 3) 源中心射线：先在局部系施加 tilt，再做机架旋转
            //    对应 RTK 的 srcCRTiltEuler，顺序 pitch→yaw→roll
            // --------------------------------------------------
            float3 srcCR = srcCR0;
            srcCR = f3_rot_axis(srcCR, make_float3(1.f, 0.f, 0.f), srcCRTiltEuler.x); // pitch
            srcCR = f3_rot_axis(srcCR, make_float3(0.f, 1.f, 0.f), srcCRTiltEuler.y); // yaw
            srcCR = f3_rot_axis(srcCR, make_float3(0.f, 0.f, 1.f), srcCRTiltEuler.z); // roll
            srcCR = f3_rotz(srcCR, t); // 最后整体随机架旋转

            // --------------------------------------------------
            // 4) 探测器 U/V 方向：先做机架旋转，再绕局部轴施加 tilt
            // --------------------------------------------------
            float3 U = f3_rotz(detU0, t);
            float3 V = f3_rotz(detV0, t);

            // OutOfPlaneAngle：绕机架旋转后的局部 X 轴
            U = f3_rot_axis(U, localX, detTiltEuler.x);
            V = f3_rot_axis(V, localX, detTiltEuler.x);

            // reserved y：绕局部 Y 轴（RTK 无对应，扩展用）
            U = f3_rot_axis(U, localY, detTiltEuler.y);
            V = f3_rot_axis(V, localY, detTiltEuler.y);

            // InPlaneAngle：绕探测器法线（U×V 方向）
            // 经过前两步后法线方向已更新，动态计算
            float3 normal = f3_normalize(f3_cross(U, V));
            U = f3_rot_axis(U, normal, detTiltEuler.z);
            V = f3_rot_axis(V, normal, detTiltEuler.z);

            // detC 也需要跟随 OutOfPlane tilt（探测器中心随探测器倾斜）
            // InPlane 是面内旋转，detC 不动
            detC = f3_rot_axis(detC, localX, detTiltEuler.x);
            detC = f3_rot_axis(detC, localY, detTiltEuler.y);

            // --------------------------------------------------
            // 5) 探测器像素原点 detS
            //    detC 是中心，detS 是像素 (0,0) 的世界坐标
            // --------------------------------------------------
            float cu = 0.5f * (Nu - 1);
            float cv = 0.5f * (Nv - 1);
            float3 detS = f3_sub(detC, f3_add(f3_mul(U, cu), f3_mul(V, cv)));

            // --------------------------------------------------
            // 6) 保存
            // --------------------------------------------------
            geo[a] = SConeProjectionVec{
                src,
                srcCR,
                detS,
                U,
                V,
                make_float3(t, 0.f, 0.f)
            };
        }
    }

    inline void build_circular_vec_geometry(
        std::vector<SConeProjectionVec>& geo,
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




    SConeProjectionVec build_from_rtk_single(
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
        std::vector<SConeProjectionVec> geo;
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
    void build_from_rtk_geometry(
        std::vector<SConeProjectionVec>& geo,
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

} // namespace YK
