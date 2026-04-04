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
        float3 detTiltEuler = make_float3(0.0f, 0.0f, 0.0f),
        float3 src_offset = make_float3(0.0f, 0.0f, 0.0f),
        float3 srcCRTiltEuler = make_float3(0.0f, 0.0f, 0.0f)   // source center-ray Euler angles (pitch, yaw, roll)
    ) // detector tilt Euler angles (pitch, yaw, roll)
    {
        geo.resize(Ang);

        const float3 detU0 = make_float3(du, 0.f, 0.f);  // detector U local
        const float3 detV0 = make_float3(0.f, 0.f, dv);  // detector V local
        const float3 src0 = make_float3(0.f, -SID, 0.f); // source position
        const float3 detC0 = make_float3(0.f, IDD, 0.f); // detector center
        const float3 srcCR0 = make_float3(0.f, 1.f, 0.f); // initial source ray along Y

        for (int a = 0; a < Ang; ++a) {
            float t = theta[a];

            // --------------------------
            // 1) 源位置 + 偏移
            // --------------------------
            float3 src = f3_translate_point(src0, src_offset);
            src = f3_rotz(src, t); // gantry rotation applied to source position as well for consistency
            // --------------------------
            // 2) 探测器中心位置 + 偏移 + gantry旋转
            // --------------------------
            float3 detC = f3_translate_point(detC0, det_offset);
            detC = f3_rotz(detC, t); // gantry rotation

            // --------------------------
            // 3) 源中心射线旋转（欧拉角 + gantry旋转）
            // --------------------------
            float3 srcCR = srcCR0;
            // Apply Euler angles (pitch=X, yaw=Y, roll=Z)
            srcCR = f3_rot_axis(srcCR, make_float3(1, 0, 0), srcCRTiltEuler.x);
            srcCR = f3_rot_axis(srcCR, make_float3(0, 1, 0), srcCRTiltEuler.y);
            srcCR = f3_rot_axis(srcCR, make_float3(0, 0, 1), srcCRTiltEuler.z);
            // Gantry rotation about Z
            srcCR = f3_rotz(srcCR, t);

            // --------------------------
            // 4) 探测器局部 U/V 方向 + tilt（欧拉角）
            // --------------------------
            float3 U = detU0;
            float3 V = detV0;

            // apply gantry rotation
            U = f3_rotz(U, t);
            V = f3_rotz(V, t);

            // detector tilt Euler angles
            U = f3_rot_axis(U, make_float3(1, 0, 0), detTiltEuler.x);
            V = f3_rot_axis(V, make_float3(1, 0, 0), detTiltEuler.x);

            U = f3_rot_axis(U, make_float3(0, 1, 0), detTiltEuler.y);
            V = f3_rot_axis(V, make_float3(0, 1, 0), detTiltEuler.y);

            U = f3_rot_axis(U, make_float3(0, 0, 1), detTiltEuler.z);
            V = f3_rot_axis(V, make_float3(0, 0, 1), detTiltEuler.z);

            // --------------------------
            // 5) 探测器像素原点 detS
            // --------------------------
            float cu = 0.5f * (Nu - 1);
            float cv = 0.5f * (Nv - 1);
            float3 detS = f3_sub(detC, f3_add(f3_mul(U, cu), f3_mul(V, cv)));

            // --------------------------
            // 6) 保存
            // --------------------------
            geo[a] = SConeProjectionVec{ src, srcCR, detS, U, V, make_float3(t,0.f,0.f) };
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

} // namespace YK
