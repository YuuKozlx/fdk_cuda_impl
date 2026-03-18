#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include "YkGlobals.h"
#include "YkVecOperation.hpp"

namespace YK {

    /**
     * @brief Circular cone-beam vector geometry with theta=0 at source on -X axis
     *
     * World: X right, Y forward, Z up (RH)
     * Rotation: theta increases CCW around +Z
     *
     * Base at theta=0:
     *   src0  = (-SOD, 0, 0)
     *   detC0 = ( +ODD, 0, 0)   where ODD = SDD - SOD
     *   detU0 = ( 0,  -du, 0)   (u increases toward -Y at theta=0)
     *   detV0 = ( 0,   0, dv)   (v increases toward +Z)
     *
     * Then apply Rz(theta) to src0, detC0, detU0, detV0.
     *
     * ----------------------------------------------------------------------------
     * IMPORTANT: semantics of detector_pixel_center_offset{U,V}_pix (THIS FUNCTION ONLY)
     * ----------------------------------------------------------------------------
     * These parameters ONLY define the in-plane shift of the detector pixel coordinate
     * system relative to the detector geometric center detC, measured in pixels along
     * the detector axes (detU, detV).
     *
     * They DO NOT represent:
     *   - principal-point (source projection) offsets,
     *   - source position correction,
     *   - any geometry calibration result.
     *
     * When detector tilt is enabled:
     *   - The offsets MUST be applied AFTER gantry rotation and AFTER detector tilt.
     *   - In other words, detector_pixel_center_offset{U,V}_pix are always interpreted
     *     along the FINAL detU / detV directions used by the pixel coordinate system.
     *
     * Concretely, the offsets are applied only when constructing detS:
     *
     *   detS = detC
     *        - ( (Nu-1)/2 + detector_pixel_center_offsetU_pix ) * detU
     *        - ( (Nv-1)/2 + detector_pixel_center_offsetV_pix ) * detV
     *
     * This guarantees that the offset meaning remains a pure in-plane pixel shift on
     * the rotated detector, independent of gantry angle and tilt.
     *
     * Tilt (optional, detector-local axes, order u->v->n):
     *   detTiltUVN = (phi_u, phi_v, phi_n) in radians
     *   Axes are defined at each view *before tilt*:
     *     e_u = normalize(detU)
     *     e_v = normalize(detV)
     *     e_n = normalize(detC - src)   (center ray direction through detC)
     *
     *   Applied order (body-fixed / about updated axes through detC):
     *     u then v then n
     *
     * Angle storage:
     *   geo[a].ang.x = theta[a] (rad), ang.y/z reserved
     */
    inline void build_circular_vec_geometry_from_theta(
        std::vector<SConeProjectionVec>& geo,
        const float* theta,          // [Ang] radians
        int Ang, int Nu, int Nv,
        float SOD, float SDD,
        float du, float dv,
        float detector_pixel_center_offsetU_pix = 0.0f,
        float detector_pixel_center_offsetV_pix = 0.0f,
        float3 detTiltUVN = make_float3(0.0f, 0.0f, 0.0f)) // (phi_u, phi_v, phi_n)
    {
        geo.resize(Ang);

        const float ODD = SDD - SOD;

        // theta=0 reference (source at -X) 源点必须这么定义
        const float3 src0 = make_float3(-SOD, 0.0f, 0.0f);
        const float3 detC0 = make_float3(ODD, 0.0f, 0.0f);

        // Match MATLAB-style U(theta) = du*[sinθ,-cosθ,0]  => U(0)=(0,-du,0)
        const float3 detU0 = make_float3(0.0f, -du, 0.0f);
        const float3 detV0 = make_float3(0.0f, 0.0f, dv);

        // pixel-center indices (in pixels) + pixel-center shift relative to detC (in pixels)
        // NOTE: offsets are applied along FINAL detU/detV after gantry rotation + tilt.
        const float cu = 0.5f * (Nu - 1) + detector_pixel_center_offsetU_pix;
        const float cv = 0.5f * (Nv - 1) + detector_pixel_center_offsetV_pix;

        const float phi_u = detTiltUVN.x;
        const float phi_v = detTiltUVN.y;
        const float phi_n = detTiltUVN.z;

        for (int a = 0; a < Ang; ++a) {

            const float t = theta[a];

            // Rotate all base elements by gantry angle (about origin)
            const float3 src = f3_rotz_p(src0, t);
            const float3 detC = f3_rotz_p(detC0, t);
            float3 detU = f3_rotz(detU0, t);
            float3 detV = f3_rotz(detV0, t);

            // Optional detector tilts in detector-local axes (body-fixed), order u -> v -> n
            if ((fabs(phi_u) >= 1e-5f) || (fabs(phi_v) >= 1e-5f) || (fabs(phi_n) >= 1e-5f)) {

                // 1) rotate about current U axis
                {
                    float3 u_unit = f3_normalize(detU);
                    detU = f3_rot_axis(detU, u_unit, phi_u);
                    detV = f3_rot_axis(detV, u_unit, phi_u);
                }

                // 2) rotate about UPDATED V axis
                {
                    float3 v_unit = f3_normalize(detV);
                    detU = f3_rot_axis(detU, v_unit, phi_v);
                    detV = f3_rot_axis(detV, v_unit, phi_v);
                }

                // 3) rotate about UPDATED detector normal n = U x V
                {
                    float3 n_unit = f3_normalize(f3_cross(detU, detV)); // right-hand: U×V
                    detU = f3_rot_axis(detU, n_unit, phi_n);
                    detV = f3_rot_axis(detV, n_unit, phi_n);
                }
            }

            // Pixel (u=0,v=0) world origin detS from detC and detector basis
            // Apply pixel-center offset ALONG FINAL detU/detV (after gantry rotation + tilt).
            const float3 detS = f3_sub(detC, f3_add(f3_mul(detU, cu), f3_mul(detV, cv)));

            // store angle (rad) in ang.x for alignment-friendly layout
            const float3 ang = make_float3(t, 0.0f, 0.0f);

            // Ensure SConeProjectionVec field order matches this initializer:
            // { src, detS, detU, detV, ang }
            geo[a] = SConeProjectionVec{ src, detS, detU, detV, ang };
        }
    }

    inline void build_circular_vec_geometry(
        std::vector<SConeProjectionVec>& geo,
        int Ang, int Nu, int Nv,
        float SOD, float SDD,
        float du, float dv,
        float detector_pixel_center_offsetU_pix = 0.0f,
        float detector_pixel_center_offsetV_pix = 0.0f,
        float3 detTiltUVN = make_float3(0.0f, 0.0f, 0.0f)) // default 0
    {
        std::vector<float> theta(Ang);
        const float two_pi = 2.0f * CUDA_PI;

        for (int a = 0; a < Ang; ++a) {
            theta[a] = two_pi * a / Ang;
        }

        build_circular_vec_geometry_from_theta(
            geo, theta.data(),
            Ang, Nu, Nv,
            SOD, SDD,
            du, dv,
            detector_pixel_center_offsetU_pix,
            detector_pixel_center_offsetV_pix,
            detTiltUVN);
    }

} // namespace YK
