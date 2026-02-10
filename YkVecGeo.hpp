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
     * Pixel (u=0,v=0):
     *   detS = detC
     *        - ((Nu-1)/2 + offsetU_pix) * detU
     *        - ((Nv-1)/2 + offsetV_pix) * detV
     *
     * Tilt (optional, detector-local axes, order u->v->n):
     *   detTiltUVN = (phi_u, phi_v, phi_n) in radians
     *   Axes are defined at each view *before tilt*:
     *     e_u = normalize(detU)
     *     e_v = normalize(detV)
     *     e_n = normalize(detC - src)   (center ray direction)
     *
     *   Applied order (column vectors):
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
        float offsetU_pix = 0.0f,
        float offsetV_pix = 0.0f,
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

        // pixel-center offsets (in pixels)
        const float cu = 0.5f * (Nu - 1) + offsetU_pix;
        const float cv = 0.5f * (Nv - 1) + offsetV_pix;

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
            if ((fabs(phi_u) >= 1e-4f) || (fabs(phi_v) >= 1e-4f) || (fabs(phi_n) >= 1e-4f)) {

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

            // Detector origin (pixel 0,0) from center + basis
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
        float offsetU_pix = 0.0f,
        float offsetV_pix = 0.0f,
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
            offsetU_pix, offsetV_pix,
            detTiltUVN);
    }

} // namespace YK
