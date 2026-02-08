#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include "YkGlobals.h"

namespace YK {



    /**
 * @brief Build circular cone-beam vector geometry (MATLAB-consistent version)
 *
 * This function builds a circular cone-beam CT trajectory using a **vector
 * geometry formulation**, fully consistent with the following MATLAB model:
 *
 *   Source position:
 *     S(theta) = SOD * [  sin(theta);
 *                         -cos(theta);
 *                          0           ]
 *
 *   Detector center:
 *     D(theta) = ODD * [ -sin(theta);
 *                         cos(theta);
 *                         0           ]
 *
 *   Detector column direction (u direction):
 *     U(theta) = du * [  cos(theta);
 *                         sin(theta);
 *                         0           ]
 *
 *   Detector row direction (v direction):
 *     V        = dv * [  0;
 *                         0;
 *                         1           ]
 *
 * where:
 *   - theta increases with view index a (counter-clockwise around +Z)
 *   - SOD = source-to-isocenter distance
 *   - SDD = source-to-detector distance
 *   - ODD = SDD - SOD (isocenter-to-detector distance)
 *
 * Coordinate / memory conventions (VERY IMPORTANT):
 * -------------------------------------------------
 *   - World coordinate system:
 *       X right, Y forward, Z up (right-handed)
 *
 *   - View index a:
 *       theta = 2*pi * a / Ang
 *
 *   - Projection data layout:
 *       proj[a][v][u]  (A-V-U order)
 *       u = detector column index (fastest)
 *       v = detector row index
 *
 *   - Detector vectors:
 *       detU : direction of increasing u (column direction)
 *       detV : direction of increasing v (row direction)
 *
 *   - Pixel (u=0, v=0) corresponds to:
 *       detector center minus half detector size:
 *         detS = detCenter
 *                - ((Nu-1)/2 + offsetU_pix) * detU
 *                - ((Nv-1)/2 + offsetV_pix) * detV
 *
 * This definition is compatible with:
 *   - FDK vector backprojection
 *   - bilinear sampling using img[v * Nu + u]
 *   - u-axis filtering (Ram-Lak along detector columns)
 *
 * @param[out] geo
 *     Output vector of SConeProjectionVec, size = Ang
 *
 * @param[in] Ang
 *     Number of projection views
 *
 * @param[in] Nu
 *     Number of detector columns (u direction)
 *
 * @param[in] Nv
 *     Number of detector rows (v direction)
 *
 * @param[in] SOD
 *     Source-to-isocenter distance
 *
 * @param[in] SDD
 *     Source-to-detector distance
 *
 * @param[in] du
 *     Detector pixel width (column spacing)
 *
 * @param[in] dv
 *     Detector pixel height (row spacing)
 *
 * @param[in] offsetU_pix
 *     Optional detector offset in u (columns), in pixel units
 *     Positive values shift the detector towards +detU
 *
 * @param[in] offsetV_pix
 *     Optional detector offset in v (rows), in pixel units
 *     Positive values shift the detector towards +detV
 */
    inline void build_circular_vec_geometry_from_theta(
        std::vector<SConeProjectionVec>& geo,
        const float* theta,          // [Ang] 外部角度（弧度）
        int Ang, int Nu, int Nv,
        float SOD, float SDD,
        float du, float dv,
        float offsetU_pix = 0.0f,
        float offsetV_pix = 0.0f)
    {
        geo.resize(Ang);

        const float ODD = SDD - SOD;

        for (int a = 0; a < Ang; ++a) {

            float t = theta[a];
            float c = cosf(t);
            float s = sinf(t);

            // -------------------------------
            // Source (绕 +Z)
            // S = [ SOD*sinθ , -SOD*cosθ , 0 ]
            // -------------------------------
            float3 src = make_float3(
                SOD * s,
                -SOD * c,
                0.0f
            );

            // -------------------------------
            // Detector center
            // D = [ -ODD*sinθ , ODD*cosθ , 0 ]
            // -------------------------------
            float3 detC = make_float3(
                -ODD * s,
                ODD * c,
                0.0f
            );

            // -------------------------------
            // Detector basis
            // -------------------------------
            float3 detU = make_float3(
                du * c,
                du * s,
                0.0f
            );

            float3 detV = make_float3(
                0.0f, 0.0f, dv
            );

            // -------------------------------
            // Detector origin (pixel 0,0)
            // -------------------------------
            float cu = (Nu - 1) * 0.5f + offsetU_pix;
            float cv = (Nv - 1) * 0.5f + offsetV_pix;

            float3 detS = make_float3(
                detC.x - cu * detU.x - cv * detV.x,
                detC.y - cu * detU.y - cv * detV.y,
                detC.z - cu * detU.z - cv * detV.z
            );

            geo[a] = SConeProjectionVec{ src, detS, detU, detV };
        }
    }


    inline void build_circular_vec_geometry(
        std::vector<SConeProjectionVec>& geo,
        int Ang, int Nu, int Nv,
        float SOD, float SDD,
        float du, float dv,
        float offsetU_pix = 0.0f,
        float offsetV_pix = 0.0f)
    {
        std::vector<float> theta(Ang);
        const float two_pi = 2.0f * 3.14159265358979323846f;

        for (int a = 0; a < Ang; ++a)
            theta[a] = two_pi * a / Ang;

        build_circular_vec_geometry_from_theta(
            geo, theta.data(),
            Ang, Nu, Nv,
            SOD, SDD,
            du, dv,
            offsetU_pix, offsetV_pix);
    }



} // namespace YK
