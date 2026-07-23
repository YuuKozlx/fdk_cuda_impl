#pragma once
/**
 * YKSampling2D.cuh
 * ------------------------------------------------------------
 * 2D sampling / interpolation (device-only)
 * Layout: Nu = width (U, fastest), Nv = height (V)
 * Addressing: img[v * Nu + u]
 *
 * Summary order:
 *  1) enums:     EInterp, EBorder
 *  2) helpers:   clampi, fetch2d
 *  3) primitives:
 *      - sample2d_nearest
 *      - sample2d_bilinear
 *  4) wrapper:   sample2d (optional)
 *
 * Defaults:
 *  - interp = Bilinear
 *  - border = Zero (out-of-range -> 0)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace YK {
    namespace Interp {

        // ============================================================
        // 1) Enums
        // ============================================================
        enum class EInterp : int {
            Nearest = 0,
            Bilinear = 1
        };

        enum class EBorder : int {
            Zero = 0,   // out-of-range -> 0 (default)
            Clamp = 1    // clamp-to-edge
        };

        // ============================================================
        // 2) Helpers
        // ============================================================
        __device__ __forceinline__ int clampi(int x, int lo, int hi)
        {
            return x < lo ? lo : (x > hi ? hi : x);
        }

        __device__ __forceinline__ float fetch2d(
            const float* img, int Nu, int Nv,
            int u, int v,
            EBorder border)
        {
            if (border == EBorder::Zero) {
                if ((unsigned)u >= (unsigned)Nu || (unsigned)v >= (unsigned)Nv) return 0.0f;
                return img[v * Nu + u];
            }

            // Clamp
            u = clampi(u, 0, Nu - 1);
            v = clampi(v, 0, Nv - 1);
            return img[v * Nu + u];
        }

        // ============================================================
        // 3) Primitives
        // ============================================================

        // 3.1 Nearest neighbor
        __device__ __forceinline__ float sample2d_nearest(
            const float* img, int Nu, int Nv,
            float u, float v,
            EBorder border = EBorder::Zero)
        {
            int iu = (int)lrintf(u);
            int iv = (int)lrintf(v);
            return fetch2d(img, Nu, Nv, iu, iv, border);
        }

        // 3.2 Bilinear
        __device__ __forceinline__ float sample2d_bilinear(
            const float* img, int Nu, int Nv,
            float u, float v,
            EBorder border = EBorder::Zero)
        {
            // Keep "Zero border" behavior consistent with your original function:
            // if continuous coordinate is outside valid range -> return 0
            if (border == EBorder::Zero) {
                if (u < 0.0f || u >(float)(Nu - 1) ||
                    v < 0.0f || v >(float)(Nv - 1))
                    return 0.0f;
            }

            int u0 = (int)floorf(u);
            int v0 = (int)floorf(v);
            int u1 = u0 + 1;
            int v1 = v0 + 1;

            float fu = u - (float)u0;
            float fv = v - (float)v0;

            float p00 = fetch2d(img, Nu, Nv, u0, v0, border);
            float p10 = fetch2d(img, Nu, Nv, u1, v0, border);
            float p01 = fetch2d(img, Nu, Nv, u0, v1, border);
            float p11 = fetch2d(img, Nu, Nv, u1, v1, border);

            float p0 = p00 + fu * (p10 - p00);
            float p1 = p01 + fu * (p11 - p01);
            return p0 + fv * (p1 - p0);
        }

        // ============================================================
        // 4) Wrapper (optional)
        // ============================================================
        __device__ __forceinline__ float sample2d(
            const float* img, int Nu, int Nv,
            float u, float v,
            EInterp interp = EInterp::Bilinear,
            EBorder border = EBorder::Zero)
        {
            switch (interp) {
            case EInterp::Nearest:
                return sample2d_nearest(img, Nu, Nv, u, v, border);
            case EInterp::Bilinear:
            default:
                return sample2d_bilinear(img, Nu, Nv, u, v, border);
            }
        }

    } // namespace Interp
} // namespace YK
