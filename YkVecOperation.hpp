#pragma once
/**
 * float3 向量基本运算（纯向量库，不包含解析几何）
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <algorithm>

#include "YkGlobals.h"

namespace YK {

    // ============================================================
    // small scalar helpers
    // ============================================================
    __host__ __device__ __forceinline__ float yk_abs(float x) { return x < 0.f ? -x : x; }
    __host__ __device__ __forceinline__ float yk_min(float a, float b) { return a < b ? a : b; }
    __host__ __device__ __forceinline__ float yk_max(float a, float b) { return a > b ? a : b; }
    __host__ __device__ __forceinline__ float yk_clamp(float x, float lo, float hi) { return yk_min(yk_max(x, lo), hi); }
    __host__ __device__ __forceinline__ float yk_lerp(float a, float b, float t) { return a + (b - a) * t; }

    // ============================================================
    // make helpers
    // ============================================================
    __host__ __device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }

    // ============================================================
    // basic arithmetic
    // ============================================================
    __host__ __device__ __forceinline__ float3 f3_add(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __host__ __device__ __forceinline__ float3 f3_sub(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __host__ __device__ __forceinline__ float3 f3_mul(float3 a, float t) { return f3(a.x * t, a.y * t, a.z * t); }
    __host__ __device__ __forceinline__ float3 f3_mul(float t, float3 a) { return f3(a.x * t, a.y * t, a.z * t); }
    __host__ __device__ __forceinline__ float3 f3_div(float3 a, float t) { float inv = 1.f / t; return f3(a.x * inv, a.y * inv, a.z * inv); }

    __host__ __device__ __forceinline__ float3 f3_mul_comp(float3 a, float3 b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
    __host__ __device__ __forceinline__ float3 f3_div_comp(float3 a, float3 b) { return f3(a.x / b.x, a.y / b.y, a.z / b.z); }

    __host__ __device__ __forceinline__ float3 f3_neg(float3 a) { return f3(-a.x, -a.y, -a.z); }

    // fused: a*t + b  (常用于累加、线性组合)
    __host__ __device__ __forceinline__ float3 f3_mad(float3 a, float t, float3 b) {
        return f3(a.x * t + b.x, a.y * t + b.y, a.z * t + b.z);
    }

    // ============================================================
    // dot / cross
    // ============================================================
    __host__ __device__ __forceinline__ float f3_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

    __host__ __device__ __forceinline__ float3 f3_cross(float3 a, float3 b) {
        return f3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    // ============================================================
    // length / normalize
    // ============================================================
    __host__ __device__ __forceinline__ float f3_len2(float3 a) { return f3_dot(a, a); }

    // host/device 分别走最合适的实现（device 用 sqrtf/rsqrtf）
    __host__ __device__ __forceinline__ float f3_len(float3 a) {
#if defined(__CUDA_ARCH__)
        return sqrtf(f3_len2(a));
#else
        return std::sqrt(f3_len2(a));
#endif
    }

    __host__ __device__ __forceinline__ float3 f3_normalize(float3 a, float eps = 1e-20f) {
        float l2 = f3_len2(a);
        if (l2 < eps) return f3(0.f, 0.f, 0.f);
#if defined(__CUDA_ARCH__)
        float inv = rsqrtf(l2);
#else
        float inv = 1.f / std::sqrt(l2);
#endif
        return f3(a.x * inv, a.y * inv, a.z * inv);
    }

    // ============================================================
    // component ops
    // ============================================================
    __host__ __device__ __forceinline__ float3 f3_min(float3 a, float3 b) { return f3(yk_min(a.x, b.x), yk_min(a.y, b.y), yk_min(a.z, b.z)); }
    __host__ __device__ __forceinline__ float3 f3_max(float3 a, float3 b) { return f3(yk_max(a.x, b.x), yk_max(a.y, b.y), yk_max(a.z, b.z)); }
    __host__ __device__ __forceinline__ float3 f3_abs(float3 a) { return f3(yk_abs(a.x), yk_abs(a.y), yk_abs(a.z)); }
    __host__ __device__ __forceinline__ float3 f3_clamp(float3 a, float lo, float hi) {
        return f3(yk_clamp(a.x, lo, hi), yk_clamp(a.y, lo, hi), yk_clamp(a.z, lo, hi));
    }

    __host__ __device__ __forceinline__ float3 f3_lerp(float3 a, float3 b, float t) {
        return f3(yk_lerp(a.x, b.x, t), yk_lerp(a.y, b.y, t), yk_lerp(a.z, b.z, t));
    }

    // ============================================================
    // comparisons (tolerance)
    // ============================================================
    __host__ __device__ __forceinline__ bool f3_all_finite(float3 a) {
#if defined(__CUDA_ARCH__)
        return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
#else
        return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
#endif
    }

    __host__ __device__ __forceinline__ bool f3_near(float3 a, float3 b, float eps = 1e-6f) {
        return yk_abs(a.x - b.x) <= eps && yk_abs(a.y - b.y) <= eps && yk_abs(a.z - b.z) <= eps;
    }

} // namespace YK
