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




    // ============================================================
    // Geometry transforms (points / vectors)
    // ------------------------------------------------------------
    // 设计原则：
    //   - point（点）：表示空间中的“位置”，平移对其有意义
    //   - vector（向量）：表示“方向 / 位移”，平移对其无意义
    //
    // 坐标系约定：
    //   - 右手坐标系
    //   - X 向右，Y 向前，Z 向上
    //   - 旋转角度为弧度，正方向遵循右手法则
    //
    // 本文件只提供“几何原语”，不引入高层语义（CT/FDK）
    // ============================================================


    // ============================================================
    // Translation
    // ------------------------------------------------------------
    // 点的平移：p' = p + t
    // 向量不参与平移（no-op）
    // ============================================================

    __host__ __device__ __forceinline__
        float3 f3_translate_point(float3 p, float3 t)
    {
        // Translate a point by vector t
        return f3_add(p, t);
    }

    __host__ __device__ __forceinline__
        float3 f3_translate_vec(float3 v, float3 /*t*/)
    {
        // Vectors are invariant under translation
        return v;
    }


    // ============================================================
    // Reflections (Mirrors)
    // ------------------------------------------------------------
    // 用于对称、翻转、坐标系变换等
    // ============================================================

    // ---------- mirror about coordinate planes (through origin) ----------
    // mirror about YZ plane (flip X)
    __host__ __device__ __forceinline__
        float3 f3_mirror_x(float3 p)
    {
        return f3(-p.x, p.y, p.z);
    }

    // mirror about XZ plane (flip Y)
    __host__ __device__ __forceinline__
        float3 f3_mirror_y(float3 p)
    {
        return f3(p.x, -p.y, p.z);
    }

    // mirror about XY plane (flip Z)
    __host__ __device__ __forceinline__
        float3 f3_mirror_z(float3 p)
    {
        return f3(p.x, p.y, -p.z);
    }


    // ---------- mirror across an arbitrary plane ----------
    // Plane passes through origin, with unit normal n_unit
    // 数学公式： p' = p - 2*(p·n)*n
    __host__ __device__ __forceinline__
        float3 f3_reflect_plane_origin(float3 p, float3 n_unit)
    {
        float k = 2.0f * f3_dot(p, n_unit);
        return f3_sub(p, f3_mul(n_unit, k));
    }

    // Plane passes through point p0, with unit normal n_unit
    // p' = p0 + reflect( p - p0 )
    __host__ __device__ __forceinline__
        float3 f3_reflect_plane(float3 p, float3 p0, float3 n_unit)
    {
        float3 q = f3_sub(p, p0);
        q = f3_reflect_plane_origin(q, n_unit);
        return f3_add(p0, q);
    }


    // ---------- reflection across an axis (line) ----------
    // 这是绕轴的 180° 旋转
    // 轴通过原点，方向为单位向量 u_unit
    // 数学公式： p' = 2*(p·u)*u - p
    __host__ __device__ __forceinline__
        float3 f3_reflect_axis_origin(float3 p, float3 u_unit)
    {
        float k = 2.0f * f3_dot(p, u_unit);
        return f3_sub(f3_mul(u_unit, k), p);
    }

    // 轴通过点 p0，方向为 u_unit
    __host__ __device__ __forceinline__
        float3 f3_reflect_axis(float3 p, float3 p0, float3 u_unit)
    {
        float3 q = f3_sub(p, p0);
        q = f3_reflect_axis_origin(q, u_unit);
        return f3_add(p0, q);
    }


    // ============================================================
    // Rotations: fixed axes (fast path)
    // ------------------------------------------------------------
    // 高性能路径：绕 X / Y / Z 轴的右手旋转
    // 用于机架旋转、理想几何、主路径计算
    // ============================================================

    // ---------- rotate vector about X axis (origin) ----------
    __host__ __device__ __forceinline__
        float3 f3_rotx(float3 v, float a)
    {
        // Rx(a) * v
        float c = cosf(a), s = sinf(a);
        return f3(
            v.x,
            c * v.y - s * v.z,
            s * v.y + c * v.z
        );
    }

    // ---------- rotate vector about Y axis (origin) ----------
    __host__ __device__ __forceinline__
        float3 f3_roty(float3 v, float a)
    {
        // Ry(a) * v
        float c = cosf(a), s = sinf(a);
        return f3(
            c * v.x + s * v.z,
            v.y,
            -s * v.x + c * v.z
        );
    }

    // ---------- rotate vector about Z axis (origin) ----------
    __host__ __device__ __forceinline__
        float3 f3_rotz(float3 v, float a)
    {
        // Rz(a) * v
        float c = cosf(a), s = sinf(a);
        return f3(
            c * v.x - s * v.y,
            s * v.x + c * v.y,
            v.z
        );
    }


    // ---------- rotate point about coordinate axis through origin ----------
    // 数值上等同于 vector 旋转
    // 语义上用于强调“这是点而不是方向”
    __host__ __device__ __forceinline__
        float3 f3_rotx_p(float3 p, float a) { return f3_rotx(p, a); }

    __host__ __device__ __forceinline__
        float3 f3_roty_p(float3 p, float a) { return f3_roty(p, a); }

    __host__ __device__ __forceinline__
        float3 f3_rotz_p(float3 p, float a) { return f3_rotz(p, a); }


    // ---------- rotate point about coordinate axis passing through p0 ----------
    // 数学公式： p' = p0 + R * (p - p0)
    __host__ __device__ __forceinline__
        float3 f3_rotx_about(float3 p, float3 p0, float a)
    {
        return f3_add(p0, f3_rotx(f3_sub(p, p0), a));
    }

    __host__ __device__ __forceinline__
        float3 f3_roty_about(float3 p, float3 p0, float a)
    {
        return f3_add(p0, f3_roty(f3_sub(p, p0), a));
    }

    __host__ __device__ __forceinline__
        float3 f3_rotz_about(float3 p, float3 p0, float a)
    {
        return f3_add(p0, f3_rotz(f3_sub(p, p0), a));
    }


    // ============================================================
    // Rotations: arbitrary axis (Rodrigues)
    // ------------------------------------------------------------
    // 最通用的三维旋转表达：绕任意单位轴 k_unit
    // 用于平板倾角、标定修正、非理想几何
    // ============================================================

    // Rotate vector v around unit axis k_unit by angle a
    // Rodrigues' rotation formula:
    //   v' = v*cos(a) + (k×v)*sin(a) + k*(k·v)*(1-cos(a))
    __host__ __device__ __forceinline__
        float3 f3_rot_axis(float3 v, float3 k_unit, float a)
    {
        float c = cosf(a), s = sinf(a);
        float3 kv = f3_cross(k_unit, v);
        float  d = f3_dot(k_unit, v);

        return f3_add(
            f3_add(f3_mul(v, c), f3_mul(kv, s)),
            f3_mul(k_unit, d * (1.0f - c))
        );
    }

    // Rotate point p about axis defined by (p0 + t*k_unit)
    __host__ __device__ __forceinline__
        float3 f3_rot_axis_about(float3 p, float3 p0, float3 k_unit, float a)
    {
        return f3_add(p0, f3_rot_axis(f3_sub(p, p0), k_unit, a));
    }


    // ============================================================
    // Optional: small rigid transform (no heavy matrices)
    // ------------------------------------------------------------
    // 表示刚体变换： p' = R*p + t
    // R 用 3 个基向量表示（列向量形式）
    // ============================================================

    struct SRigidTf
    {
        // Rotation matrix columns in world coordinates
        float3 ex;   // local +X maps to
        float3 ey;   // local +Y maps to
        float3 ez;   // local +Z maps to
        float3 t;    // translation

        // Apply rotation only (vector)
        __host__ __device__ __forceinline__
            float3 apply_vec(float3 v) const
        {
            return f3_add(
                f3_add(f3_mul(ex, v.x), f3_mul(ey, v.y)),
                f3_mul(ez, v.z)
            );
        }

        // Apply full rigid transform (point)
        __host__ __device__ __forceinline__
            float3 apply_point(float3 p) const
        {
            return f3_add(apply_vec(p), t);
        }
    };


    // Build a rigid transform from axis-angle representation
    // k_unit must be normalized
    __host__ __device__ __forceinline__
        SRigidTf f3_rigid_from_axis_angle(float3 k_unit, float a, float3 t = f3(0, 0, 0))
    {
        SRigidTf tf;
        tf.ex = f3_rot_axis(f3(1, 0, 0), k_unit, a);
        tf.ey = f3_rot_axis(f3(0, 1, 0), k_unit, a);
        tf.ez = f3_rot_axis(f3(0, 0, 1), k_unit, a);
        tf.t = t;
        return tf;
    }


} // namespace YK
