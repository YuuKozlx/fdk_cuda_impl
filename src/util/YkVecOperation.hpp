#pragma once
/**
 * YkVec.h �� YK ������ѧ��
 * ������������ helper_math.h��CUDA SDK��
 * ���ļ�ֻ�ṩ helper_math.h δ���ǵ���չ
 */
#include <cuda_runtime.h>
#include "helper_math.h"
#include <cmath>

namespace YK {

    using vector3 = float3;
    using point3 = float3;

    // ============================================================
    // scalar helpers
    // ============================================================

    YK_HD YK_FORCE_INLINE float yk_abs(float x) { return x < 0.f ? -x : x; }
    YK_HD YK_FORCE_INLINE float yk_clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
    YK_HD YK_FORCE_INLINE float yk_lerp(float a, float b, float t) { return a + (b - a) * t; }

    // ============================================================
    // float3 ��չ
    // ============================================================

    YK_HD YK_FORCE_INLINE float3 f3_normalize(float3 a, float eps = 1e-20f) {
        float l2 = dot(a, a);
        if (l2 < eps) return make_float3(0.f, 0.f, 0.f);
#if defined(__CUDA_ARCH__)
        return a * rsqrtf(l2);
#else
        return a * (1.f / std::sqrt(l2));
#endif
    }

    YK_HD YK_FORCE_INLINE bool f3_all_finite(float3 a) {
#if defined(__CUDA_ARCH__)
        return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
#else
        return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
#endif
    }

    YK_HD YK_FORCE_INLINE bool f3_near(float3 a, float3 b, float eps = 1e-6f) {
        float3 d = fabs(a - b);
        return d.x <= eps && d.y <= eps && d.z <= eps;
    }

    // ============================================================
    // float3 ��ת
    // ============================================================

    YK_HD YK_FORCE_INLINE float3 f3_rotx(float3 v, float a) {
        float c = cosf(a), s = sinf(a);
        return make_float3(v.x, c * v.y - s * v.z, s * v.y + c * v.z);
    }

    YK_HD YK_FORCE_INLINE float3 f3_roty(float3 v, float a) {
        float c = cosf(a), s = sinf(a);
        return make_float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
    }

    YK_HD YK_FORCE_INLINE float3 f3_rotz(float3 v, float a) {
        float c = cosf(a), s = sinf(a);
        return make_float3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
    }

    YK_HD YK_FORCE_INLINE float3 f3_rot_axis(float3 v, float3 k, float a) {
        float c = cosf(a), s = sinf(a);
        return v * c + cross(k, v) * s + k * dot(k, v) * (1.f - c);
    }

    YK_HD YK_FORCE_INLINE float3 f3_rotx_about(float3 p, float3 p0, float a) { return p0 + f3_rotx(p - p0, a); }
    YK_HD YK_FORCE_INLINE float3 f3_roty_about(float3 p, float3 p0, float a) { return p0 + f3_roty(p - p0, a); }
    YK_HD YK_FORCE_INLINE float3 f3_rotz_about(float3 p, float3 p0, float a) { return p0 + f3_rotz(p - p0, a); }
    YK_HD YK_FORCE_INLINE float3 f3_rot_axis_about(float3 p, float3 p0, float3 k, float a) { return p0 + f3_rot_axis(p - p0, k, a); }

    // ============================================================
    // float3 ����
    // ============================================================

    YK_HD YK_FORCE_INLINE float3 f3_reflect_plane_origin(float3 p, float3 n) { return p - 2.f * dot(p, n) * n; }
    YK_HD YK_FORCE_INLINE float3 f3_reflect_plane(float3 p, float3 p0, float3 n) { return p0 + f3_reflect_plane_origin(p - p0, n); }
    YK_HD YK_FORCE_INLINE float3 f3_reflect_axis_origin(float3 p, float3 u) { return 2.f * dot(p, u) * u - p; }
    YK_HD YK_FORCE_INLINE float3 f3_reflect_axis(float3 p, float3 p0, float3 u) { return p0 + f3_reflect_axis_origin(p - p0, u); }

    // ============================================================
    // float4 ��չ��w ��ԶΪ 0����������ֱ���� helper_math��
    // ============================================================

    // ��ת��w �̶�Ϊ 0
    YK_HD YK_FORCE_INLINE float4 f3_to_f4(float3 v) { return make_float4(v.x, v.y, v.z, 0.f); }
    YK_HD YK_FORCE_INLINE float3 f4_to_f3(float4 v) { return make_float3(v.x, v.y, v.z); }

    // cross��helper_math û�� float4 �汾��
    YK_HD YK_FORCE_INLINE float4 f4_cross(float4 a, float4 b) {
        return make_float4(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
            0.f
        );
    }

    // normalize �� eps ������helper_math �ޱ�����
    YK_HD YK_FORCE_INLINE float4 f4_normalize(float4 a, float eps = 1e-20f) {
        float l2 = dot(a, a);  // w=0���ȼ��� xyz dot
        if (l2 < eps) return make_float4(0.f, 0.f, 0.f, 0.f);
#if defined(__CUDA_ARCH__)
        return a * rsqrtf(l2);
#else
        return a * (1.f / std::sqrt(l2));
#endif
    }

    YK_HD YK_FORCE_INLINE bool f4_all_finite(float4 a) {
#if defined(__CUDA_ARCH__)
        return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
#else
        return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
#endif
    }

    YK_HD YK_FORCE_INLINE bool f4_near(float4 a, float4 b, float eps = 1e-6f) {
        float4 d = fabs(a - b);
        return d.x <= eps && d.y <= eps && d.z <= eps;
    }

    // ��ת��w ��� 0
    YK_HD YK_FORCE_INLINE float4 f4_rotx(float4 v, float a) { return f3_to_f4(f3_rotx(f4_to_f3(v), a)); }
    YK_HD YK_FORCE_INLINE float4 f4_roty(float4 v, float a) { return f3_to_f4(f3_roty(f4_to_f3(v), a)); }
    YK_HD YK_FORCE_INLINE float4 f4_rotz(float4 v, float a) { return f3_to_f4(f3_rotz(f4_to_f3(v), a)); }
    YK_HD YK_FORCE_INLINE float4 f4_rot_axis(float4 v, float4 k, float a) { return f3_to_f4(f3_rot_axis(f4_to_f3(v), f4_to_f3(k), a)); }

    // ============================================================
    // SRigidTf
    // ============================================================

    struct SRigidTf {
        float3 ex, ey, ez;
        float3 t;

        YK_HD YK_FORCE_INLINE float3 apply_vec(float3 v) const { return ex * v.x + ey * v.y + ez * v.z; }
        YK_HD YK_FORCE_INLINE float3 apply_point(float3 p) const { return apply_vec(p) + t; }
    };

    YK_HD YK_FORCE_INLINE SRigidTf f3_rigid_from_axis_angle(float3 k, float a, float3 t = make_float3(0, 0, 0)) {
        SRigidTf tf;
        tf.ex = f3_rot_axis(make_float3(1, 0, 0), k, a);
        tf.ey = f3_rot_axis(make_float3(0, 1, 0), k, a);
        tf.ez = f3_rot_axis(make_float3(0, 0, 1), k, a);
        tf.t = t;
        return tf;
    }

    // ============================================================
    // SMat4f
    // ============================================================

    struct SMat4f {
        float m[4][4];

        YK_HD SMat4f() {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    m[i][j] = (i == j ? 1.f : 0.f);
        }

        YK_HD YK_FORCE_INLINE float3 apply_point(float3 p) const {
            return make_float3(
                m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
                m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
                m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]
            );
        }

        YK_HD YK_FORCE_INLINE float3 apply_vec(float3 v) const {
            return make_float3(
                m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
            );
        }

        YK_HD YK_FORCE_INLINE float4 operator*(float4 v) const {
            return make_float4(
                m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
            );
        }

        YK_HD SMat4f operator*(const SMat4f& B) const {
            SMat4f R;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    R.m[i][j] = 0.f;
                    for (int k = 0; k < 4; k++)
                        R.m[i][j] += m[i][k] * B.m[k][j];
                }
            return R;
        }

        YK_HD SMat4f inverse_rigid() const {
            SMat4f inv;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    inv.m[i][j] = m[j][i];
            inv.m[0][3] = -(inv.m[0][0] * m[0][3] + inv.m[0][1] * m[1][3] + inv.m[0][2] * m[2][3]);
            inv.m[1][3] = -(inv.m[1][0] * m[0][3] + inv.m[1][1] * m[1][3] + inv.m[1][2] * m[2][3]);
            inv.m[2][3] = -(inv.m[2][0] * m[0][3] + inv.m[2][1] * m[1][3] + inv.m[2][2] * m[2][3]);
            inv.m[3][0] = inv.m[3][1] = inv.m[3][2] = 0.f;
            inv.m[3][3] = 1.f;
            return inv;
        }

        YK_HD static SMat4f translate(float3 t) {
            SMat4f tf;
            tf.m[0][3] = t.x; tf.m[1][3] = t.y; tf.m[2][3] = t.z;
            return tf;
        }

        YK_HD static SMat4f rot_x(float a) {
            SMat4f tf; float c = cosf(a), s = sinf(a);
            tf.m[1][1] = c; tf.m[1][2] = -s;
            tf.m[2][1] = s; tf.m[2][2] = c;
            return tf;
        }

        YK_HD static SMat4f rot_y(float a) {
            SMat4f tf; float c = cosf(a), s = sinf(a);
            tf.m[0][0] = c; tf.m[0][2] = s;
            tf.m[2][0] = -s; tf.m[2][2] = c;
            return tf;
        }

        YK_HD static SMat4f rot_z(float a) {
            SMat4f tf; float c = cosf(a), s = sinf(a);
            tf.m[0][0] = c; tf.m[0][1] = -s;
            tf.m[1][0] = s; tf.m[1][1] = c;
            return tf;
        }

        YK_HD static SMat4f from_axis_angle(float3 k, float a, float3 t = make_float3(0, 0, 0)) {
            SMat4f R;
            float c = cosf(a), s = sinf(a), d = 1.f - c;
            float x = k.x, y = k.y, z = k.z;
            R.m[0][0] = c + x * x * d;    R.m[0][1] = x * y * d - z * s;  R.m[0][2] = x * z * d + y * s;  R.m[0][3] = t.x;
            R.m[1][0] = y * x * d + z * s;  R.m[1][1] = c + y * y * d;    R.m[1][2] = y * z * d - x * s;  R.m[1][3] = t.y;
            R.m[2][0] = z * x * d - y * s;  R.m[2][1] = z * y * d + x * s;  R.m[2][2] = c + z * z * d;    R.m[2][3] = t.z;
            R.m[3][0] = 0.f;        R.m[3][1] = 0.f;         R.m[3][2] = 0.f;         R.m[3][3] = 1.f;
            return R;
        }

        YK_HD static SMat4f from_rigid(const SRigidTf& tf) {
            SMat4f m;
            m.m[0][0] = tf.ex.x; m.m[0][1] = tf.ey.x; m.m[0][2] = tf.ez.x; m.m[0][3] = tf.t.x;
            m.m[1][0] = tf.ex.y; m.m[1][1] = tf.ey.y; m.m[1][2] = tf.ez.y; m.m[1][3] = tf.t.y;
            m.m[2][0] = tf.ex.z; m.m[2][1] = tf.ey.z; m.m[2][2] = tf.ez.z; m.m[2][3] = tf.t.z;
            m.m[3][0] = 0.f;     m.m[3][1] = 0.f;     m.m[3][2] = 0.f;     m.m[3][3] = 1.f;
            return m;
        }
    };

    // ============================================================
    // ���ݲ�
    // ============================================================

    // float3
    YK_HD YK_FORCE_INLINE float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
    YK_HD YK_FORCE_INLINE float3 f3_add(float3 a, float3 b) { return a + b; }
    YK_HD YK_FORCE_INLINE float3 f3_sub(float3 a, float3 b) { return a - b; }
    YK_HD YK_FORCE_INLINE float3 f3_scale(float3 a, float t) { return a * t; }
    YK_HD YK_FORCE_INLINE float3 f3_mul(float t, float3 a) { return t * a; }
    YK_HD YK_FORCE_INLINE float3 f3_div(float3 a, float t) { return a / t; }
    YK_HD YK_FORCE_INLINE float3 f3_neg(float3 a) { return -a; }
    YK_HD YK_FORCE_INLINE float3 f3_mul_comp(float3 a, float3 b) { return a * b; }
    YK_HD YK_FORCE_INLINE float3 f3_div_comp(float3 a, float3 b) { return a / b; }
    YK_HD YK_FORCE_INLINE float3 f3_mad(float3 a, float t, float3 b) { return a * t + b; }
    YK_HD YK_FORCE_INLINE float  f3_dot(float3 a, float3 b) { return dot(a, b); }
    YK_HD YK_FORCE_INLINE float3 f3_cross(float3 a, float3 b) { return cross(a, b); }
    YK_HD YK_FORCE_INLINE float  f3_len2(float3 a) { return dot(a, a); }
    YK_HD YK_FORCE_INLINE float  f3_len(float3 a) { return length(a); }
    YK_HD YK_FORCE_INLINE float3 f3_min(float3 a, float3 b) { return fminf(a, b); }
    YK_HD YK_FORCE_INLINE float3 f3_max(float3 a, float3 b) { return fmaxf(a, b); }
    YK_HD YK_FORCE_INLINE float3 f3_abs(float3 a) { return fabs(a); }
    YK_HD YK_FORCE_INLINE float3 f3_clamp(float3 a, float lo, float hi) { return clamp(a, lo, hi); }
    YK_HD YK_FORCE_INLINE float3 f3_lerp(float3 a, float3 b, float t) { return lerp(a, b, t); }

    // float4
    YK_HD YK_FORCE_INLINE float4 f4(float x, float y, float z, float w = 0.f) { return make_float4(x, y, z, w); }
    YK_HD YK_FORCE_INLINE float4 f4_add(float4 a, float4 b) { return a + b; }
    YK_HD YK_FORCE_INLINE float4 f4_sub(float4 a, float4 b) { return a - b; }
    YK_HD YK_FORCE_INLINE float4 f4_scale(float4 a, float t) { return a * t; }
    YK_HD YK_FORCE_INLINE float4 f4_mul(float t, float4 a) { return t * a; }
    YK_HD YK_FORCE_INLINE float4 f4_div(float4 a, float t) { return a / t; }
    YK_HD YK_FORCE_INLINE float4 f4_neg(float4 a) { return -a; }
    YK_HD YK_FORCE_INLINE float4 f4_mul_comp(float4 a, float4 b) { return a * b; }
    YK_HD YK_FORCE_INLINE float4 f4_div_comp(float4 a, float4 b) { return a / b; }
    YK_HD YK_FORCE_INLINE float4 f4_mad(float4 a, float t, float4 b) { return a * t + b; }
    YK_HD YK_FORCE_INLINE float4 f4_min(float4 a, float4 b) { return fminf(a, b); }
    YK_HD YK_FORCE_INLINE float4 f4_max(float4 a, float4 b) { return fmaxf(a, b); }
    YK_HD YK_FORCE_INLINE float4 f4_abs(float4 a) { return fabs(a); }
    YK_HD YK_FORCE_INLINE float4 f4_clamp(float4 a, float lo, float hi) { return clamp(a, lo, hi); }
    YK_HD YK_FORCE_INLINE float4 f4_lerp(float4 a, float4 b, float t) { return lerp(a, b, t); }

} // namespace YK