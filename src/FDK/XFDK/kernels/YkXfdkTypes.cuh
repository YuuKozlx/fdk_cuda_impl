#pragma once

#include <cuda_runtime.h>

#include "global/YkGlobals.h"

namespace YK::Fdk::detail {

inline constexpr float kXfdkPi = 3.14159265358979323846f;

// Grimmer 等（2009）xFDK 的理想圆轨迹参数。输入是平板探测器上的
// [alpha][v][u] 投影，重排后为 [vartheta][gamma][xi]，其中 xi 使用 mm，
// gamma 使用弧度。该结构只描述算法工作区，不替代公共逐视图 geometry。
struct SXfdkGeometry {
    int input_u = 0;
    int input_v = 0;
    int views = 0;
    int output_xi = 0;
    int output_gamma = 0;
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float input_du_mm = 0.f;
    float input_dv_mm = 0.f;
    float beta_max_rad = 0.f;
    float gamma_max_rad = 0.f;
    float xi_min_mm = 0.f;
    float dxi_mm = 0.f;
    float gamma_min_rad = 0.f;
    float dgamma_rad = 0.f;
    float alpha0_rad = 0.f;
    float signed_dtheta_rad = 0.f;
    float dtheta_rad = 0.f;
    float reconstruction_radius_mm = 0.f; // R_M
    float c_cot_gamma = 0.f;              // cot(gamma_max)
    float delta_r_mm = 0.f;               // R_M^2/(2 R_F)
};

// 一次 xFDK 计算所消费的重排角度区间和原始投影窗口。source_begin 使用
// 连续逻辑视图编号，允许小于 0 或大于总视图数；host 侧在装入窗口时按
// 完整圆扫描周期取模。kernel 因而只访问紧凑的局部纹理，不保存全量投影。
struct SXfdkChunk {
    int theta_begin = 0;
    int theta_count = 0;
    int source_begin = 0;
    int source_count = 0;
};

// 原文 Eq. (4)-(5) 的覆盖角余量。可重建时返回 [0, pi/2]，低于 180 度
// 数据支持时返回负值。z 的正负只通过 |z| 进入，因为论文先推导 z>0，
// 再利用上下对称性扩展到整个体积。
__host__ __device__ inline float xfdkCoverageHalfRange(
    const SXfdkGeometry& g, float r, float z)
{
    const float az = fabsf(z);
    const float cz = g.c_cot_gamma * az;
    const float dem = g.sid_mm - cz;
    if (!(r >= 0.f) || r >= g.sid_mm) return -1.f;
    const float last_cz = (g.sid_mm * g.sid_mm - r * r) / g.sid_mm;
    const float tolerance = 8e-6f * g.sid_mm;
    if (cz > last_cz + tolerance) return -1.f;
    if (r <= dem) return 0.5f * kXfdkPi;
    const float d2 = r * r - dem * dem;
    if (d2 <= 0.f) return 0.5f * kXfdkPi;
    const float first = asinf(fminf(fmaxf(dem / r, -1.f), 1.f));
    const float denominator = cz;
    if (!(denominator > 1e-12f)) return -1.f;
    const float second = atanf(sqrtf(d2) / denominator);
    return fmaxf(0.f, fminf(0.5f * kXfdkPi, first - second));
}

// 原文 Eq. (7) 的平滑函数 s(t)=sin(pi*t/2)。这里不截断 t，调用方只在
// 对应过渡区间内使用。
__host__ __device__ inline float xfdkSmooth(float t)
{
    return sinf(0.5f * kXfdkPi * t);
}

// 原文“体素相关部分扫描权重”分段式。注意此函数的幅值是 0..2，
// 反投影主公式另有 1/2 系数，不能在这里再次乘 1/2。
__host__ __device__ inline float xfdkPartialWeight(
    float vartheta, float phi, float delta)
{
    if (delta < 0.f) return 0.f;
    float relative = vartheta - phi;
    relative = fmodf(relative + kXfdkPi, 2.f * kXfdkPi);
    if (relative < 0.f) relative += 2.f * kXfdkPi;
    relative -= kXfdkPi;
    if (delta <= 1e-7f)
        return relative >= -0.5f * kXfdkPi &&
            relative < 0.5f * kXfdkPi ? 2.f : 0.f;
    if (relative < -0.5f * kXfdkPi - delta ||
        relative >= 0.5f * kXfdkPi + delta) return 0.f;
    if (relative < -0.5f * kXfdkPi + delta)
        return 1.f + xfdkSmooth((relative + 0.5f * kXfdkPi) / delta);
    if (relative < 0.5f * kXfdkPi - delta) return 2.f;
    if (relative < 0.5f * kXfdkPi + delta)
        return 1.f - xfdkSmooth((relative - 0.5f * kXfdkPi) / delta);
    return 0.f;
}

// 原文径向过渡系数 w_T，取值范围为 [0,1]。
__host__ __device__ inline float xfdkTransitionWeight(
    const SXfdkGeometry& g, float r, float z)
{
    const float r0 = g.sid_mm - g.c_cot_gamma * fabsf(z) - g.delta_r_mm;
    if (g.delta_r_mm <= 1e-12f || r < r0 - g.delta_r_mm) return 0.f;
    if (r < r0 + g.delta_r_mm)
        return 0.5f * (1.f + xfdkSmooth((r - r0) / g.delta_r_mm));
    return 1.f;
}

// 原文最终复合权重 w_C。部分扫描权重幅值为 0..2，故这里仍不
// 引入 1/2；1/2 只出现在最终反投影积分系数中。
__host__ __device__ inline float xfdkCompositeWeight(
    const SXfdkGeometry& g, float r, float z, float vartheta, float phi)
{
    const float delta = xfdkCoverageHalfRange(g, r, z);
    if (delta < 0.f) return 0.f;
    const float wt = xfdkTransitionWeight(g, r, z);
    return (1.f - wt) + wt * xfdkPartialWeight(vartheta, phi, delta);
}

} // namespace YK::Fdk::detail
