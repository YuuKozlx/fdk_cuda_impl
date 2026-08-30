#pragma once

#include <cuda_runtime.h>

#include "global/YkGlobals.h"

namespace YK::Fdk::detail {

// 论文 C-FDK（Li et al., 2011）使用的理想圆轨迹参数。
// 原始投影坐标 (a,b) 位于经过旋转中心的虚拟平板上；输入纹理仍保存真实
// 平板投影，因此 rebin kernel 会用 SDD/SID 将 (a,b) 映射回真实探测器。
struct SCurveFilteredFdkGeometry {
    int input_u = 0;
    int input_v = 0;
    int views = 0;
    int output_t = 0;
    int output_c = 0;
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
    float input_du_mm = 0.f;
    float input_dv_mm = 0.f;
    float beta0_rad = 0.f;
    float signed_dtheta_rad = 0.f;
    float dtheta_rad = 0.f;
    float t_min_mm = 0.f;
    float dt_mm = 0.f;
    float c_min_mm = 0.f;
    float dc_mm = 0.f;
    float c0_mm = 0.f;
    float s0_mm = 0.f;
    float bm_mm = 0.f;
};

struct SCurveFilteredFdkVolume {
    SVolGeom geometry{};
};

// 由论文式 (25)、(27) 定义的三段曲线反解式 (35)。论文的前向曲线以
// |c| 插值，因此反解时也必须先用 |z| 求 c 的幅值，再恢复 z 的符号。
// 返回的 branch 分别为 0/1/2；超出可重建双锥区域时返回 false。
__host__ __device__ inline bool mapCurveFilteredFdkBackprojectionC(
    const SCurveFilteredFdkGeometry& g, float t, float v, float z,
    float& c, int* branch = nullptr)
{
    const float r2 = g.sid_mm * g.sid_mm;
    const float q2 = r2 - t * t;
    if (q2 <= 0.f) return false;
    const float q = sqrtf(q2);
    const float q_plus_v = q + v;
    const float sc = 2.f * q - g.sid_mm;
    if (q_plus_v <= 0.f || sc <= 0.f) return false;

    const float abs_z = fabsf(z);
    const float limit0 = q_plus_v / sc * g.c0_mm;
    const float limit1 = q_plus_v / q * g.s0_mm;
    const float limit2 = q * q_plus_v / r2 * g.bm_mm;
    // 三个区间在论文中都是闭区间。给边界一个相对 ULP 量级的容差，
    // 避免由 sqrt/div 浮点舍入把理论上的 c0/s0/bm 边界拒绝掉。
    const float boundary_eps = 8e-6f * fmaxf(limit2, 1.f);
    if (abs_z <= limit0 + boundary_eps) {
        c = z * sc / q_plus_v;
        if (branch) *branch = 0;
        return true;
    }
    if (abs_z <= limit1 + boundary_eps) {
        const float numerator = sc * g.s0_mm - g.c0_mm * q;
        const float denominator = q_plus_v * (g.s0_mm - g.c0_mm) +
            abs_z * (q - g.sid_mm);
        if (denominator <= 0.f) return false;
        c = z * numerator / denominator;
        if (branch) *branch = 1;
        return true;
    }
    if (abs_z <= limit2 + boundary_eps) {
        const float numerator = (g.bm_mm - g.s0_mm) * q2 -
            g.s0_mm * t * t;
        const float denominator = q * q_plus_v *
            (g.bm_mm - g.s0_mm) - abs_z * t * t;
        if (denominator <= 0.f) return false;
        c = z * numerator / denominator;
        if (branch) *branch = 2;
        return true;
    }
    return false;
}

} // namespace YK::Fdk::detail
