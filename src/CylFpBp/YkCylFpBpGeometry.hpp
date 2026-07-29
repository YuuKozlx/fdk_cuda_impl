#pragma once

#include <cmath>
#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "Heli/YkHeliCTParams.h"

namespace YK::CylFpBp {

constexpr float kPi = 3.14159265358979323846f;

inline float4 point4(float3 value)
{ return make_float4(value.x, value.y, value.z, 0.f); }

inline float3 add3(float3 a, float3 b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline float3 scale3(float3 a, float scale)
{ return make_float3(a.x * scale, a.y * scale, a.z * scale); }

// 构造一般圆柱探测器几何。SDD 是焦点到主射线落点的距离，curvature_radius_mm
// 是圆柱半径，二者互相独立。R != SDD 时圆柱轴线不经过焦点：其横向位置为
// source + radial * (SDD - R)。du_mm 是圆弧上的物理像素弧长；若显式给定
// channel_angle_step_rad，则以该角度为准，物理弧长随 R 变化。
inline std::vector<SCylConeProjGeomVec> buildCylindricalArcGeometry(
    const SHeliCTParam& p, float curvature_radius_mm,
    float channel_angle_step_rad = 0.f)
{
    if (!(curvature_radius_mm > 0.f)) return {};
    std::vector<SCylConeProjGeomVec> geometry(p.angle_list.size());
    const float principal_u = 0.5f * (p.iPU - 1) - p.offsetU_mm / p.du_mm;
    const float principal_v = 0.5f * (p.iPV - 1) - p.offsetV_mm / p.dv_mm;
    const float angle_step = channel_angle_step_rad > 0.f
        ? channel_angle_step_rad
        : p.du_mm / curvature_radius_mm;
    const float arc_pixel = curvature_radius_mm * angle_step;

    for (size_t i = 0; i < p.angle_list.size(); ++i) {
        const float theta = p.angle_list[i];
        const float z = p.start_z_mm + p.pitch_mm * theta / (2.f * kPi);
        const float3 source = make_float3(p.SID * sinf(theta),
            -p.SID * cosf(theta), z);
        const float3 radial = make_float3(-sinf(theta), cosf(theta), 0.f);
        const float3 tangent = make_float3(cosf(theta), sinf(theta), 0.f);
        // detector_principal 是圆柱面的中心通道点，与源点相距 SDD。
        const float3 center = add3(source, scale3(radial, p.SDD));
        geometry[i] = {
            point4(source), point4(center), point4(scale3(tangent, arc_pixel)),
            make_float4(0.f, 0.f, p.dv_mm, 0.f),
            make_float4(theta, 0.f, 0.f, 0.f),
            curvature_radius_mm, principal_u, principal_v, 0.f
        };
    }
    return geometry;
}

// FreeCT 原生等角弧面：圆柱轴线经过焦点，R=SDD，每个通道法线均指向焦点。
// 保留旧函数签名以维持全部现有调用的几何与像素采样不变。
inline std::vector<SCylConeProjGeomVec> buildFreeCtArcGeometry(
    const SHeliCTParam& p, float channel_angle_step_rad = 0.f)
{
    return buildCylindricalArcGeometry(p, p.SDD, channel_angle_step_rad);
}

} // namespace YK::CylFpBp
