#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkRigidTransform.hpp"

namespace YK {

// Vector cone-beam geometry for one view. Positions use millimetres; detU
// and detV are one-pixel steps. angle.x is the single angle source in radians.
struct SConeProjGeomVec {
    float4 src;
    float4 detS;
    float4 detU;
    float4 detV;
    float4 angle;
};

// ASTRA cyl_cone_vec 风格的逐视图圆柱探测器几何。detectorCenter 是探测器
// 数组中心在圆柱表面上的点；detectorU 是该点处一个通道的切向步长，
// detectorV 是一个探测器行的轴向步长。曲率半径和采集角度保存在
// viewParameters 中；采集角度仅用于排序和权重，不参与曲面姿态定义。
//
// 探测器尺寸 Nu/Nv 属于算子上下文，因此主数组坐标恒为
// ((Nu - 1) / 2, (Nv - 1) / 2)，不在逐视图结构中重复保存。
struct SCylConeProjGeomVec {
    float4 source{};
    float4 detectorCenter{};
    float4 detectorU{};
    float4 detectorV{};
    // x=采集角度(rad)，y=圆柱曲率半径(mm)，z/w 预留。
    // 请通过下方具名函数访问，避免调用端依赖分量布局。
    float4 viewParameters{};
};

static_assert(sizeof(SCylConeProjGeomVec) == 5 * sizeof(float4),
    "SCylConeProjGeomVec must remain a packed five-float4 ABI type");

__host__ __device__ inline float cylViewAngle(
    const SCylConeProjGeomVec& geometry)
{ return geometry.viewParameters.x; }

__host__ __device__ inline void setCylViewAngle(
    SCylConeProjGeomVec& geometry, float angle_rad)
{ geometry.viewParameters.x = angle_rad; }

__host__ __device__ inline float cylDetectorRadius(
    const SCylConeProjGeomVec& geometry)
{ return geometry.viewParameters.y; }

__host__ __device__ inline void setCylDetectorRadius(
    SCylConeProjGeomVec& geometry, float radius_mm)
{ geometry.viewParameters.y = radius_mm; }

// 由 ASTRA 风格圆柱 vector geometry 和探测器尺寸唯一派生的视图坐标架。
// detectorCenter 对应连续像素坐标 principalU/principalV；它们只由 Nu/Nv
// 决定，不再作为可与物理几何互相矛盾的外部输入。
struct SCylProjectionFrame {
    float3 detectorCenter{};
    float3 cylinderCenter{};
    float3 radialUnit{};
    float3 tangentUnit{};
    float3 axisUnit{};
    float radius_mm = 0.f;
    float principalU = 0.f;
    float principalV = 0.f;
    float channelStepRad = 0.f;
    float rowStepMm = 0.f;
};

inline bool deriveCylProjectionFrame(const SCylConeProjGeomVec& geometry,
    int detectorPixelsU, int detectorPixelsV, SCylProjectionFrame& result,
    float epsilon = 1e-6f)
{
    if (detectorPixelsU <= 0 || detectorPixelsV <= 0 ||
        !std::isfinite(cylDetectorRadius(geometry)) ||
        cylDetectorRadius(geometry) <= epsilon)
        return false;
    const auto vector3 = [](const float4& value) {
        return make_float3(value.x, value.y, value.z);
    };
    const auto finite3 = [](const float4& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
            std::isfinite(value.z);
    };
    const auto dot = [](const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    const auto cross = [](const float3& a, const float3& b) {
        return make_float3(a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    };
    const auto normalized = [&](const float3& value, float& length) {
        length = std::sqrt(dot(value, value));
        return length > epsilon ? make_float3(value.x / length,
            value.y / length, value.z / length) : make_float3(0.f, 0.f, 0.f);
    };
    if (!finite3(geometry.source) || !finite3(geometry.detectorCenter) ||
        !finite3(geometry.detectorU) || !finite3(geometry.detectorV))
        return false;

    const float3 source = vector3(geometry.source);
    result.detectorCenter = vector3(geometry.detectorCenter);
    float tangentLength = 0.f, rowLength = 0.f, radialLength = 0.f;
    result.tangentUnit = normalized(vector3(geometry.detectorU), tangentLength);
    result.axisUnit = normalized(vector3(geometry.detectorV), rowLength);
    // ASTRA cyl_cone_vec 的 U 是圆柱切向，V 是圆柱轴向；V x U 指向
    // 探测器表面的径向外侧。符号由 source -> detectorCenter 统一定向。
    float3 radial = normalized(cross(result.axisUnit, result.tangentUnit),
        radialLength);
    if (tangentLength <= epsilon || rowLength <= epsilon || radialLength <= epsilon)
        return false;
    const float3 sourceToDetector = make_float3(
        result.detectorCenter.x - source.x,
        result.detectorCenter.y - source.y,
        result.detectorCenter.z - source.z);
    if (dot(radial, sourceToDetector) < 0.f)
        radial = make_float3(-radial.x, -radial.y, -radial.z);
    constexpr float kOrthogonalTolerance = 1e-4f;
    if (std::fabs(dot(result.tangentUnit, result.axisUnit)) >
        kOrthogonalTolerance) return false;

    result.radialUnit = radial;
    result.radius_mm = cylDetectorRadius(geometry);
    result.principalU = 0.5f * static_cast<float>(detectorPixelsU - 1);
    result.principalV = 0.5f * static_cast<float>(detectorPixelsV - 1);
    result.channelStepRad = tangentLength / result.radius_mm;
    result.rowStepMm = rowLength;
    result.cylinderCenter = make_float3(
        result.detectorCenter.x - radial.x * result.radius_mm,
        result.detectorCenter.y - radial.y * result.radius_mm,
        result.detectorCenter.z - radial.z * result.radius_mm);
    return true;
}

// 由最终投影几何唯一派生的视图坐标架。公共输入不保存这些冗余量，防止
// detector offset/tilt 或刚体变换后出现多套互相矛盾的方向定义。
struct SProjectionFrame {
    float3 detectorCenter{}; // 探测器几何中心
    float3 centerRay{};      // 从源点指向 detectorCenter 的单位向量
    float3 detectorNormal{}; // 从源点朝向探测器平面的单位法向
    float3 principalPoint{}; // 源点到探测器平面的垂足
    float planeDistance = 0.f;
};

inline bool deriveProjectionFrame(const SConeProjGeomVec& geometry,
    int detectorPixelsU, int detectorPixelsV, SProjectionFrame& result,
    float epsilon = 1e-6f)
{
    if (detectorPixelsU <= 0 || detectorPixelsV <= 0) return false;

    const auto finite3 = [](const float4& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
            std::isfinite(value.z);
    };
    if (!finite3(geometry.src) || !finite3(geometry.detS) ||
        !finite3(geometry.detU) || !finite3(geometry.detV))
        return false;

    const float3 source = make_float3(
        geometry.src.x, geometry.src.y, geometry.src.z);
    const float3 detectorStart = make_float3(
        geometry.detS.x, geometry.detS.y, geometry.detS.z);
    const float3 detectorU = make_float3(
        geometry.detU.x, geometry.detU.y, geometry.detU.z);
    const float3 detectorV = make_float3(
        geometry.detV.x, geometry.detV.y, geometry.detV.z);
    const float centerU = 0.5f * static_cast<float>(detectorPixelsU - 1);
    const float centerV = 0.5f * static_cast<float>(detectorPixelsV - 1);
    result.detectorCenter = make_float3(
        detectorStart.x + centerU * detectorU.x + centerV * detectorV.x,
        detectorStart.y + centerU * detectorU.y + centerV * detectorV.y,
        detectorStart.z + centerU * detectorU.z + centerV * detectorV.z);

    const float3 centerDirection = make_float3(
        result.detectorCenter.x - source.x,
        result.detectorCenter.y - source.y,
        result.detectorCenter.z - source.z);
    const float centerLength2 = centerDirection.x * centerDirection.x +
        centerDirection.y * centerDirection.y + centerDirection.z * centerDirection.z;
    if (!std::isfinite(centerLength2) || centerLength2 <= epsilon * epsilon)
        return false;
    const float inverseCenterLength = 1.f / std::sqrt(centerLength2);
    result.centerRay = make_float3(
        centerDirection.x * inverseCenterLength,
        centerDirection.y * inverseCenterLength,
        centerDirection.z * inverseCenterLength);

    float3 normal = make_float3(
        detectorV.y * detectorU.z - detectorV.z * detectorU.y,
        detectorV.z * detectorU.x - detectorV.x * detectorU.z,
        detectorV.x * detectorU.y - detectorV.y * detectorU.x);
    const float normalLength2 = normal.x * normal.x + normal.y * normal.y +
        normal.z * normal.z;
    if (!std::isfinite(normalLength2) || normalLength2 <= epsilon * epsilon)
        return false;
    const float inverseNormalLength = 1.f / std::sqrt(normalLength2);
    normal.x *= inverseNormalLength;
    normal.y *= inverseNormalLength;
    normal.z *= inverseNormalLength;
    if (normal.x * centerDirection.x + normal.y * centerDirection.y +
        normal.z * centerDirection.z < 0.f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    result.detectorNormal = normal;

    const float3 detectorFromSource = make_float3(
        detectorStart.x - source.x,
        detectorStart.y - source.y,
        detectorStart.z - source.z);
    result.planeDistance = detectorFromSource.x * normal.x +
        detectorFromSource.y * normal.y + detectorFromSource.z * normal.z;
    if (!std::isfinite(result.planeDistance) || result.planeDistance <= epsilon)
        return false;
    result.principalPoint = make_float3(
        source.x + result.planeDistance * normal.x,
        source.y + result.planeDistance * normal.y,
        source.z + result.planeDistance * normal.z);
    return true;
}

inline SConeProjGeomVec transformProjectionGeometry(
    const SConeProjGeomVec& geometry, const SRigidTransform& targetFromSource)
{
    const auto point = [&](const float4& value) {
        const float3 transformed = targetFromSource.transformPoint(
            make_float3(value.x, value.y, value.z));
        return make_float4(transformed.x, transformed.y, transformed.z, value.w);
    };
    const auto vector = [&](const float4& value) {
        const float3 transformed = targetFromSource.transformVector(
            make_float3(value.x, value.y, value.z));
        return make_float4(transformed.x, transformed.y, transformed.z, value.w);
    };

    SConeProjGeomVec result = geometry;
    result.src = point(geometry.src);
    result.detS = point(geometry.detS);
    result.detU = vector(geometry.detU);
    result.detV = vector(geometry.detV);
    // angle.x 是采集顺序和冗余权重元数据，不是三维姿态的替代表示。
    return result;
}

} // namespace YK
