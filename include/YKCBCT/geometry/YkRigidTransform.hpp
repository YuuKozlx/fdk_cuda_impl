#pragma once

#include <cmath>
#include <cuda_runtime.h>

namespace YK {

// 三维刚体坐标变换，旋转矩阵按行存储。
// 命名建议使用 targetFromSource：变换后的坐标 = R * 原坐标 + t。
struct SRigidTransform {
    float4 row0 = make_float4(1.f, 0.f, 0.f, 0.f);
    float4 row1 = make_float4(0.f, 1.f, 0.f, 0.f);
    float4 row2 = make_float4(0.f, 0.f, 1.f, 0.f);
    float3 translation = make_float3(0.f, 0.f, 0.f);

    static SRigidTransform identity() { return {}; }

    // 由单位旋转轴和弧度角构造刚体变换。unitAxis 必须已归一化；若调用方
    // 传入非单位轴，后续 isRigid() 会拒绝生成的非正交矩阵。
    static SRigidTransform fromAxisAngle(float3 unitAxis, float angleRad,
        float3 translation = make_float3(0.f, 0.f, 0.f))
    {
        const float c = std::cos(angleRad);
        const float s = std::sin(angleRad);
        const float d = 1.f - c;
        const float x = unitAxis.x;
        const float y = unitAxis.y;
        const float z = unitAxis.z;

        SRigidTransform result;
        result.row0 = make_float4(
            c + x * x * d, x * y * d - z * s, x * z * d + y * s, 0.f);
        result.row1 = make_float4(
            y * x * d + z * s, c + y * y * d, y * z * d - x * s, 0.f);
        result.row2 = make_float4(
            z * x * d - y * s, z * y * d + x * s, c + z * z * d, 0.f);
        result.translation = translation;
        return result;
    }

    bool isRigid(float tolerance = 1e-4f) const
    {
        const auto dot3 = [](const float4& a, const float4& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        };
        const float determinant =
            row0.x * (row1.y * row2.z - row1.z * row2.y) -
            row0.y * (row1.x * row2.z - row1.z * row2.x) +
            row0.z * (row1.x * row2.y - row1.y * row2.x);
        return std::isfinite(determinant) &&
            std::isfinite(translation.x) && std::isfinite(translation.y) &&
            std::isfinite(translation.z) &&
            std::fabs(dot3(row0, row0) - 1.f) <= tolerance &&
            std::fabs(dot3(row1, row1) - 1.f) <= tolerance &&
            std::fabs(dot3(row2, row2) - 1.f) <= tolerance &&
            std::fabs(dot3(row0, row1)) <= tolerance &&
            std::fabs(dot3(row0, row2)) <= tolerance &&
            std::fabs(dot3(row1, row2)) <= tolerance &&
            std::fabs(determinant - 1.f) <= tolerance;
    }

    // 构造绕任意坐标点 pivot 的旋转，并在旋转后追加 translation。
    // 体积中心不必位于原点；绕体积中心旋转时应把 SVolGeom::center 传入 pivot。
    static SRigidTransform aroundPoint(float4 r0, float4 r1, float4 r2,
        float3 pivot, float3 translation = make_float3(0.f, 0.f, 0.f))
    {
        SRigidTransform result;
        result.row0 = r0;
        result.row1 = r1;
        result.row2 = r2;
        const float3 rotated_pivot = result.transformVector(pivot);
        result.translation = make_float3(
            pivot.x + translation.x - rotated_pivot.x,
            pivot.y + translation.y - rotated_pivot.y,
            pivot.z + translation.z - rotated_pivot.z);
        return result;
    }

    // 轴角版本的枢轴旋转。体积中心不在原点时，直接把 volume.center 作为
    // pivot；函数自动生成 t = pivot + translation - R * pivot。
    static SRigidTransform aroundAxisPoint(float3 unitAxis, float angleRad,
        float3 pivot, float3 translation = make_float3(0.f, 0.f, 0.f))
    {
        const SRigidTransform rotation = fromAxisAngle(unitAxis, angleRad);
        return aroundPoint(rotation.row0, rotation.row1, rotation.row2,
            pivot, translation);
    }

    float3 transformVector(float3 value) const
    {
        return make_float3(
            row0.x * value.x + row0.y * value.y + row0.z * value.z,
            row1.x * value.x + row1.y * value.y + row1.z * value.z,
            row2.x * value.x + row2.y * value.y + row2.z * value.z);
    }

    float3 transformPoint(float3 point) const
    {
        const float3 rotated = transformVector(point);
        return make_float3(rotated.x + translation.x,
            rotated.y + translation.y, rotated.z + translation.z);
    }

    // 仅适用于正交旋转矩阵；刚体变换的逆旋转为 R 的转置。
    SRigidTransform inverse() const
    {
        SRigidTransform result;
        result.row0 = make_float4(row0.x, row1.x, row2.x, 0.f);
        result.row1 = make_float4(row0.y, row1.y, row2.y, 0.f);
        result.row2 = make_float4(row0.z, row1.z, row2.z, 0.f);
        const float3 inverse_translation = result.transformVector(translation);
        result.translation = make_float3(-inverse_translation.x,
            -inverse_translation.y, -inverse_translation.z);
        return result;
    }

    // 返回 this * source：先执行 source，再执行当前变换。
    SRigidTransform compose(const SRigidTransform& source) const
    {
        SRigidTransform result;
        const float3 source_column0 = make_float3(
            source.row0.x, source.row1.x, source.row2.x);
        const float3 source_column1 = make_float3(
            source.row0.y, source.row1.y, source.row2.y);
        const float3 source_column2 = make_float3(
            source.row0.z, source.row1.z, source.row2.z);
        result.row0 = make_float4(
            row0.x * source_column0.x + row0.y * source_column0.y + row0.z * source_column0.z,
            row0.x * source_column1.x + row0.y * source_column1.y + row0.z * source_column1.z,
            row0.x * source_column2.x + row0.y * source_column2.y + row0.z * source_column2.z, 0.f);
        result.row1 = make_float4(
            row1.x * source_column0.x + row1.y * source_column0.y + row1.z * source_column0.z,
            row1.x * source_column1.x + row1.y * source_column1.y + row1.z * source_column1.z,
            row1.x * source_column2.x + row1.y * source_column2.y + row1.z * source_column2.z, 0.f);
        result.row2 = make_float4(
            row2.x * source_column0.x + row2.y * source_column0.y + row2.z * source_column0.z,
            row2.x * source_column1.x + row2.y * source_column1.y + row2.z * source_column1.z,
            row2.x * source_column2.x + row2.y * source_column2.y + row2.z * source_column2.z, 0.f);
        result.translation = transformPoint(source.translation);
        return result;
    }
};

} // namespace YK
