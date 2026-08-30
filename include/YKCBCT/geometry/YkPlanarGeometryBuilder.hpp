#pragma once

#include <cmath>
#include <functional>
#include <vector>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK {

// 固定平面探测器几何。探测器位于 y=detector_y_mm，U/V 分别沿世界 X/Z；
// source_position 是唯一的焦点轨迹来源，angle 仅保存采集顺序和积分元数据。
inline bool buildPlanarProjectionGeometry(
    const std::vector<float>& angles_rad, int channels, int rows,
    float channel_size_mm, float row_size_mm, float detector_y_mm,
    const std::function<float3(float)>& source_position,
    std::vector<SConeProjGeomVec>& geometry)
{
    if (angles_rad.empty() || channels <= 0 || rows <= 0 ||
        !(channel_size_mm > 0.f) || !(row_size_mm > 0.f) ||
        !std::isfinite(detector_y_mm) || !source_position) return false;
    const float3 du = make_float3(channel_size_mm, 0.f, 0.f);
    const float3 dv = make_float3(0.f, 0.f, row_size_mm);
    const float cu = 0.5f * (channels - 1), cv = 0.5f * (rows - 1);
    const float3 start = make_float3(-du.x * cu, detector_y_mm, -dv.z * cv);
    geometry.resize(angles_rad.size());
    for (size_t i = 0; i < angles_rad.size(); ++i) {
        const float angle = angles_rad[i];
        const float3 source = source_position(angle);
        if (!std::isfinite(angle) || !std::isfinite(source.x) ||
            !std::isfinite(source.y) || !std::isfinite(source.z)) {
            geometry.clear();
            return false;
        }
        geometry[i] = {make_float4(source.x, source.y, source.z, 0.f),
            make_float4(start.x, start.y, start.z, 0.f),
            make_float4(du.x, du.y, du.z, 0.f),
            make_float4(dv.x, dv.y, dv.z, 0.f),
            make_float4(angle, 0.f, 0.f, 0.f)};
    }
    return true;
}

inline bool buildPlanarRotatingSourceGeometry(
    const std::vector<float>& angles_rad, int channels, int rows,
    float channel_size_mm, float row_size_mm, float sid_mm,
    float detector_y_mm, float source_lateral_mm,
    std::vector<SConeProjGeomVec>& geometry)
{
    return buildPlanarProjectionGeometry(angles_rad, channels, rows,
        channel_size_mm, row_size_mm, detector_y_mm,
        [=](float angle) {
            return make_float3(source_lateral_mm * std::cos(angle), -sid_mm,
                -source_lateral_mm * std::sin(angle));
        }, geometry);
}

inline bool buildPlanarEllipticSourceGeometry(
    const std::vector<float>& angles_rad, int channels, int rows,
    float channel_size_mm, float row_size_mm, float sid_mm,
    float detector_y_mm, float axis_x_mm, float axis_z_mm,
    std::vector<SConeProjGeomVec>& geometry)
{
    return buildPlanarProjectionGeometry(angles_rad, channels, rows,
        channel_size_mm, row_size_mm, detector_y_mm,
        [=](float angle) {
            return make_float3(axis_x_mm * std::cos(angle), -sid_mm,
                -axis_z_mm * std::sin(angle));
        }, geometry);
}

} // namespace YK
