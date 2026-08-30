#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "global/YkGlobals.h"

namespace YK::CylFpBp {

struct IterativeWeightConfig {
    bool view_quadrature = true;
    bool ray_coverage = true;
};

namespace detail {

inline float3 weightAdd(float3 a, float3 b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3 weightSub(float3 a, float3 b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3 weightScale(float3 a, float s)
{ return make_float3(a.x * s, a.y * s, a.z * s); }
inline float weightDot(float3 a, float3 b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

inline bool rayIntersectsVolume(float3 source, float3 detector,
    const SVolGeom& volume)
{
    const float3 direction = weightSub(detector, source);
    const float3 origin = volume.origin();
    const float lower[3] = {
        origin.x - 0.5f * volume.vox_x,
        origin.y - 0.5f * volume.vox_y,
        origin.z - 0.5f * volume.vox_z
    };
    const float upper[3] = {
        origin.x + (volume.Nx - 0.5f) * volume.vox_x,
        origin.y + (volume.Ny - 0.5f) * volume.vox_y,
        origin.z + (volume.Nz - 0.5f) * volume.vox_z
    };
    const float s[3] = { source.x, source.y, source.z };
    const float d[3] = { direction.x, direction.y, direction.z };
    float t_min = 0.f, t_max = 1.f;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(d[axis]) < 1e-12f) {
            if (s[axis] < lower[axis] || s[axis] > upper[axis]) return false;
            continue;
        }
        float a = (lower[axis] - s[axis]) / d[axis];
        float b = (upper[axis] - s[axis]) / d[axis];
        if (a > b) std::swap(a, b);
        t_min = std::max(t_min, a);
        t_max = std::min(t_max, b);
        if (t_max <= t_min) return false;
    }
    return t_max > 0.f;
}

inline float3 detectorSample(const SCylConeProjGeomVec& geometry,
    const SCylProjectionFrame& frame, int channels, int rows,
    int channel, int row)
{
    const float delta = (static_cast<float>(channel) - frame.principalU) *
        frame.channelStepRad;
    const float3 surface = weightAdd(frame.cylinderCenter,
        weightAdd(weightScale(frame.radialUnit, frame.radius_mm * std::cos(delta)),
            weightScale(frame.tangentUnit, frame.radius_mm * std::sin(delta))));
    return weightAdd(surface, weightScale(frame.axisUnit,
        (static_cast<float>(row) - frame.principalV) * frame.rowStepMm));
}

} // namespace detail

// 构造投影样本权重 W(view,row,channel)。逐视图梯形积分权重按均值
// 归一化，因此不会改变现有 relaxation 的整体量级；ray_coverage 将不与
// 重建体积相交的圆柱射线置零。随后由 B(W*1) 形成空间覆盖归一化。
inline std::vector<float> buildIterativeProjectionWeights(
    const SVolGeom& volume, int channels, int rows,
    const std::vector<SCylConeProjGeomVec>& geometry,
    const IterativeWeightConfig& config)
{
    if (channels <= 0 || rows <= 0 || geometry.empty()) return {};
    const size_t view_elements = static_cast<size_t>(channels) * rows;
    std::vector<float> result(view_elements * geometry.size(), 1.f);
    std::vector<float> view_weights(geometry.size(), 1.f);

    if (config.view_quadrature && geometry.size() > 1) {
        std::vector<float> angles(geometry.size());
        angles[0] = cylViewAngle(geometry[0]);
        constexpr float two_pi = 6.28318530717958647692f;
        for (size_t i = 1; i < geometry.size(); ++i) {
            float angle = cylViewAngle(geometry[i]);
            float delta = angle - angles[i - 1];
            while (delta <= -3.14159265358979323846f) delta += two_pi;
            while (delta > 3.14159265358979323846f) delta -= two_pi;
            if (std::fabs(delta) <= 1e-8f) return {};
            angles[i] = angles[i - 1] + delta;
        }
        view_weights.front() = std::fabs(angles[1] - angles[0]);
        view_weights.back() = std::fabs(angles.back() - angles[angles.size() - 2]);
        for (size_t i = 1; i + 1 < angles.size(); ++i)
            view_weights[i] = 0.5f * std::fabs(angles[i + 1] - angles[i - 1]);
        float mean = 0.f;
        for (float value : view_weights) mean += value;
        mean /= static_cast<float>(view_weights.size());
        if (!(mean > 0.f) || !std::isfinite(mean)) return {};
        for (float& value : view_weights) value /= mean;
    }

    for (size_t view = 0; view < geometry.size(); ++view) {
        SCylProjectionFrame frame{};
        if (!deriveCylProjectionFrame(geometry[view], channels, rows, frame))
            return {};
        const float3 source = make_float3(geometry[view].source.x,
            geometry[view].source.y, geometry[view].source.z);
        for (int row = 0; row < rows; ++row) {
            for (int channel = 0; channel < channels; ++channel) {
                const size_t index = view * view_elements +
                    static_cast<size_t>(row) * channels + channel;
                if (config.ray_coverage) {
                    const float3 detector = detail::detectorSample(geometry[view],
                        frame, channels, rows, channel, row);
                    if (!detail::rayIntersectsVolume(source, detector, volume)) {
                        result[index] = 0.f;
                        continue;
                    }
                }
                result[index] = view_weights[view];
            }
        }
    }
    return result;
}

} // namespace YK::CylFpBp
