#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"
#include "global/YkFdkKernelTypes.hpp"

namespace YK::NonStandardFpBp {

// Derive the quantities consumed by the analytic FDK stages directly from a
// per-view ConeVec pose. The reference ray is source -> reconstruction center;
// no standard circular detector pose is synthesized.
inline bool buildNonStandardFdkDerivedGeometry(int detector_u, int detector_v,
    float scan_range_rad, float3 reconstruction_center,
    const std::vector<SConeProjGeomVec>& geometry,
    std::vector<SFDKGeoParamPerView>& result)
{
    constexpr float pi = 3.14159265358979323846f;
    const auto dot = [](float3 a, float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    const auto add = [](float3 a, float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    };
    const auto sub = [](float3 a, float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    };
    const auto mul = [](float3 a, float value) {
        return make_float3(a.x * value, a.y * value, a.z * value);
    };
    const auto xyz = [](float4 a) { return make_float3(a.x, a.y, a.z); };
    const auto packed = [](float3 a) { return make_float4(a.x, a.y, a.z, 0.f); };

    const int views = static_cast<int>(geometry.size());
    if (detector_u <= 0 || detector_v <= 0 || views <= 0 ||
        !(scan_range_rad > 0.f)) return false;
    std::vector<float> theta(views);
    for (int i = 0; i < views; ++i) theta[i] = geometry[i].angle.x;
    for (int i = 1; i < views; ++i) {
        while (theta[i] - theta[i - 1] > pi) theta[i] -= 2.f * pi;
        while (theta[i] - theta[i - 1] < -pi) theta[i] += 2.f * pi;
    }

    result.assign(views, {});
    for (int i = 0; i < views; ++i) {
        const auto& g = geometry[i];
        auto& out = result[i];
        out.Nu = detector_u; out.Nv = detector_v; out.theta = theta[i];
        if (views == 1) out.dtheta = 1e-8f;
        else if (i == 0) out.dtheta = std::abs(theta[1] - theta[0]);
        else if (i + 1 == views) out.dtheta = std::abs(theta[i] - theta[i - 1]);
        else out.dtheta = 0.5f * std::abs(theta[i + 1] - theta[i - 1]);
        out.dtheta = std::max(out.dtheta, 1e-8f);
        out.fScaleDTheta = 2.f * pi / scan_range_rad;

        const float3 du = xyz(g.detU), dv = xyz(g.detV), src = xyz(g.src);
        const float uu = dot(du, du), vv = dot(dv, dv), uv = dot(du, dv);
        const float gram = uu * vv - uv * uv;
        if (!(uu > 1e-20f) || !(vv > 1e-20f) || std::abs(gram) < 1e-20f)
            return false;
        out.du_mm = std::sqrt(uu); out.dv_mm = std::sqrt(vv);
        out.inv_du_mm = 1.f / out.du_mm; out.inv_dv_mm = 1.f / out.dv_mm;
        out.UU = uu; out.VV = vv; out.UV = uv; out.invDetUV = 1.f / gram;
        out.det_u = packed(mul(du, out.inv_du_mm));
        out.det_v = packed(mul(dv, out.inv_dv_mm));

        SProjectionFrame frame{};
        if (!deriveProjectionFrame(g, detector_u, detector_v, frame)) return false;
        const float3 reference = sub(reconstruction_center, src);
        const float reference_length2 = dot(reference, reference);
        if (!(reference_length2 > 1e-20f)) return false;
        out.source_to_axis_mm = std::sqrt(reference_length2);
        const float3 ray = mul(reference, 1.f / out.source_to_axis_mm);
        const float denominator = dot(ray, frame.detectorNormal);
        if (std::abs(denominator) < 1e-12f) return false;
        const float distance = frame.planeDistance / denominator;
        if (!(distance > 0.f)) return false;
        const float3 principal = add(src, mul(ray, distance));
        const float3 delta = sub(principal, xyz(g.detS));
        const float du_dot = dot(delta, du), dv_dot = dot(delta, dv);
        const float u = (du_dot * vv - dv_dot * uv) / gram;
        const float v = (-du_dot * uv + dv_dot * uu) / gram;
        out.offsetU_pix = u - 0.5f * (detector_u - 1);
        out.offsetV_pix = v - 0.5f * (detector_v - 1);
        out.source_to_radial_detector_mm = distance;
        out.source_to_detector_plane_mm = frame.planeDistance;
        out.radial_ray = packed(ray);
        out.det_n = packed(frame.detectorNormal);
        const float3 detector_from_source = sub(xyz(g.detS), src);
        out.detS_sub_src_dot_dU = dot(detector_from_source, du);
        out.detS_sub_src_dot_dV = dot(detector_from_source, dv);
    }
    return true;
}

} // namespace YK::NonStandardFpBp
