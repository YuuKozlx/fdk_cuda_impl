#pragma once

#include <cmath>
#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/analytic/YkCylAnalyticProjectionMapper.hpp"
#include "YKCBCT/geometry/YkProjectionGeometry.hpp"
#include "YKCBCT/geometry/YkRigidTransform.hpp"

namespace YK::CylFpBp::Analytic {

struct GeometryCanonicalizationConfig {
    float source_to_detector_mm = 0.f;
    SRigidTransform canonical_from_acquisition = SRigidTransform::identity();
    SKernelLaunchPolicy launch{};
};

struct CanonicalGeometry {
    // corrected_geometry 仍描述物理探测器，只是已经进入目标世界坐标系。
    std::vector<SCylConeProjGeomVec> corrected_geometry{};
    // canonical_geometry 描述 map 后的目标平板几何；其姿态由中心射线
    // 定义，offset 通过主点保留。
    std::vector<SCylConeProjGeomVec> canonical_geometry{};
    // map 后供标准 Flat-FDK 使用的逐视图平板 geometry。
    std::vector<SConeProjGeomVec> flat_geometry{};
    ProjectionMapConfig projection_map{};
};

// 采集/配置层的纯 CPU 几何规范化器。它不接触投影数据，也不执行重建；
// SVolGeom 必须由调用方使用同一个目标坐标系定义，后端不会隐式移动体积。
class GeometryCanonicalizer {
public:
    bool canonicalize(int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& raw_geometry,
        const GeometryCanonicalizationConfig& config,
        CanonicalGeometry& result) const
    {
        result = {};
        if (channels < 2 || rows < 2 || raw_geometry.size() < 2 ||
            !(config.source_to_detector_mm > 0.f) ||
            !config.canonical_from_acquisition.isRigid()) return false;

        result.corrected_geometry.reserve(raw_geometry.size());
        for (const auto& raw : raw_geometry)
            result.corrected_geometry.push_back(transform_(raw,
                config.canonical_from_acquisition));

        const float sdd = config.source_to_detector_mm;
        SCylProjectionFrame first{};
        if (!deriveCylProjectionFrame(result.corrected_geometry.front(),
                channels, rows, first)) return false;
        ProjectionMapConfig map{};
        map.channels = channels; map.rows = rows;
        map.views = static_cast<int>(result.corrected_geometry.size());
        map.source_to_detector_mm = sdd;
        map.physical_radius_mm = first.radius_mm;
        const float3 first_source = xyz_(result.corrected_geometry.front().source);
        const float3 first_to_axis = make_float3(
            first.cylinderCenter.x - first_source.x,
            first.cylinderCenter.y - first_source.y,
            first.cylinderCenter.z - first_source.z);
        map.physical_source_axis_mm =
            first_to_axis.x * first.radialUnit.x +
            first_to_axis.y * first.radialUnit.y +
            first_to_axis.z * first.radialUnit.z;
        if (!std::isfinite(map.physical_source_axis_mm)) return false;
        map.physical_detector_axial_offset_mm =
            (first.detectorCenter.x - first_source.x) * first.axisUnit.x +
            (first.detectorCenter.y - first_source.y) * first.axisUnit.y +
            (first.detectorCenter.z - first_source.z) * first.axisUnit.z;
        map.physical_source_axis_axial_mm =
            (first_source.x - first.cylinderCenter.x) * first.axisUnit.x +
            (first_source.y - first.cylinderCenter.y) * first.axisUnit.y +
            (first_source.z - first.cylinderCenter.z) * first.axisUnit.z;
        map.physical_center_axial_mm = 0.f;
        map.physical_arc_step_mm = first.radius_mm * first.channelStepRad;
        map.physical_row_step_mm = first.rowStepMm;
        float principal_u = 0.f, principal_v = 0.f;
        if (!physicalPrincipal_(result.corrected_geometry.front(), first, sdd,
                principal_u, principal_v)) return false;
        // 目标数组是姿态为零的局部平板；offset 由主点保留。
        // 目标平板采用与物理通道相同的名义采样间距；其姿态固定为
        // 中心射线法向，物理横向 offset 仅通过 principal_u 保留。
        map.target_du_mm = map.physical_arc_step_mm;
        map.target_row_step_mm = first.rowStepMm;
        map.target_principal_u = principal_u;
        map.target_principal_v = principal_v;
        map.launch = config.launch;
        map.physical_principal_u = principal_u;
        map.physical_principal_v = principal_v;

        result.canonical_geometry.reserve(result.corrected_geometry.size());
        result.flat_geometry.reserve(result.corrected_geometry.size());
        for (const auto& physical : result.corrected_geometry) {
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(physical, channels, rows, frame)) return false;
            const float tolerance = std::max(1e-3f, 1e-4f * sdd);
            if (std::fabs(frame.radius_mm - first.radius_mm) > tolerance ||
                std::fabs(frame.channelStepRad - first.channelStepRad) >
                    tolerance / std::max(1.f, first.radius_mm) ||
                std::fabs(frame.rowStepMm - first.rowStepMm) > tolerance)
                return false;
            const float3 source_i = xyz_(physical.source);
            const float3 to_axis_i = make_float3(
                frame.cylinderCenter.x - source_i.x,
                frame.cylinderCenter.y - source_i.y,
                frame.cylinderCenter.z - source_i.z);
            const float source_axis_i = to_axis_i.x * frame.radialUnit.x +
                to_axis_i.y * frame.radialUnit.y +
                to_axis_i.z * frame.radialUnit.z;
            if (std::fabs(source_axis_i - map.physical_source_axis_mm) >
                std::max(1e-3f, 1e-4f * sdd)) return false;
            float pu = 0.f, pv = 0.f;
            if (!physicalPrincipal_(physical, frame, sdd, pu, pv) ||
                std::fabs(pu - map.physical_principal_u) > 1e-3f ||
                std::fabs(pv - map.physical_principal_v) > 1e-3f) return false;

            const float3 source = xyz_(physical.source);
            const float axial = (frame.detectorCenter.x - source.x) * frame.axisUnit.x +
                (frame.detectorCenter.y - source.y) * frame.axisUnit.y +
                (frame.detectorCenter.z - source.z) * frame.axisUnit.z;
            const float3 detector = make_float3(
                source.x + frame.radialUnit.x * sdd + frame.axisUnit.x * axial,
                source.y + frame.radialUnit.y * sdd + frame.axisUnit.y * axial,
                source.z + frame.radialUnit.z * sdd + frame.axisUnit.z * axial);
            SCylConeProjGeomVec canonical = physical;
            canonical.detectorCenter = make_float4(detector.x, detector.y,
                detector.z, physical.detectorCenter.w);
            canonical.detectorU = make_float4(
                frame.tangentUnit.x * sdd * frame.channelStepRad,
                frame.tangentUnit.y * sdd * frame.channelStepRad,
                frame.tangentUnit.z * sdd * frame.channelStepRad,
                physical.detectorU.w);
            setCylDetectorRadius(canonical, sdd);
            result.canonical_geometry.push_back(canonical);

            const float3 u = make_float3(frame.tangentUnit.x * map.target_du_mm,
                frame.tangentUnit.y * map.target_du_mm,
                frame.tangentUnit.z * map.target_du_mm);
            const float3 v = make_float3(frame.axisUnit.x * map.target_row_step_mm,
                frame.axisUnit.y * map.target_row_step_mm,
                frame.axisUnit.z * map.target_row_step_mm);
            const float3 center = make_float3(
                source.x + frame.radialUnit.x * sdd,
                source.y + frame.radialUnit.y * sdd,
                source.z + frame.radialUnit.z * sdd);
            SConeProjGeomVec flat{};
            flat.src = physical.source;
            flat.detS = make_float4(
                center.x - map.target_principal_u * u.x - map.target_principal_v * v.x,
                center.y - map.target_principal_u * u.y - map.target_principal_v * v.y,
                center.z - map.target_principal_u * u.z - map.target_principal_v * v.z,
                physical.detectorCenter.w);
            flat.detU = make_float4(u.x, u.y, u.z, physical.detectorU.w);
            flat.detV = make_float4(v.x, v.y, v.z, physical.detectorV.w);
            flat.angle = physical.viewParameters;
            result.flat_geometry.push_back(flat);
        }
        result.projection_map = map;
        return true;
    }

private:
    static float3 xyz_(float4 v) { return make_float3(v.x, v.y, v.z); }
    static SCylConeProjGeomVec transform_(const SCylConeProjGeomVec& g,
        const SRigidTransform& transform)
    {
        const auto point = [&](float4 v) {
            const float3 p = transform.transformPoint(xyz_(v));
            return make_float4(p.x, p.y, p.z, v.w);
        };
        const auto vector = [&](float4 v) {
            const float3 p = transform.transformVector(xyz_(v));
            return make_float4(p.x, p.y, p.z, v.w);
        };
        SCylConeProjGeomVec result = g;
        result.source = point(g.source);
        result.detectorCenter = point(g.detectorCenter);
        result.detectorU = vector(g.detectorU);
        result.detectorV = vector(g.detectorV);
        return result;
    }

    static bool physicalPrincipal_(const SCylConeProjGeomVec& geometry,
        const SCylProjectionFrame& frame, float sdd, float& principal_u,
        float& principal_v)
    {
        const float3 source = xyz_(geometry.source);
        const float3 center_to_detector = make_float3(
            frame.detectorCenter.x - source.x,
            frame.detectorCenter.y - source.y,
            frame.detectorCenter.z - source.z);
        const float radial = center_to_detector.x * frame.radialUnit.x +
            center_to_detector.y * frame.radialUnit.y +
            center_to_detector.z * frame.radialUnit.z;
        if (!(radial > 1e-6f) || !std::isfinite(radial)) return false;
        const float tangent = center_to_detector.x * frame.tangentUnit.x +
            center_to_detector.y * frame.tangentUnit.y +
            center_to_detector.z * frame.tangentUnit.z;
        const float delta = std::atan2(
            tangent, radial);
        principal_u = frame.principalU + delta / frame.channelStepRad;
        const float source_axial =
            (source.x - frame.cylinderCenter.x) * frame.axisUnit.x +
            (source.y - frame.cylinderCenter.y) * frame.axisUnit.y +
            (source.z - frame.cylinderCenter.z) * frame.axisUnit.z;
        const float detector_axial =
            (frame.detectorCenter.x - frame.cylinderCenter.x) * frame.axisUnit.x +
            (frame.detectorCenter.y - frame.cylinderCenter.y) * frame.axisUnit.y +
            (frame.detectorCenter.z - frame.cylinderCenter.z) * frame.axisUnit.z;
        const float lambda = std::sqrt(radial * radial + tangent * tangent) /
            std::max(sdd, 1e-6f);
        // q = source_axial + lambda*v；令目标平板中心射线落到物理
        // 探测器中心的轴向位置，轴向 offset 由 principal_v 保留。
        principal_v = frame.principalV - (detector_axial - source_axial) /
            (std::max(lambda, 1e-6f) * frame.rowStepMm);
        return true;
    }
};

} // namespace YK::CylFpBp::Analytic
