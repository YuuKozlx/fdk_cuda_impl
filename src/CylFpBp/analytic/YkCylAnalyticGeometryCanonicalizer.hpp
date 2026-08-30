#pragma once

#include <cmath>
#include <vector>

#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "CylFpBp/analytic/YkCylAnalyticProjectionMapper.hpp"
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
    // canonical_geometry 描述 map 后的虚拟等角柱面，严格满足 R=SDD。
    std::vector<SCylConeProjGeomVec> canonical_geometry{};
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
        map.physical_arc_step_mm = first.radius_mm * first.channelStepRad;
        map.physical_row_step_mm = first.rowStepMm;
        map.target_angle_step_rad = first.channelStepRad;
        map.target_row_step_mm = first.rowStepMm;
        map.target_principal_u = first.principalU;
        map.target_principal_v = first.principalV;
        map.launch = config.launch;

        float principal_u = 0.f, principal_v = 0.f;
        if (!physicalPrincipal_(result.corrected_geometry.front(), first, sdd,
                principal_u, principal_v)) return false;
        map.physical_principal_u = principal_u;
        map.physical_principal_v = principal_v;

        result.canonical_geometry.reserve(result.corrected_geometry.size());
        for (const auto& physical : result.corrected_geometry) {
            SCylProjectionFrame frame{};
            if (!deriveCylProjectionFrame(physical, channels, rows, frame)) return false;
            const float tolerance = std::max(1e-3f, 1e-4f * sdd);
            if (std::fabs(frame.radius_mm - first.radius_mm) > tolerance ||
                std::fabs(frame.channelStepRad - first.channelStepRad) >
                    tolerance / std::max(1.f, first.radius_mm) ||
                std::fabs(frame.rowStepMm - first.rowStepMm) > tolerance)
                return false;
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
        const float tolerance = std::max(1e-3f, 1e-4f * sdd);
        const float center_distance = sdd - frame.radius_mm;
        float3 central{};
        if (std::fabs(center_distance) > tolerance) {
            central = make_float3(
                (frame.cylinderCenter.x - source.x) / center_distance,
                (frame.cylinderCenter.y - source.y) / center_distance,
                (frame.cylinderCenter.z - source.z) / center_distance);
        } else {
            const float axial = source.x * frame.axisUnit.x +
                source.y * frame.axisUnit.y + source.z * frame.axisUnit.z;
            central = make_float3(-source.x + axial * frame.axisUnit.x,
                -source.y + axial * frame.axisUnit.y,
                -source.z + axial * frame.axisUnit.z);
            const float length = std::sqrt(central.x * central.x +
                central.y * central.y + central.z * central.z);
            if (!(length > tolerance)) return false;
            central.x /= length; central.y /= length; central.z /= length;
        }
        const float length = std::sqrt(central.x * central.x +
            central.y * central.y + central.z * central.z);
        if (std::fabs(length - 1.f) > 1e-3f) return false;
        const float delta = std::atan2(
            central.x * frame.tangentUnit.x + central.y * frame.tangentUnit.y +
                central.z * frame.tangentUnit.z,
            central.x * frame.radialUnit.x + central.y * frame.radialUnit.y +
                central.z * frame.radialUnit.z);
        principal_u = frame.principalU + delta / frame.channelStepRad;
        const float axial_offset =
            (frame.detectorCenter.x - source.x) * frame.axisUnit.x +
            (frame.detectorCenter.y - source.y) * frame.axisUnit.y +
            (frame.detectorCenter.z - source.z) * frame.axisUnit.z;
        principal_v = frame.principalV - axial_offset / frame.rowStepMm;
        return true;
    }
};

} // namespace YK::CylFpBp::Analytic
