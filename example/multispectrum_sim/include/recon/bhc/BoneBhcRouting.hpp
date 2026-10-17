#pragma once
#include "config/SimConfig.hpp"
#include <cmath>
#include <stdexcept>
namespace yk::spectral::bone_bhc {
struct Routing {
    bool before_water = false;
    bool use_lut = false;
};
inline Routing resolveRouting(const SimulationConfig& c) {
    const auto& b=c.reconstruction.bone_hardening;
    Routing r;
    if(b.method=="physical_after_water")r={false,false};
    else if(b.method=="physical_before_water")r={true,false};
    else if(b.method=="two_material_lut")r={true,true};
    else throw std::invalid_argument("Unknown bone hardening method");
    if(!b.enabled)return r;
    if(b.max_iterations<1 || !std::isfinite(b.stop_ratio) || b.stop_ratio<=0 || b.stop_ratio>1 ||
       !std::isfinite(b.air_threshold_hu) || !std::isfinite(b.bone_threshold_hu) || b.air_threshold_hu>=b.bone_threshold_hu ||
       !std::isfinite(b.bone_thickness_scale) || b.bone_thickness_scale<=0)
        throw std::invalid_argument("Invalid bone hardening iteration/segmentation parameters");
    if(!c.reconstruction.preprocessing.beam_hardening.enabled)
        throw std::invalid_argument("Bone hardening needs water BHC for the initial HU reconstruction/reference");
    if(c.geometry.kind!=GeometryKind::FlatCbct && c.geometry.kind!=GeometryKind::FlatHelical && c.geometry.kind!=GeometryKind::CylHelical)
        throw std::invalid_argument("Bone hardening supports flat circular and flat/cylindrical helical scans");
    if(r.use_lut){
        if(b.calibration_file.empty())throw std::invalid_argument("Bone hardening LUT calibration file is missing");
    }else{
        if(c.projection.detector_response.enabled)throw std::invalid_argument("Physical bone hardening requires ideal energy-integrating detector response");
        for(const auto* m:{&b.soft_material,&b.bone_material})
            if(m->preset.empty() || !std::isfinite(m->density_g_cm3) || m->density_g_cm3<=0)
                throw std::invalid_argument("Invalid bone hardening material preset/density");
    }
    return r;
}
}
