#pragma once
#include "config/SimConfig.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
namespace yk::spectral {
struct ReconstructionGeometryField {
    const char* key;
    const char* label;
    double GeometryConfig::* member;
    bool angle = false;
};
inline const std::array<ReconstructionGeometryField,11>& reconstructionGeometryFields(){
    static const std::array<ReconstructionGeometryField,11> fields{{
        {"sid_mm","SID (mm)",&GeometryConfig::sid_mm},
        {"sdd_mm","SDD (mm)",&GeometryConfig::sdd_mm},
        {"offset_u_mm","Detector U (mm)",&GeometryConfig::offset_u_mm},
        {"offset_v_mm","Detector V (mm)",&GeometryConfig::offset_v_mm},
        {"offset_n_mm","Detector normal (mm)",&GeometryConfig::offset_n_mm},
        {"tilt_u_rad","Detector tilt U (deg)",&GeometryConfig::tilt_u_rad,true},
        {"tilt_v_rad","Detector tilt V (deg)",&GeometryConfig::tilt_v_rad,true},
        {"tilt_n_rad","Detector tilt N (deg)",&GeometryConfig::tilt_n_rad,true},
        {"source_offset_x_mm","Source X (mm)",&GeometryConfig::source_offset_x_mm},
        {"source_offset_y_mm","Source Y (mm)",&GeometryConfig::source_offset_y_mm},
        {"source_offset_z_mm","Source Z (mm)",&GeometryConfig::source_offset_z_mm}
    }};
    return fields;
}
inline GeometryConfig resolveReconstructionGeometry(const SimulationConfig& c){
    auto g=c.geometry.parameters;
    if(!c.reconstruction.geometry_override.enabled)return g;
    for(const auto& f:reconstructionGeometryFields()){
        double value=c.reconstruction.geometry_override.parameters.*f.member;
        if(!std::isfinite(value))throw std::invalid_argument(std::string("Invalid reconstruction geometry: ")+f.key);
        g.*f.member=value;
    }
    if(g.sid_mm<=0 || g.sdd_mm<=g.sid_mm)throw std::invalid_argument("Reconstruction geometry requires SDD > SID > 0");
    return g;
}
}
