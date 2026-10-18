#include "input/config/ReconstructionGeometry.hpp"
#include "client/ProjectionUiConfig.hpp"
#include "input/materials/XcomAttenuationProvider.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace yk::spectral {
namespace {
    const char* kind(GeometryKind k)
    {
        switch (k) {
        case GeometryKind::FlatCbct:
            return "flat_cbct";
        case GeometryKind::FlatHelical:
            return "flat_helical";
        case GeometryKind::PlanarSourceOrbit:
            return "planar_source_orbit";
        case GeometryKind::NikonTilted:
            return "nikon_tilted";
        default:
            throw std::runtime_error("ImGui supports only the four flat detector scan types");
        }
    }

}
bool setProjectionUiScanType(SimulationConfig& c,GeometryKind target) {
    if(c.geometry.kind==target)return false;
    auto& g=c.geometry.parameters;
    const bool was_helical=c.geometry.kind==GeometryKind::FlatHelical;
    const bool helical=target==GeometryKind::FlatHelical;
    const double half_turns=g.views_per_turn>0 ? .5*g.rotation_direction*g.views/g.views_per_turn : 0;
    // Preserve the center of the scanned Z range when changing trajectory.
    const double center_z=g.start_z_mm+(was_helical?g.pitch_mm_per_turn*half_turns:0);
    if(helical && g.pitch_mm_per_turn==0)g.pitch_mm_per_turn=100;
    if(was_helical!=helical)g.start_z_mm=center_z-(helical?g.pitch_mm_per_turn*half_turns:0);
    c.geometry.kind=target;
    if(target==GeometryKind::PlanarSourceOrbit){
        if(g.planar_detector_y_mm<=0)g.planar_detector_y_mm=std::max(g.sdd_mm-g.sid_mm,1.0);
        if(g.source_orbit_radius_mm<=0)g.source_orbit_radius_mm=g.sid_mm;
        g.reconstruction_detector_u=g.detector_u;g.reconstruction_detector_v=g.detector_v;
    }
    if(target==GeometryKind::PlanarSourceOrbit || target==GeometryKind::NikonTilted)c.projection.detector_response.enabled=false;
    if(c.reconstruction.type=="analytic"){
        if(helical && c.reconstruction.analytic.pipeline=="fdk")c.reconstruction.analytic.pipeline="wfbp";
        else if(!helical && c.reconstruction.analytic.pipeline=="wfbp")c.reconstruction.analytic.pipeline="fdk";
    }
    return true;
}
void selectBuiltinMaterial(MaterialSpec& material,std::string_view id) {
    const auto* preset=findBuiltinMaterial(id);
    if(!preset)throw std::runtime_error("Unknown builtin material: "+std::string(id));
    material.preset=preset->id;
    material.formula.clear();
    material.name=preset->display_name;
    material.density_g_cm3=preset->density_g_cm3;
}
void validateProjectionUiInputs(const SimulationConfig& c, const std::vector<std::uint8_t>& labels)
{
    kind(c.geometry.kind);
    const auto& g = c.geometry.parameters;
    if (g.volume_x <= 0 || g.volume_y <= 0 || g.volume_z <= 0 || static_cast<std::uint64_t>(g.volume_x) * g.volume_y * g.volume_z != labels.size())
        throw std::runtime_error("Raw uint8 file size must equal X * Y * Z");
    for(double value:{g.sid_mm,g.sdd_mm,g.voxel_x_mm,g.voxel_y_mm,g.voxel_z_mm,g.pixel_u_mm,g.pixel_v_mm})
        if(!std::isfinite(value) || value<=0)throw std::runtime_error("Distances, voxel spacing and pixel spacing must be finite and positive");
    if(g.views<=0 || g.views_per_turn<=0 || g.detector_u<=0 || g.detector_v<=0 || c.projection.sampling.samples_per_pixel<=0)
        throw std::runtime_error("View count, detector dimensions and ray count must be positive");
    if(g.rotation_direction!=1 && g.rotation_direction!=-1)throw std::runtime_error("Rotation direction must be +1 or -1");
    if(!std::isfinite(g.start_z_mm) || !std::isfinite(g.start_angle_rad))throw std::runtime_error("Scan start position and angle must be finite");
    if(c.geometry.kind==GeometryKind::FlatHelical && (!std::isfinite(g.pitch_mm_per_turn) || g.pitch_mm_per_turn==0))
        throw std::runtime_error("Helical pitch must be finite and nonzero");
    if(c.projection.output_file.empty())throw std::runtime_error("Set a projection output path");
    if (g.sdd_mm <= g.sid_mm)
        throw std::runtime_error("SDD must be greater than SID");
    if (c.projection.materials.empty() || c.projection.materials.size() > 64)
        throw std::runtime_error("Use 1..64 material mappings");
    std::array<bool, 256> defined {};
    for (const auto& m : c.projection.materials) {
        if (defined[m.label])
            throw std::runtime_error("Duplicate material label");
        defined[m.label] = true;
    }
    for (const auto label : labels)
        if (label != 0 && !defined[label])
            throw std::runtime_error("Missing material for label " + std::to_string(label));
}
}
