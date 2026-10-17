#pragma once
#include "config/SimConfig.hpp"
#include "ProjectionUiConfig.hpp"
#include "ui/Widgets.hpp"
#include <cfloat>
#include <vector>
namespace yk::spectral::ui {
struct ScanPanel {
    static bool scanTypeCombo(SimulationConfig& draft)
    {
        const GeometryKind kinds[]={GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::PlanarSourceOrbit,GeometryKind::NikonTilted};
        int index=0;
        for(int i=0;i<4;++i)if(draft.geometry.kind==kinds[i])index=i;
        if(!ImGui::Combo("Scan type",&index,"Flat circular\0Flat helical\0Planar CT\0Tilted CT\0"))return false;
        return setProjectionUiScanType(draft,kinds[index]);
    }
    static bool draw(SimulationConfig& draft)
    {
        ImGui::SeparatorText("Trajectory");
        const GeometryKind kinds[] = { GeometryKind::FlatCbct, GeometryKind::FlatHelical, GeometryKind::PlanarSourceOrbit, GeometryKind::NikonTilted };
        int index = 0;
        for (int i = 0; i < 4; ++i)
            if (draft.geometry.kind == kinds[i])
                index = i;
        auto& g = draft.geometry.parameters;
        const bool changed=scanTypeCombo(draft);
        for(int i=0;i<4;++i)if(draft.geometry.kind==kinds[i])index=i;
        integer("Views", g.views);
        integer("Views / turn", g.views_per_turn);
        angle("Start angle (deg)", g.start_angle_rad);
        integer("Direction (+/-1)", g.rotation_direction);
        if (index == 1) {
            number("Pitch (mm / turn)", g.pitch_mm_per_turn);
            number("Start Z (mm)", g.start_z_mm);
        }
        if(index==0)number("Orbit Z (mm)",g.start_z_mm);
        if (index >= 2)
            angle("Scan range (deg)", g.scan_range_rad);
        if (index == 2) {
            number("Detector Y (mm)", g.planar_detector_y_mm);
            number("Source orbit R (mm)", g.source_orbit_radius_mm);
            number("Source lateral (mm)", g.source_lateral_offset_mm);
        }
        if (index == 3) {
            angle("Axis tilt (deg)", g.nikon_tilt_rad);
            angle("Azimuth (deg)", g.nikon_tilt_axis_azimuth_rad);
            number("Pivot X (mm)", g.rotation_axis_pivot_x_mm);
            number("Pivot Y (mm)", g.rotation_axis_pivot_y_mm);
            number("Pivot Z (mm)", g.rotation_axis_pivot_z_mm);
        }
        if (ImGui::CollapsingHeader("Detector", ImGuiTreeNodeFlags_DefaultOpen)) {
            integer("Columns", g.detector_u);
            integer("Rows", g.detector_v);
            number("Pixel U (mm)", g.pixel_u_mm);
            number("Pixel V (mm)", g.pixel_v_mm);
            number("SID (mm)", g.sid_mm);
            number("SDD (mm)", g.sdd_mm);
            if (ImGui::TreeNode("Detector position and tilt")) {
                number("Offset U (mm)", g.offset_u_mm);
                number("Offset N (mm)", g.offset_n_mm);
                number("Offset V (mm)", g.offset_v_mm);
                angle("Tilt U (deg)", g.tilt_u_rad);
                angle("Tilt V (deg)", g.tilt_v_rad);
                angle("Tilt N (deg)", g.tilt_n_rad);
                ImGui::TreePop();
            }
        }
        if (ImGui::CollapsingHeader("Source")) {
            number("Source X (mm)", g.source_offset_x_mm);
            number("Source Y (mm)", g.source_offset_y_mm);
            number("Source Z (mm)", g.source_offset_z_mm);
        }
        return changed;
    }
};
}
