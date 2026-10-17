#pragma once
#include "../../visualization/LiveView.hpp"
#include "WaterReference.hpp"
#include <imgui.h>

namespace yk::spectral::ui {
// Presentation state and queued edits; no reconstruction algorithm or data-file ownership.
struct ReconstructionViewerPanel {
    std::array<int, 3> recon_dimensions {}, recon_slices {};
    std::array<double, 3> spacing { 1, 1, 1 };
    double window_low = 0, window_high = 1;
    bool slices_changed = false, window_changed = false, auto_window = false;
    bool expanded = false, layout_changed = false, crosshairs = true;
    int single_panel = -1, preset = 6;
    int rotation_panel = -1;
    double rotation_angle = 0;
    yk::visualization::MprSettings mpr;
    void reset(const yk::visualization::FloatVolume& volume)
    {
        recon_dimensions = volume.dimensions;
        spacing = volume.spacing;
        for (int a = 0; a < 3; ++a)
            recon_slices[a] = recon_dimensions[a] / 2;
        mpr = {};
        mpr.indices = recon_slices;
        single_panel = -1;
        rotation_panel = -1;
        preset = 6;
        slices_changed = window_changed = auto_window = false;
        layout_changed = true;
    }
    void controls(yk::visualization::LiveView& view, const WaterReference& reference)
    {
        using namespace yk::visualization;
        if (!view.hasReconstruction()) {
            ImGui::TextWrapped("Run reconstruction or load a saved volume to view MPR.");
            return;
        }
        if (ImGui::BeginTable("Slice indices", 3, ImGuiTableFlags_SizingStretchSame)) {
            const char* names[] = { "X", "Y", "Z" };
            for (int a = 0; a < 3; ++a) {
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(-22);
                if (ImGui::SliderInt(names[a], &recon_slices[a], 0, recon_dimensions[a] - 1))
                    slices_changed = true;
            }
            ImGui::EndTable();
        }
        const float third = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3;
        ImGui::SetNextItemWidth(third);
        if (ImGui::BeginCombo("##Preset", preset < 6 ? ctWindowPresets[preset].name : "Custom / auto")) {
            for (int i = 0; i < 6; ++i)
                if (ImGui::Selectable(ctWindowPresets[i].name, preset == i)) {
                    preset = i;
                    window_low = ctWindowPresets[i].level - ctWindowPresets[i].width / 2;
                    window_high = ctWindowPresets[i].level + ctWindowPresets[i].width / 2;
                    window_changed = true;
                }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        double ww = window_high - window_low, wl = (window_high + window_low) / 2;
        ImGui::SetNextItemWidth(third - 32);
        bool cw = ImGui::InputDouble("WW", &ww, 0, 0, "%.1f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(third - 32);
        bool cl = ImGui::InputDouble("WL", &wl, 0, 0, "%.1f");
        if ((cw || cl) && std::isfinite(ww) && std::isfinite(wl)) {
            ww = std::max(1., ww);
            window_low = wl - ww / 2;
            window_high = wl + ww / 2;
            window_changed = true;
            preset = 6;
        }
        ImGui::TextUnformatted("MPR rotation (grid X / Y / Z, degrees)");
        if (ImGui::BeginTable("Oblique angles", 3, ImGuiTableFlags_SizingStretchSame)) {
            for (int a = 0; a < 3; ++a) {
                ImGui::TableNextColumn();
                ImGui::PushID(a);
                ImGui::SetNextItemWidth(-1);
                const double low = -180, high = 180;
                auto angles = mpr.rotation_deg;
                if (ImGui::SliderScalar("##Angle", ImGuiDataType_Double, &angles[a], &low, &high, "%.1f deg")) {
                    setMprRotation(mpr, angles);
                    slices_changed = true;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        double diagonal = 0;
        for (int a = 0; a < 3; ++a)
            diagonal += std::pow((recon_dimensions[a] - 1) * spacing[a], 2);
        diagonal = std::sqrt(diagonal);
        const double negative = -diagonal;
        ImGui::SetNextItemWidth(-110);
        if (ImGui::SliderScalar("Plane offset", ImGuiDataType_Double, &mpr.offset_mm, &negative, &diagonal, "%.2f mm"))
            slices_changed = true;
        int interpolation = int(mpr.interpolation);
        ImGui::SetNextItemWidth(150);
        if (ImGui::Combo("##Interpolation", &interpolation, "Nearest\0Linear\0Cubic\0")) {
            mpr.interpolation = MprInterpolation(interpolation);
            slices_changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset MPR")) {
            mpr.rotation_deg = {};
            mpr.offset_mm = 0;
            mpr.reference_axes = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            mpr.reference_roll_deg = {};
            slices_changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Center")) {
            for (int a = 0; a < 3; ++a)
                recon_slices[a] = recon_dimensions[a] / 2;
            mpr.offset_mm = 0;
            slices_changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Auto WW")) {
            auto_window = true;
            preset = 6;
        }
        if (ImGui::Checkbox("Expand viewer", &expanded))
            layout_changed = true;
        ImGui::SameLine();
        ImGui::Checkbox("Crosshair", &crosshairs);
        if (single_panel >= 0) {
            ImGui::SameLine();
            if (ImGui::Button("Four views")) {
                single_panel = -1;
                layout_changed = true;
            }
        }
        const auto point = view.reconstructionCursorWorld();
        ImGui::Text("Cursor %.1f / %.1f / %.1f mm | %.1f HU", point[0], point[1], point[2], view.reconstructionCursorValue());
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Water %.2f keV, %.7f mm^-1\nCoordinates follow grid metadata, not a patient orientation convention.", reference.energy_keV, reference.mu_water_mm_inv);
    }
    void imageInteractions(yk::visualization::LiveView& view, bool enabled)
    {
        if (!view.hasReconstruction())
            return;
        if (!enabled || !ImGui::IsMouseDown(ImGuiMouseButton_Left))
            rotation_panel = -1;
        for (int panel = 0; panel < 4; ++panel) {
            const auto rect = view.reconstructionPanelRect(panel);
            if (rect[2] <= rect[0] || rect[3] <= rect[1])
                continue;
            ImGui::SetNextWindowPos({ float(rect[0]), float(rect[1]) });
            ImGui::SetNextWindowSize({ float(rect[2] - rect[0]), float(rect[3] - rect[1]) });
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
            ImGui::PushID(panel);
            const std::string title = "MPR image##" + std::to_string(panel);
            ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNav);
            ImGui::InvisibleButton("Image", ImGui::GetContentRegionAvail(), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            auto* list = ImGui::GetWindowDrawList();
            bool over_reference = false;
            const auto pos = view.reconstructionCrosshair(panel);
            if (crosshairs && panel < 3) {
                const auto& mouse = ImGui::GetIO().MousePos;
                const double dx = mouse.x - pos[0], dy = mouse.y - pos[1], radius = std::hypot(dx, dy);
                const auto references = view.reconstructionReferenceLines(panel);
                const double length = std::hypot(rect[2] - rect[0], rect[3] - rect[1]);
                list->PushClipRect({ float(rect[0]), float(rect[1]) }, { float(rect[2]), float(rect[3]) }, true);
                for (const auto& line : references) {
                    const bool line_hovered = radius > 20 && std::abs(dx * line.direction[1] - dy * line.direction[0]) < 8;
                    over_reference = over_reference || line_hovered;
                    const ImU32 color = line.panel == 0 ? IM_COL32(112, 190, 255, 210) : line.panel == 1 ? IM_COL32(126, 224, 169, 210)
                                                                                                         : IM_COL32(248, 172, 120, 210);
                    auto point = [&](double d) { return ImVec2(float(pos[0] + d * line.direction[0]), float(pos[1] + d * line.direction[1])); };
                    list->AddLine(point(-length), point(-8), color, line_hovered ? 2.5f : 1.3f);
                    list->AddLine(point(8), point(length), color, line_hovered ? 2.5f : 1.3f);
                    list->AddCircleFilled(point(64), line_hovered ? 6.f : 4.f, color);
                    list->AddCircleFilled(point(-64), line_hovered ? 6.f : 4.f, color);
                }
                list->AddCircle({ float(pos[0]), float(pos[1]) }, 7, IM_COL32(235, 240, 250, 180));
                list->PopClipRect();
            } else if (crosshairs) {
                const auto pos = view.reconstructionCrosshair(panel);
                const ImU32 color = panel == 0 ? IM_COL32(112, 190, 255, 180) : panel == 1 ? IM_COL32(126, 224, 169, 180)
                                                                                           : IM_COL32(248, 172, 120, 180);
                const float x = float(pos[0]), y = float(pos[1]);
                if (x >= rect[0] && x <= rect[2] && y >= rect[1] && y <= rect[3]) {
                    list->AddLine({ float(rect[0]) + 8, y }, { x - 7, y }, color);
                    list->AddLine({ x + 7, y }, { float(rect[2]) - 8, y }, color);
                    list->AddLine({ x, float(rect[1]) + 8 }, { x, y - 7 }, color);
                    list->AddLine({ x, y + 7 }, { x, float(rect[3]) - 8 }, color);
                }
            }
            if (enabled) {
                const auto& io = ImGui::GetIO();
                if (ImGui::IsItemHovered()) {
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        single_panel = single_panel == panel ? -1 : panel;
                        layout_changed = true;
                    }
                    if (io.MouseWheel != 0) {
                        if (panel < 3) {
                            if (yk::visualization::mprReferenceIsOrthogonal(mpr)) {
                                const int a = panel == 0 ? 2 : panel == 1 ? 1
                                                                          : 0;
                                recon_slices[a] = std::clamp(recon_slices[a] + int(io.MouseWheel), 0, recon_dimensions[a] - 1);
                            } else {
                                const auto normal = view.reconstructionPlaneNormal(panel);
                                double inverse_step = 0;
                                for (int a = 0; a < 3; ++a)
                                    inverse_step += normal[a] * normal[a] / (spacing[a] * spacing[a]);
                                const double step = io.MouseWheel / std::sqrt(inverse_step);
                                for (int a = 0; a < 3; ++a)
                                    recon_slices[a] = std::clamp(recon_slices[a] + int(std::lround(step * normal[a] / spacing[a])), 0, recon_dimensions[a] - 1);
                            }
                        } else
                            mpr.offset_mm = std::clamp(mpr.offset_mm + io.MouseWheel * *std::min_element(spacing.begin(), spacing.end()), -10000., 10000.);
                        slices_changed = true;
                    }
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                        ImGui::SetTooltip("Drag colored reference lines / dots: rotate linked planes\nDrag center or click image: move cursor | Wheel: slice\nRight drag: window width / level | Double click: maximize");
                    if (over_reference)
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    if (over_reference && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        rotation_panel = panel;
                        rotation_angle = std::atan2(io.MousePos.y - pos[1], io.MousePos.x - pos[0]);
                    }
                }
                if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    if (rotation_panel == panel) {
                        const double dx = io.MousePos.x - pos[0], dy = io.MousePos.y - pos[1];
                        if (std::hypot(dx, dy) > 16) {
                            const double angle = std::atan2(dy, dx), delta = std::remainder(angle - rotation_angle, 6.283185307179586);
                            if (std::abs(delta) > 1e-7) {
                                yk::visualization::rotateMprReference(mpr, panel, -delta);
                                slices_changed = true;
                            }
                            rotation_angle = angle;
                        }
                    } else if (const auto picked = view.pickReconstruction(panel, io.MousePos.x, io.MousePos.y)) {
                        if (recon_slices != *picked || (panel == 3 && mpr.offset_mm != 0)) {
                            recon_slices = *picked;
                            if (panel == 3)
                                mpr.offset_mm = 0;
                            slices_changed = true;
                        }
                    }
                }
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                    const double width = std::max(1., window_high - window_low);
                    const double new_width = std::clamp(width * std::exp(io.MouseDelta.x * .006), 1., 100000.);
                    const double level = (window_high + window_low) * .5 + io.MouseDelta.y * width * .003;
                    window_low = level - new_width * .5;
                    window_high = level + new_width * .5;
                    window_changed = true;
                    preset = 6;
                }
            }
            ImGui::End();
            ImGui::PopID();
            ImGui::PopStyleVar(2);
        }
    }
};
}
