#include "XcomAttenuationProvider.hpp"
#pragma once
#include "ProjectionUiConfig.hpp"
#include "ui/Widgets.hpp"
#include <cstdint>
namespace yk::spectral::ui {
struct VolumePanel {
    ImGuiTextFilter material_filter;
    int material_category = 0;
    void builtinPicker(MaterialSpec& material)
    {
        const auto* selected = findBuiltinMaterial(material.preset);
        const std::string preview = selected ? std::string(selected->display_name) + " [" + selected->category + "]" : material.preset;
        const float popup_width = std::clamp(ImGui::GetContentRegionAvail().x, 240.f, 430.f);
        ImGui::SetNextItemWidth(-100);
        ImGui::SetNextWindowSizeConstraints({ popup_width, 0 }, { popup_width, 480 });
        const auto preview_short = shortLabel(preview, popup_width - 150);
        if (ImGui::BeginCombo("Material", preview_short.c_str(), ImGuiComboFlags_HeightLargest)) {
            material_filter.Draw("Search", -1);
            ImGui::SetNextItemWidth(-85);
            ImGui::Combo("Category", &material_category, "All\0Tissue\0Element\0Compound\0Mixture\0ICRP145\0");
            const char* categories[] = { "", "tissue", "element", "compound", "mixture", "ICRP145" };
            ImGui::BeginChild("Material list", { 0, 260 }, false);
            int shown = 0;
            for (auto* info = builtinMaterialsBegin(); info != builtinMaterialsEnd(); ++info) {
                if (material_category && std::string_view(info->category) != categories[material_category])
                    continue;
                const std::string searchable = std::string(info->id) + " " + info->display_name + " " + info->aliases;
                if (!material_filter.PassFilter(searchable.c_str()))
                    continue;
                ++shown;
                ImGui::PushID(info->id);
                const std::string full_label = std::string(info->display_name) + " [" + info->category + "]";
                const auto label = shortLabel(full_label, ImGui::GetContentRegionAvail().x - 10);
                if (ImGui::Selectable(label.c_str(), selected && std::string_view(info->id) == selected->id)) {
                    selectBuiltinMaterial(material, info->id);
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24);
                    ImGui::TextWrapped("%s", info->display_name);
                    ImGui::TextWrapped("%s", info->id);
                    ImGui::Text("Default density: %.6g g/cm3", info->density_g_cm3);
                    ImGui::TextWrapped("Element mass fractions: %s", info->composition);
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                ImGui::PopID();
            }
            if (!shown)
                ImGui::TextUnformatted("No matching materials");
            ImGui::EndChild();
            ImGui::Text("%d matching / %d builtin materials", shown, int(builtinMaterialsEnd() - builtinMaterialsBegin()));
            ImGui::EndCombo();
        }
        selected = findBuiltinMaterial(material.preset);
        if (selected) {
            ImGui::TextWrapped("ID: %s", selected->id);
            ImGui::Text("Default density: %.6g g/cm3", selected->density_g_cm3);
            if (ImGui::SmallButton("Use default density"))
                material.density_g_cm3 = selected->density_g_cm3;
        }
    }
    void draw(SimulationConfig& draft, const std::array<std::uint64_t, 256>& counts)
    {
        auto& g = draft.geometry.parameters;
        ImGui::TextWrapped("Raw uint8 labels, order Z/Y/X. Full voxel resolution is retained.");
        pathInput("Label volume", draft.projection.label_volume, L"Label volumes\0*.mat;*.raw;*.bin\0All files\0*.*\0\0");
        integer("Size X", g.volume_x);
        integer("Size Y", g.volume_y);
        integer("Size Z", g.volume_z);
        number("Voxel X (mm)", g.voxel_x_mm);
        number("Voxel Y (mm)", g.voxel_y_mm);
        number("Voxel Z (mm)", g.voxel_z_mm);
        if (ImGui::CollapsingHeader("Object position and orientation")) {
            number("Object X (mm)", g.phantom_offset_x_mm);
            number("Object Y (mm)", g.phantom_offset_y_mm);
            number("Object Z (mm)", g.phantom_offset_z_mm);
            angle("Object Rx (deg)", g.phantom_rotation_x_rad);
            angle("Object Ry (deg)", g.phantom_rotation_y_rad);
            angle("Object Rz (deg)", g.phantom_rotation_z_rad);
        }
        ImGui::SeparatorText("Label -> material");
        ImGui::TextWrapped("Label 0 may be unmapped background. Every other label needs a material. Histogram reflects the loaded volume.");
        auto& materials = draft.projection.materials;
        for (size_t i = 0; i < materials.size(); ++i) {
            ImGui::PushID(int(i));
            auto& m = materials[i];
            if (ImGui::TreeNodeEx("Material", ImGuiTreeNodeFlags_None, "Label %d: %s", int(m.label), m.name.c_str())) {
                int label = int(m.label);
                if (ImGui::InputInt("Label", &label))
                    m.label = std::uint8_t(std::clamp(label, 0, 255));
                ImGui::Text("Voxels: %llu", static_cast<unsigned long long>(counts[m.label]));
                stringInput("Name", m.name);
                int mode = m.preset.empty() ? 1 : 0;
                if (ImGui::Combo("Material source", &mode, "Builtin list\0Chemical formula\0")) {
                    if (mode == 0)
                        selectBuiltinMaterial(m, "compound.water");
                    else {
                        m.preset.clear();
                        m.formula = "H2O";
                    }
                }
                if (mode == 0)
                    builtinPicker(m);
                else {
                    stringInput("Formula", m.formula);
                    ImGui::TextWrapped("Formula uses atom counts, e.g. H2O. Density is specified separately.");
                }
                number("Density (g/cm3)", m.density_g_cm3);
                const bool remove = ImGui::Button("Remove material");
                ImGui::TreePop();
                if (remove) {
                    materials.erase(materials.begin() + i);
                    ImGui::PopID();
                    break;
                }
            }
            ImGui::PopID();
        }
        if (materials.size() < 64 && ImGui::Button("Add material")) {
            MaterialSpec m;
            m.label = 1;
            for (const auto& x : materials)
                if (m.label <= x.label)
                    m.label = x.label + 1;
            selectBuiltinMaterial(m, "compound.water");
            materials.push_back(m);
        }
    }
};
}
