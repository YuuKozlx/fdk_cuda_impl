#pragma once
#include "config/SimConfig.hpp"
#include "ui/Widgets.hpp"
#include <cfloat>
#include <vector>
namespace yk::spectral::ui {
struct SpectrumPanel {
    static void draw(SimulationConfig& draft, const std::vector<float>& spectrum_plot, bool& preview, bool& export_spectrum)
    {
        auto& sp = draft.projection.spectrum;
        int mode = sp.source == "generated" ? 1 : 0;
        if (ImGui::Combo("Spectrum source", &mode, "Import file\0Generate locally\0")) {
            sp.source = mode ? "generated" : "file";
            if (mode)
                draft.projection.spectrum_file.clear();
        }
        if (mode == 0)
            pathInput("Spectrum file", draft.projection.spectrum_file, L"Spectrum\0*.spec;*.csv;*.txt\0All files\0*.*\0\0");
        else {
            int model = sp.model == "spekcalc" ? 1 : 0;
            if (ImGui::Combo("Model", &model, "spekpy-v1\0spekcalc\0"))
                sp.model = model ? "spekcalc" : "spekpy-v1";
            number("Tube voltage (kVp)", sp.kvp);
            number("Anode angle (deg)", sp.anode_angle_deg);
            number("Bin width (keV)", sp.bin_width_keV);
            for (size_t i = 0; i < sp.filters.size(); ++i) {
                ImGui::PushID(int(i));
                auto& f = sp.filters[i];
                stringInput("Filter element", f.material);
                number("Thickness (mm)", f.thickness_mm);
                if (f.atomic_number) {
                    integer("Atomic number", f.atomic_number);
                    number("Filter density", f.density_g_cm3);
                }
                const bool remove = ImGui::Button("Remove filter");
                ImGui::PopID();
                if (remove) {
                    sp.filters.erase(sp.filters.begin() + i);
                    break;
                }
            }
            if (ImGui::Button("Add Al filter"))
                sp.filters.push_back({ "Al", 2.0 });
        }
        number("mAs / view", draft.projection.mAs_per_view);
        if (ImGui::Button("Preview spectrum"))
            preview = true;
        ImGui::SameLine();
        if (ImGui::Button("Export spectrum"))
            export_spectrum = true;
        if (!spectrum_plot.empty()) {
            ImGui::PlotLines("Fluence", spectrum_plot.data(), int(spectrum_plot.size()), 0, nullptr, 0, FLT_MAX, { 0, 130 });
            ImGui::Text("%d energy bins", int(spectrum_plot.size()));
        }
        ImGui::TextWrapped("Fluence: photons / (keV cm2 mAs) at 1 m. Exposure and distance are applied once by the projection backend.");
    }
};
}