#pragma once
#include "config/SimConfig.hpp"
#include "ui/Widgets.hpp"
#include <cfloat>
#include <vector>
namespace yk::spectral::ui {
struct EffectsPanel {
    static void draw(SimulationConfig& draft)
    {
        auto& p = draft.projection;
        ImGui::SeparatorText("Exposure and noise");
        ImGui::Checkbox("Geometry flux", &p.geometry_flux.enabled);
        ImGui::Checkbox("Quantum noise", &p.quantum_noise.enabled);
        if (p.quantum_noise.enabled) {
            ImGui::SetNextItemWidth(-185);
            ImGui::InputScalar("Noise seed", ImGuiDataType_U32, &p.quantum_noise.seed);
        }
        if (ImGui::CollapsingHeader("Performance / advanced")) {
            integer("Batch views", p.batching.views);
            integer("Batch buffer MiB", p.batching.max_buffer_mib);
            ImGui::TextWrapped("Defaults: automatic batches up to 256 MiB (64-view safety limit). Actual batch size adapts to available GPU memory and reserves space for VTK. Set views to 1 for comparison.");
        }
        ImGui::Spacing();
        ImGui::SeparatorText("Sampling and focal spot");
        integer("Rays / pixel", p.sampling.samples_per_pixel);
        ImGui::SetNextItemWidth(-185);
        ImGui::InputScalar("Sampling seed", ImGuiDataType_U32, &p.sampling.seed);
        ImGui::TextWrapped("1..4096 rays per pixel. Higher sampling improves focal/pixel averaging and increases calculation time.");
        ImGui::Checkbox("Finite normal focus", &p.focal_spot.enabled);
        if (p.focal_spot.enabled) {
            number("Normal U (mm)", p.focal_spot.normal_size_u_mm);
            number("Normal V (mm)", p.focal_spot.normal_size_v_mm);
        } else {
            ImGui::TextDisabled("Normal emission uses a point source.");
        }
        ImGui::Checkbox("Defocus", &p.focal_spot.defocus_enabled);
        if (p.focal_spot.defocus_enabled) {
            number("Defocus U (mm)", p.focal_spot.defocus_size_u_mm);
            number("Defocus V (mm)", p.focal_spot.defocus_size_v_mm);
            number("Defocus fraction", p.focal_spot.epsilon);
        }
        ImGui::Spacing();
        ImGui::SeparatorText("Scatter");
        ImGui::Checkbox("Scatter", &p.scatter.enabled);
        if (p.scatter.enabled) {
            pathInput("Scatter model", p.scatter.model_file, L"Scatter model\0*.json\0All files\0*.*\0\0");
            int kernel = p.scatter.kernel.mode == "segmented" ? 1 : 0;
            if (ImGui::Combo("Scatter kernel evaluation", &kernel, "All bins\0Segmented\0"))
                p.scatter.kernel.mode = kernel ? "segmented" : "all";
            if (kernel) {
                integer("Scatter kernel segments", p.scatter.kernel.segments);
                ImGui::TextWrapped("At least 2 segments. Fewer kernels reduce FFT work; kernel shapes are interpolated from the model.");
            }
        }
        ImGui::Spacing();
        ImGui::SeparatorText("Detector response");
        ImGui::Checkbox("Optical crosstalk", &p.detector_postprocess.optical_crosstalk.enabled);
        ImGui::Checkbox("Electronic crosstalk", &p.detector_postprocess.electronic_crosstalk.enabled);
        ImGui::Checkbox("Afterglow", &p.detector_postprocess.afterglow.enabled);
        if (p.detector_postprocess.afterglow.enabled) {
            auto& a = p.detector_postprocess.afterglow;
            number("Afterglow fraction", a.p);
            int init = a.initialization == "steady_state" ? 1 : 0;
            if (ImGui::Combo("Initial state", &init, "Zero\0Steady state\0"))
                a.initialization = init ? "steady_state" : "zero";
            number("Frame time (ms)", a.frame_time_ms);
            number("Exposure (ms)", a.exposure_time_ms);
            if (a.weights.empty()) {
                a.weights = { 1 };
                a.time_constants_ms = { 2 };
            }
            for (size_t i = 0; i < a.weights.size(); ++i) {
                ImGui::PushID(int(i));
                ImGui::Text("Temporal component %d", int(i) + 1);
                number("Weight", a.weights[i]);
                if (i < a.time_constants_ms.size())
                    number("Time constant (ms)", a.time_constants_ms[i]);
                const bool remove = ImGui::SmallButton("Remove component");
                ImGui::PopID();
                if (remove && a.weights.size() > 1) {
                    a.weights.erase(a.weights.begin() + i);
                    if (i < a.time_constants_ms.size())
                        a.time_constants_ms.erase(a.time_constants_ms.begin() + i);
                    break;
                }
            }
            if (ImGui::Button("Add component")) {
                a.weights.push_back(0);
                a.time_constants_ms.push_back(2);
            }
            ImGui::SameLine();
            if (ImGui::Button("Normalize weights")) {
                double sum = 0;
                for (double weight : a.weights)
                    sum += weight;
                if (sum > 0)
                    for (auto& weight : a.weights)
                        weight /= sum;
            }
            ImGui::TextWrapped("Weights must sum to 1. Exposure time must not exceed frame time.");
        }
        ImGui::Checkbox("DAS / electronic noise", &p.detector_postprocess.das.enabled);
        if (p.detector_postprocess.das.enabled)
            number("Noise std (electrons)", p.detector_postprocess.das.electronic_noise_std_electrons);
        ImGui::TextWrapped("Crosstalk kernels and detector layers retain settings from TOML. Sampling uses Rays / pixel above.");
    }
};
}