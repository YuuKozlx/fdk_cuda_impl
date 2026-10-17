#pragma once
#include "config/ReconstructionGeometry.hpp"
#include "ui/Widgets.hpp"
#include "ui/VolumePanel.hpp"
namespace yk::spectral::ui {
// Edits configuration and returns user intent; owns no computation, file cache or VTK state.
struct ReconstructionPanel {
    static void geometry(SimulationConfig& draft, bool& initialized)
    {
        auto& custom = draft.reconstruction.geometry_override;
        if (ImGui::Checkbox("Use independent geometry", &custom.enabled) && custom.enabled && !initialized) {
            custom.parameters = draft.geometry.parameters;
            initialized = true;
        }
        if (!custom.enabled)
            return;
        ImGui::Spacing();
        if (ImGui::Button("Copy scan geometry")) {
            custom.parameters = draft.geometry.parameters;
            initialized = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset offsets")) {
            for (const auto& field : reconstructionGeometryFields())
                if (field.member != &GeometryConfig::sid_mm && field.member != &GeometryConfig::sdd_mm)
                    custom.parameters.*field.member = 0;
        }
        for (const auto& field : reconstructionGeometryFields()) {
            auto& value = custom.parameters.*field.member;
            if (field.angle)
                angle(field.label, value);
            else
                number(field.label, value);
        }
        ImGui::TextDisabled("Applies to reconstruction only.");
    }
    static void algorithm(SimulationConfig& draft)
    {
        auto& r = draft.reconstruction;
        int method = r.type == "statistical" ? 2 : r.type == "iterative" ? 1
            : r.analytic.pipeline == "wfbp"                              ? 3
                                                                         : 0;
        if (ImGui::Combo("Method", &method, "FDK\0Algebraic iterative\0Statistical\0WFBP (helical)\0")) {
            r.type = method == 2 ? "statistical" : method == 1 ? "iterative"
                                                               : "analytic";
            r.analytic.pipeline = method == 3 ? "wfbp" : "fdk";
        }
        if (method == 0 || method == 3) {
            if (method == 3) {
                ImGui::TextWrapped("WFBP: analytic helical reconstruction. Uses the complete projection sequence.");
                if (draft.geometry.kind != GeometryKind::FlatHelical)
                    ImGui::TextWrapped("Select Flat helical in Geometry to use WFBP.");

            } else {
                ImGui::TextWrapped("FDK: Flat circular / Planar / Tilted. For Flat helical select WFBP or an iterative method.");
                integer("Chunk views", r.analytic.chunk_views);
            }
            const char* filters[] = { "ramlak", "shepp-logan", "cosine", "hann", "hamming", "blackman", "butterworth", "kaiser", "tukey" };
            if (ImGui::BeginCombo("Filter", r.analytic.filter.c_str())) {
                for (const auto* name : filters) {
                    bool selected = r.analytic.filter == name;
                    if (ImGui::Selectable(name, selected))
                        r.analytic.filter = name;
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            number("Cutoff (0, 0.5] cycles/sample", r.analytic.cutoff);
            number("Filter gain", r.analytic.gain);
            if (r.analytic.filter == "butterworth")
                number("Butterworth order", r.analytic.butterworth_order);
            if (r.analytic.filter == "kaiser")
                number("Kaiser beta", r.analytic.kaiser_beta);
            if (r.analytic.filter == "tukey")
                number("Tukey alpha", r.analytic.tukey_alpha);
        } else if (method == 1) {
            const char* algorithms[] = { "sirt", "ossart", "cgls", "tigre_sart", "tigre_os_sart", "tigre_sart_tv", "tigre_os_sart_tv", "sart", "tigre_sirt" };
            bool listed = false;
            for (const auto* algorithm : algorithms)
                listed = listed || r.iterative.algorithm == algorithm;
            if (ImGui::BeginCombo("Algorithm", listed ? r.iterative.algorithm.c_str() : "Choose supported algorithm")) {
                for (const auto* algorithm : algorithms) {
                    const bool selected = r.iterative.algorithm == algorithm;
                    if (ImGui::Selectable(algorithm, selected)) {
                        r.iterative.algorithm = algorithm;
                        if (r.iterative.algorithm != "ossart" && r.iterative.algorithm != "tigre_os_sart" && r.iterative.algorithm != "tigre_os_sart_tv")
                            r.iterative.subsets = 1;
                    }
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            const bool internal_regularization = r.iterative.algorithm == "sirt" || r.iterative.algorithm == "sart" || r.iterative.algorithm == "ossart";
            if (!internal_regularization)
                r.iterative.regularization.plugin = "none";
            if (!listed)
                ImGui::TextWrapped("This UI supports TV plugins with internal SIRT / SART / OS-SART only. Choose an algorithm from the list before reconstructing.");
            integer("Iterations", r.iterative.iterations);
            number("Relaxation", r.iterative.relaxation);
            if (r.iterative.algorithm == "ossart" || r.iterative.algorithm == "tigre_os_sart" || r.iterative.algorithm == "tigre_os_sart_tv")
                integer("Subsets", r.iterative.subsets);
            if (ImGui::TreeNode("Projectors / advanced")) {
                const char* fp_values[] = { "auto", "siddon", "joseph", "footprint" };
                int fp_index = 0;
                for (int i = 0; i < 4; ++i)
                    if (r.iterative.forward_projector == fp_values[i])
                        fp_index = i;
                if (ImGui::Combo("Forward projector", &fp_index, "Auto\0Siddon\0Joseph\0Separable footprint\0"))
                    r.iterative.forward_projector = fp_values[fp_index];
                const char* bp_values[] = { "auto", "fdk_pseudo_matched", "siddon_ray", "joseph", "joseph_v3", "footprint" };
                int bp_index = 0;
                for (int i = 0; i < 6; ++i)
                    if (r.iterative.back_projector == bp_values[i])
                        bp_index = i;
                if (ImGui::Combo("Backprojector", &bp_index, "Auto\0FDK pseudo matched\0Siddon ray driven\0Joseph matched\0Joseph V3 fast\0Separable footprint\0"))
                    r.iterative.back_projector = bp_values[bp_index];
                ImGui::TreePop();
            }
            ImGui::Checkbox("Non negative", &r.iterative.non_negative);
            if (r.iterative.algorithm == "tigre_sart_tv" || r.iterative.algorithm == "tigre_os_sart_tv") {
                ImGui::SeparatorText("TIGRE TV");
                integer("TV iterations", r.iterative.tv_iterations);
                number("TV alpha", r.iterative.tv_alpha);
            }
            if (internal_regularization) {
                ImGui::SeparatorText("Internal regularization");
                auto& tv = r.iterative.regularization;
                int plugin = tv.plugin == "huber_tv_prox" ? 2 : tv.plugin == "smoothed_tv" ? 1
                                                                                           : 0;
                if (ImGui::Combo("Regularization", &plugin, "None\0Smoothed TV\0Huber TV Prox\0")) {
                    tv.plugin = plugin == 2 ? "huber_tv_prox" : plugin == 1 ? "smoothed_tv"
                                                                            : "none";
                    if (plugin && tv.strength == 0)
                        tv.strength = plugin == 2 ? 1e-4 : .002;
                }
                if (plugin) {
                    number("TV strength", tv.strength);
                    if (plugin != 2)
                        integer("TV inner steps", tv.inner_iterations);
                    int dims = tv.dimensions == 2 ? 0 : 1;
                    if (ImGui::Combo("TV dimensions", &dims, "2D slices\0Full 3D\0"))
                        tv.dimensions = dims ? 3 : 2;
                    if (plugin != 2)
                        number("TV strength reduction", tv.strength_reduction);
                    if (plugin == 2) {
                        number("Huber delta", tv.huber_delta);
                        ImGui::TextWrapped("Prox solver accuracy is managed automatically.");
                    } else
                        number("TV epsilon", tv.epsilon);
                    ImGui::TextWrapped("Applied after every full angle sweep. Uses GPU volume buffers.");
                }
            }
        } else {
            auto& s = r.statistical;
            ImGui::SeparatorText("Data fidelity");
            ImGui::TextUnformatted("Weighted least squares");
            if (ImGui::TreeNode("Projectors / advanced")) {
                const char* fp_values[] = { "auto", "siddon", "joseph", "footprint" };
                int fp_index = 0;
                for (int i = 0; i < 4; ++i)
                    if (s.forward_projector == fp_values[i])
                        fp_index = i;
                if (ImGui::Combo("Forward projector", &fp_index, "Auto\0Siddon\0Joseph\0Separable footprint\0"))
                    s.forward_projector = fp_values[fp_index];
                const char* bp_values[] = { "auto", "fdk_pseudo_matched", "siddon_ray", "joseph", "joseph_v3", "footprint" };
                int bp_index = 0;
                for (int i = 0; i < 6; ++i)
                    if (s.back_projector == bp_values[i])
                        bp_index = i;
                if (ImGui::Combo("Backprojector", &bp_index, "Auto\0FDK pseudo matched\0Siddon ray driven\0Joseph matched\0Joseph V3 fast\0Separable footprint\0"))
                    s.back_projector = bp_values[bp_index];
                ImGui::TreePop();
            }
            number("Data scale", s.data_fidelity.scale);
            ImGui::Checkbox("Mean data term", &s.data_fidelity.normalize_by_ray_count);
            pathInput("Weights (optional)", s.data_fidelity.weights_file, L"Weights\0*.raw\0All files\0*.*\0\0");
            ImGui::TextWrapped("Empty weights file uses W=I. Weight RAW must match projection dimensions and contain nonnegative float32 values.");
            ImGui::SeparatorText("Regularization");
            auto& reg = s.regularization;
            int plugin = reg.plugin == "smoothed_tv" ? 1 : (reg.plugin == "huber_tv" || reg.plugin == "huber_tv_prox") ? 2
                : reg.plugin == "tikhonov"                                                                             ? 3
                : reg.plugin == "tv"                                                                                   ? 4
                                                                                                                       : 0;
            if (ImGui::Combo("Penalty", &plugin, "None\0Smoothed TV\0Huber TV\0Tikhonov\0Standard TV\0"))
                reg.plugin = plugin == 1 ? "smoothed_tv" : plugin == 2 ? "huber_tv"
                    : plugin == 3                                      ? "tikhonov"
                    : plugin == 4                                      ? "tv"
                                                                       : "none";
            if (plugin) {
                number("Penalty weight", reg.strength);
                if (plugin == 1 || plugin == 2)
                    number(plugin == 1 ? "TV smooth scale" : "Huber threshold", reg.shape);
                if (plugin != 3) {
                    int dims = reg.dimensions == 2 ? 0 : 1;
                    if (ImGui::Combo("Penalty dimensions", &dims, "2D slices\0Full 3D\0"))
                        reg.dimensions = dims ? 3 : 2;
                }
            } else
                reg.strength = 0;
            ImGui::SeparatorText("Optimizer");
            int update = (s.optimizer.type == "raw_gradient_descent" || s.optimizer.type == "gradient_descent") ? 1 : 0;
            if (ImGui::Combo("Data update", &update, "Regularized algebraic (SIRT)\0Gradient descent\0")) {
                s.optimizer.type = update ? "gradient_descent" : "regularized_algebraic";
            }
            ImGui::TextWrapped(update ? "One full-batch gradient update per iteration." : "One normalized full-view SIRT update per sweep. Zero penalty matches internal SIRT.");
            integer(update ? "GD iterations" : "Algebraic sweeps", s.optimizer.iterations);
            number(update ? "GD step (0 = auto)" : "Penalty step (0 = auto)", s.optimizer.step_size);
            if (update || (plugin != 2 && plugin != 4))
                ImGui::Checkbox(update ? "Objective backtracking" : "Penalty backtracking", &s.optimizer.backtracking);
            if (s.optimizer.backtracking && (update || (plugin != 2 && plugin != 4))) {
                integer("Maximum backtracking", s.optimizer.max_backtracking);
                number("Step shrink", s.optimizer.step_shrink);
            }
            number("Relative update tolerance", s.optimizer.relative_tolerance);
            integer("Minimum iterations", s.optimizer.minimum_iterations);
            ImGui::Checkbox("Non negative volume", &s.non_negative);
            ImGui::TextWrapped("Regularization integration is selected internally. FISTA is not yet available.");
            if (update && plugin == 4)
                ImGui::TextWrapped("Standard TV is available with Regularized algebraic.");
        }
    }
    static void output(SimulationConfig& draft)
    {
        auto& r = draft.reconstruction;
        auto& g = draft.geometry.parameters;

        integer("Recon X", g.reconstruction_volume_x);
        integer("Recon Y", g.reconstruction_volume_y);
        integer("Recon Z", g.reconstruction_volume_z);
        number("Recon voxel X", g.reconstruction_voxel_x_mm);
        number("Recon voxel Y", g.reconstruction_voxel_y_mm);
        number("Recon voxel Z", g.reconstruction_voxel_z_mm);
        pathInput("Input projection", r.input_projection_file, L"Projection\0*.raw\0All files\0*.*\0\0");
        std::string path = r.output_volume_file.generic_string();
        stringInput("Output volume", path);
        r.output_volume_file = path;
        ImGui::Checkbox("Save orthogonal slices", &r.save_slices);
        if (r.save_slices) {
            path = r.slice_prefix.generic_string();
            stringInput("Slice prefix", path);
            r.slice_prefix = path;
        }
    }
    static void corrections(SimulationConfig& draft)
    {
        auto& r = draft.reconstruction;

        auto& pp = r.preprocessing;
        ImGui::Checkbox("Enable preprocessing", &pp.enabled);
        if (pp.enabled)
            ImGui::TextWrapped("Corrected projections are saved beside the input and reused when correction inputs and settings match.");
        if (r.type == "iterative" && r.iterative.algorithm == "tigre_sart")
            ImGui::TextWrapped("SART updates one view at a time. One iteration is a complete sweep over all views; large volumes can take several minutes per sweep.");
        if (pp.enabled) {
            auto& bh = pp.beam_hardening;
            ImGui::Checkbox("Beam hardening correction", &bh.enabled);
            if (bh.enabled) {
                ImGui::Checkbox("Fit water plate", &bh.fit_from_water_plate);
                integer("Polynomial order", bh.order);
                number("Clamp min", bh.clamp_min);
                number("Clamp max", bh.clamp_max);
                ImGui::TextWrapped("Clamp bounds apply to corrected line integrals, not geometric length.");
                number("Reference (keV)", bh.reference_energy_keV);
                if (bh.fit_from_water_plate) {
                    number("Water max (mm)", bh.water_plate_max_mm);
                    integer("Water samples", bh.water_plate_samples);
                } else {
                    if (bh.order >= 1 && bh.order <= 16)
                        bh.coefficients.resize(size_t(bh.order) + 1, 0);
                    for (size_t i = 0; i < bh.coefficients.size(); ++i) {
                        const auto label = "Coefficient c" + std::to_string(i);
                        number(label.c_str(), bh.coefficients[i]);
                    }
                    pathInput("BHC spectrum", bh.spectrum_file, L"Spectrum\0*.spec;*.csv;*.txt\0All files\0*.*\0\0");
                }
            }
            if (ImGui::Checkbox("Scatter correction", &pp.scatter.enabled) && pp.scatter.enabled) {
                if (pp.scatter.air_projection_file.empty())
                    pp.scatter.air_projection_file = draft.projection.air_output_file;
                if (pp.scatter.model_file.empty())
                    pp.scatter.model_file = draft.projection.scatter.model_file;
            }
            if (pp.scatter.enabled) {
                auto& sc = pp.scatter;
                pathInput("Correction air", sc.air_projection_file, L"Air projection\0*.raw\0All files\0*.*\0\0");
                pathInput("Correction model", sc.model_file, L"Scatter model\0*.json\0All files\0*.*\0\0");
                if (ImGui::Button("Use forward scatter inputs")) {
                    sc.air_projection_file = draft.projection.air_output_file;
                    sc.model_file = draft.projection.scatter.model_file;
                }
                int solver = sc.solver.type == "adaptive_experimental" ? 1 : 0;
                if (ImGui::Combo("Solver", &solver, "Fixed\0Adaptive (experimental)\0"))
                    sc.solver.type = solver ? "adaptive_experimental" : "fixed";
                if (!solver) {
                    integer("Correction iterations", sc.correction.iterations);
                    number("Damping", sc.correction.damping);
                } else {
                    auto& a = sc.solver.adaptive;
                    integer("Adaptive iterations", a.iterations);
                    number("Initial damping", a.initial_damping);
                    number("Minimum damping", a.min_damping);
                    number("Damping decay", a.damping_decay);
                    if (ImGui::CollapsingHeader("Transmission guard")) {
                        number("Low quantile", a.transmission_guard.low_quantile);
                        number("Risk threshold", a.transmission_guard.risk_threshold);
                        number("Positivity margin", a.transmission_guard.positivity_margin);
                    }
#ifndef YK_SPECTRAL_ADAPTIVE_SCATTER
                    ImGui::TextWrapped("This build requires MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER=ON to run adaptive correction.");
#endif
                }
                number("Scatter scale", sc.correction.scatter_scale);
                number("Mask threshold", sc.correction.object_mask_threshold);
                ImGui::Checkbox("Scatter guidance", &sc.guide.enabled);
                if (sc.guide.enabled) {
                    number("Guide sigma U (px)", sc.guide.sigma_u_pixels);
                    number("Guide sigma V (px)", sc.guide.sigma_v_pixels);
                }
                int kernel = sc.kernel.mode == "segmented" ? 1 : 0;
                if (ImGui::Combo("Kernel evaluation", &kernel, "All bins\0Segmented\0"))
                    sc.kernel.mode = kernel ? "segmented" : "all";
                if (kernel)
                    integer("Kernel segments", sc.kernel.segments);
                ImGui::TextWrapped("Correction uses processed air and energy-compatible calibration. Paths are checked before reconstruction.");
            }
        }
    }
    static void boneCorrection(SimulationConfig& draft)
    {
        auto& b=draft.reconstruction.bone_hardening;
        if(ImGui::Checkbox("Enable bone correction (experimental)",&b.enabled) && b.enabled){
            auto& pp=draft.reconstruction.preprocessing;
            if(!pp.beam_hardening.enabled){pp.beam_hardening.enabled=true;pp.beam_hardening.fit_from_water_plate=true;}
            pp.enabled=true;
            if(pp.beam_hardening.reference_energy_keV<=0)pp.beam_hardening.reference_energy_keV=70;
        }
        if(!b.enabled)return;
        const char* values[]={"physical_after_water","physical_before_water","two_material_lut"};
        int selected=b.method==values[2]?2:b.method==values[1]?1:0;
        ImGui::SetNextItemWidth(-1);
        if(ImGui::Combo("##Bone method",&selected,"Physical model after water BHC\0Physical model before water BHC\0Two-material LUT before water BHC\0"))b.method=values[selected];
        ImGui::TextWrapped(selected==0?
            "Water correction -> initial reconstruction -> bone model correction -> reconstruction. Water BHC is not repeated.":
            "Scatter-corrected log -> bone correction -> reconstruction. Water BHC is used only to initialize segmentation.");
        if(!draft.reconstruction.preprocessing.enabled)
            ImGui::TextWrapped(selected==0?"Preprocessing is off: input must already be water-corrected log.":
                "Preprocessing is off: input must already be scatter-corrected log before water BHC.");
        integer("Bone correction rounds",b.max_iterations);
        number("Air threshold (HU)",b.air_threshold_hu);number("Bone threshold (HU)",b.bone_threshold_hu);
        if(selected==2){
            pathInput("Two-material calibration",b.calibration_file,L"Calibration\0*.json\0All files\0*.*\0\0");
            number("Equivalent bone thickness scale",b.bone_thickness_scale);
            ImGui::TextWrapped("Calibration must declare a matching reference energy and equivalent bone thickness basis. LUT replaces water BHC on the final projection.");
        }else{
            if(b.max_iterations>1)number("Stop ratio",b.stop_ratio);
            static VolumePanel soft_picker,bone_picker;
            ImGui::PushID("bone-soft");ImGui::TextUnformatted("Soft tissue model");soft_picker.builtinPicker(b.soft_material);
            number("Density (g/cm3)",b.soft_material.density_g_cm3);ImGui::PopID();
            ImGui::PushID("bone-bone");ImGui::TextUnformatted("Bone model");bone_picker.builtinPicker(b.bone_material);
            number("Density (g/cm3)",b.bone_material.density_g_cm3);ImGui::PopID();
        }
        ImGui::TextWrapped("Each round resegments the current reconstruction and corrects the fixed input projection. Intermediate results are saved beside the output volume.");
        ImGui::TextDisabled("EBHC uses a separate image-domain stage and is not selected here.");
    }
    static void draw(SimulationConfig& draft, bool& automatic, bool& initialized)
    {
        ImGui::TextUnformatted("Reconstruction");
        ImGui::TextDisabled("Choose a method, then set the output.");
        ImGui::Spacing();
        ImGui::Checkbox("Reconstruct after projection", &automatic);
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Method", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("method");
            algorithm(draft);
            ImGui::PopID();
        }
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Volume and files", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("output");
            output(draft);
            ImGui::PopID();
        }
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Projection corrections")) {
            ImGui::PushID("corrections");
            corrections(draft);
            ImGui::PopID();
        }
        ImGui::Spacing();
        // Keep the switch accessible, but never draw inactive geometry fields.
        if(ImGui::CollapsingHeader("Bone hardening correction")){
            ImGui::PushID("bone-hardening");boneCorrection(draft);ImGui::PopID();
        }
        ImGui::Spacing();
        ImGui::SeparatorText("Geometry");
        ImGui::PushID("geometry");
        geometry(draft, initialized);
        ImGui::PopID();
    }
};
}
