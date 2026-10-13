#include "MultiSpecFpStage.hpp"
#include <windows.h>

#include "ImGuiProjection.hpp"
#include "ProjectionUiConfig.hpp"
#include "LibraryReconstructor.hpp"
#include "SimLogger.hpp"
#include "ReconstructionDisplay.hpp"
#include "WaterReference.hpp"
#include "SpectralProjectionPipeline.hpp"
#include "XcomAttenuationProvider.hpp"
#include "primary/CudaProjectionBackend.hpp"
#include "primary/DisplayGeometry.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <commdlg.h>
#include <fstream>
#include <future>
#include <optional>
#include <mutex>
#include <iostream>
#include <imgui.h>
#include <sstream>
#include <thread>

namespace yk::spectral {
namespace {
    constexpr double degrees = 57.29577951308232;
    std::filesystem::path chooseFile(bool save, const wchar_t* filter)
    {
        std::array<wchar_t, 32768> buffer {};
        OPENFILENAMEW info {};
        info.lStructSize = sizeof(info);
        info.lpstrFile = buffer.data();
        info.nMaxFile = DWORD(buffer.size());
        info.lpstrFilter = filter;
        info.nFilterIndex = 1;
        info.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST | (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
        info.hwndOwner = GetActiveWindow();
        if (save ? GetSaveFileNameW(&info) : GetOpenFileNameW(&info))
            return buffer.data();
        return {};
    }
    void stringInput(const char* name, std::string& text)
    {
        std::array<char, 2048> b {};
        std::copy_n(text.c_str(), std::min(text.size(), b.size() - 1), b.data());
        if (ImGui::InputText(name, b.data(), b.size()))
            text = b.data();
    }
    void pathInput(const char* name, std::filesystem::path& path, const wchar_t* filter)
    {
        ImGui::PushID(name);
        std::string s = path.generic_string();
        stringInput(name, s);
        path = s;
        if (ImGui::Button("Browse...")) {
            const auto selected = chooseFile(false, filter);
            if (!selected.empty())
                path = selected;
        }
        ImGui::PopID();
    }
    std::string shortLabel(std::string text,float width) {
    if(ImGui::CalcTextSize(text.c_str()).x<=width)return text;
    while(!text.empty() && ImGui::CalcTextSize((text+"...").c_str()).x>width){
        std::size_t start=text.size()-1;
        while(start>0 && (static_cast<unsigned char>(text[start])&0xc0)==0x80)--start;
        text.resize(start);
    }
    return text+"...";
}
void number(const char* label, double& value)
    {
        ImGui::SetNextItemWidth(-185);
        ImGui::InputDouble(label, &value, 0, 0, "%.6g");
    }
    void integer(const char* label, int& value)
    {
        ImGui::SetNextItemWidth(-185);
        ImGui::InputInt(label, &value);
    }
    void angle(const char* label, double& value)
    {
        double d = value * degrees;
        number(label, d);
        value = d / degrees;
    }
    struct UiState {
        SimulationConfig draft;
        std::vector<float> spectrum_plot;
        std::string message;
        bool calculate = false, load = false, save = false, preview = false, export_spectrum = false;
        bool reconstruct=false, auto_reconstruct=false, opacity_changed=false;
        float opacity_scale=.03f;
        int result_mode=0;
        bool relative_mode=false,relative_changed=false;
        bool drawer_open=false,drawer_changed=false;
        bool select_image_page=true;
        bool result_changed=false,slices_changed=false,window_changed=false,auto_window=false,load_reconstruction=false;
        std::array<int,3> recon_dimensions{},recon_slices{};
        bool select_results=false;
        double window_low=0,window_high=1;
        WaterReference ct_reference;
        bool reconstruction_running=false;
        int progress_completed=0,progress_total=0;
        ImGuiTextFilter material_filter;
        int material_category=0;
        bool show_beam = true, show_trajectory = true;
        int requested_frame = -1, shown_frame = 0;
        std::array<std::uint64_t, 256> counts {};
        void histogram(const std::vector<std::uint8_t>& labels)
        {
            counts.fill(0);
            for (auto label : labels)
                ++counts[label];
        }
        void spectrum(const std::vector<SpectrumPoint>& points)
        {
            spectrum_plot.clear();
            for (auto p : points)
                spectrum_plot.push_back(float(p.fluence_per_keV_cm2_mAs_at_1m));
        }
        void scan()
        {
            const GeometryKind kinds[] = { GeometryKind::FlatCbct, GeometryKind::FlatHelical, GeometryKind::PlanarSourceOrbit, GeometryKind::NikonTilted };
            int index = 0;
            for (int i = 0; i < 4; ++i)
                if (draft.geometry.kind == kinds[i])
                    index = i;
            auto& g = draft.geometry.parameters;
            if (ImGui::Combo("Scan type", &index, "Flat circular\0Flat helical\0Planar CT\0Tilted CT\0")) {
                draft.geometry.kind = kinds[index];
                if (index == 1 && g.pitch_mm_per_turn == 0)
                    g.pitch_mm_per_turn = 100;
                if (index == 2) {
                    if (g.planar_detector_y_mm <= 0)
                        g.planar_detector_y_mm = std::max(g.sdd_mm - g.sid_mm, 1.0);
                    if (g.source_orbit_radius_mm <= 0)
                        g.source_orbit_radius_mm = g.sid_mm;
                    g.reconstruction_detector_u = g.detector_u;
                    g.reconstruction_detector_v = g.detector_v;
                }
                if (index >= 2)
                    draft.projection.detector_response.enabled = false;
            }
            integer("Views", g.views);
            integer("Views / turn", g.views_per_turn);
            angle("Start angle (deg)", g.start_angle_rad);
            integer("Direction (+/-1)", g.rotation_direction);
            if (index == 1) {
                number("Pitch (mm / turn)", g.pitch_mm_per_turn);
                number("Start Z (mm)", g.start_z_mm);
            }
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
                number("Offset U (mm)", g.offset_u_mm);
                number("Offset N (mm)", g.offset_n_mm);
                number("Offset V (mm)", g.offset_v_mm);
                angle("Tilt U (deg)", g.tilt_u_rad);
                angle("Tilt V (deg)", g.tilt_v_rad);
                angle("Tilt N (deg)", g.tilt_n_rad);
            }
            if (ImGui::CollapsingHeader("Source")) {
                number("Source X (mm)", g.source_offset_x_mm);
                number("Source Y (mm)", g.source_offset_y_mm);
                number("Source Z (mm)", g.source_offset_z_mm);
            }
        }
        void builtinPicker(MaterialSpec& material) {
            const auto* selected=findBuiltinMaterial(material.preset);
            const std::string preview=selected?std::string(selected->display_name)+" ["+selected->category+"]":material.preset;
            const float popup_width=std::clamp(ImGui::GetContentRegionAvail().x,240.f,430.f);
            ImGui::SetNextItemWidth(-100);
            ImGui::SetNextWindowSizeConstraints({popup_width,0},{popup_width,480});
            const auto preview_short=shortLabel(preview,popup_width-150);
            if(ImGui::BeginCombo("Material",preview_short.c_str(),ImGuiComboFlags_HeightLargest)){
                material_filter.Draw("Search",-1);
                ImGui::SetNextItemWidth(-85);
                ImGui::Combo("Category",&material_category,"All\0Tissue\0Element\0Compound\0Mixture\0");
                const char* categories[]={"","tissue","element","compound","mixture"};
                ImGui::BeginChild("Material list",{0,260},false);
                int shown=0;
                for(auto* info=builtinMaterialsBegin();info!=builtinMaterialsEnd();++info){
                    if(material_category && std::string_view(info->category)!=categories[material_category])continue;
                    const std::string searchable=std::string(info->id)+" "+info->display_name+" "+info->aliases;
                    if(!material_filter.PassFilter(searchable.c_str()))continue;
                    ++shown;ImGui::PushID(info->id);
                    const std::string full_label=std::string(info->display_name)+" ["+info->category+"]";
                    const auto label=shortLabel(full_label,ImGui::GetContentRegionAvail().x-10);
                    if(ImGui::Selectable(label.c_str(),selected && std::string_view(info->id)==selected->id)){
                        selectBuiltinMaterial(material,info->id);ImGui::CloseCurrentPopup();}
                    if(ImGui::IsItemHovered()){
                        ImGui::BeginTooltip();ImGui::PushTextWrapPos(ImGui::GetFontSize()*24);
                        ImGui::TextWrapped("%s",info->display_name);ImGui::TextWrapped("%s",info->id);
                        ImGui::Text("Default density: %.6g g/cm3",info->density_g_cm3);
                        ImGui::TextWrapped("Element mass fractions: %s",info->composition);ImGui::PopTextWrapPos();ImGui::EndTooltip();}
                    ImGui::PopID();
                }
                if(!shown)ImGui::TextUnformatted("No matching materials");
                ImGui::EndChild();
                ImGui::Text("%d matching / %d builtin materials",shown,int(builtinMaterialsEnd()-builtinMaterialsBegin()));
                ImGui::EndCombo();
            }
            selected=findBuiltinMaterial(material.preset);
            if(selected){
                ImGui::TextWrapped("ID: %s",selected->id);
                ImGui::Text("Default density: %.6g g/cm3",selected->density_g_cm3);
                if(ImGui::SmallButton("Use default density"))material.density_g_cm3=selected->density_g_cm3;
            }
        }
        void volume()
        {
            auto& g = draft.geometry.parameters;
            ImGui::TextWrapped("Raw uint8 labels, order Z/Y/X. Full voxel resolution is retained.");
            pathInput("Label volume", draft.projection.label_volume, L"Raw labels\0*.raw;*.bin\0All files\0*.*\0\0");
            integer("Size X", g.volume_x);
            integer("Size Y", g.volume_y);
            integer("Size Z", g.volume_z);
            number("Voxel X (mm)", g.voxel_x_mm);
            number("Voxel Y (mm)", g.voxel_y_mm);
            number("Voxel Z (mm)", g.voxel_z_mm);
            if (ImGui::CollapsingHeader("Object pose", ImGuiTreeNodeFlags_DefaultOpen)) {
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
                if (ImGui::TreeNodeEx("Material", ImGuiTreeNodeFlags_DefaultOpen, "Label %d: %s", int(m.label), m.name.c_str())) {
                    int label = int(m.label);
                    if (ImGui::InputInt("Label", &label))
                        m.label = std::uint8_t(std::clamp(label, 0, 255));
                    ImGui::Text("Voxels: %llu", static_cast<unsigned long long>(counts[m.label]));
                    stringInput("Name", m.name);
                    int mode=m.preset.empty()?1:0;
                    if(ImGui::Combo("Material source",&mode,"Builtin list\0Chemical formula\0")){
                        if(mode==0)selectBuiltinMaterial(m,"compound.water");
                        else{m.preset.clear();m.formula="H2O";}
                    }
                    if(mode==0)builtinPicker(m);
                    else{stringInput("Formula",m.formula);ImGui::TextWrapped("Formula uses atom counts, e.g. H2O. Density is specified separately.");}
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
            if (materials.size() < 32 && ImGui::Button("Add material")) {
                MaterialSpec m;
                m.label = 1;
                for (const auto& x : materials)
                    if (m.label <= x.label)
                        m.label = x.label + 1;
                selectBuiltinMaterial(m,"compound.water");
                materials.push_back(m);
            }
        }
        void energy()
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
        void effects()
        {
            auto& p = draft.projection;
            ImGui::Checkbox("Geometry flux", &p.geometry_flux.enabled);
            ImGui::Checkbox("Quantum noise", &p.quantum_noise.enabled);
            ImGui::SetNextItemWidth(-185);ImGui::InputScalar("Noise seed",ImGuiDataType_U32,&p.quantum_noise.seed);
            ImGui::SeparatorText("Projection batches");
            integer("Batch views",p.batching.views);
            integer("Batch buffer MiB",p.batching.max_buffer_mib);
            ImGui::TextWrapped("Defaults: automatic batches up to 256 MiB (64-view safety limit). Actual batch size adapts to available GPU memory and reserves space for VTK. Set views to 1 for comparison.");
            ImGui::SeparatorText("Ray sampling");
            integer("Rays / pixel",p.sampling.samples_per_pixel);
            ImGui::SetNextItemWidth(-185);ImGui::InputScalar("Sampling seed",ImGuiDataType_U32,&p.sampling.seed);
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
            ImGui::Checkbox("Optical crosstalk", &p.detector_postprocess.optical_crosstalk.enabled);
            ImGui::Checkbox("Electronic crosstalk", &p.detector_postprocess.electronic_crosstalk.enabled);
            ImGui::Checkbox("Afterglow", &p.detector_postprocess.afterglow.enabled);
            if (p.detector_postprocess.afterglow.enabled){
                auto& a=p.detector_postprocess.afterglow;
                number("Afterglow fraction",a.p);
                int init=a.initialization=="steady_state"?1:0;
                if(ImGui::Combo("Initial state",&init,"Zero\0Steady state\0"))a.initialization=init?"steady_state":"zero";
                number("Frame time (ms)",a.frame_time_ms);number("Exposure (ms)",a.exposure_time_ms);
                if(a.weights.empty()){a.weights={1};a.time_constants_ms={2};}
                for(size_t i=0;i<a.weights.size();++i){ImGui::PushID(int(i));
                    ImGui::Text("Temporal component %d",int(i)+1);number("Weight",a.weights[i]);
                    if(i<a.time_constants_ms.size())number("Time constant (ms)",a.time_constants_ms[i]);
                    const bool remove=ImGui::SmallButton("Remove component");ImGui::PopID();
                    if(remove && a.weights.size()>1){a.weights.erase(a.weights.begin()+i);if(i<a.time_constants_ms.size())a.time_constants_ms.erase(a.time_constants_ms.begin()+i);break;}}
                if(ImGui::Button("Add component")){a.weights.push_back(0);a.time_constants_ms.push_back(2);}
                ImGui::SameLine();if(ImGui::Button("Normalize weights")){double sum=0;for(double weight:a.weights)sum+=weight;if(sum>0)for(auto& weight:a.weights)weight/=sum;}
                ImGui::TextWrapped("Weights must sum to 1. Exposure time must not exceed frame time.");
            }
            ImGui::Checkbox("DAS / electronic noise", &p.detector_postprocess.das.enabled);
            if (p.detector_postprocess.das.enabled)
                number("Noise std (electrons)", p.detector_postprocess.das.electronic_noise_std_electrons);
            ImGui::TextWrapped("Crosstalk kernels and detector layers retain settings from TOML. Sampling uses Rays / pixel above.");
        }
        void reconstruction() {
            auto& r=draft.reconstruction;auto& g=draft.geometry.parameters;
            ImGui::Checkbox("Reconstruct after projection",&auto_reconstruct);
            int method=r.type=="statistical"?2:r.type=="iterative"?1:r.analytic.pipeline=="wfbp"?3:0;
            if(ImGui::Combo("Method",&method,"FDK\0Algebraic iterative\0Statistical\0WFBP (helical)\0")){
                r.type=method==2?"statistical":method==1?"iterative":"analytic";
                r.analytic.pipeline=method==3?"wfbp":"fdk";

            }
            if(method==0||method==3){
                if(method==3){
                    ImGui::TextWrapped("WFBP: analytic helical reconstruction. Uses the complete projection sequence.");
                    if(draft.geometry.kind!=GeometryKind::FlatHelical)ImGui::TextWrapped("Select Flat helical in Geometry to use WFBP.");

                }else{
                    ImGui::TextWrapped("FDK: Flat circular / Planar / Tilted. For Flat helical select WFBP or an iterative method.");
                    integer("Chunk views",r.analytic.chunk_views);
                }
                const char* filters[]={"ramlak","shepp-logan","cosine","hann","hamming","blackman","butterworth","kaiser","tukey"};
                if(ImGui::BeginCombo("Filter",r.analytic.filter.c_str())) {
                    for(const auto* name:filters) {
                        bool selected=r.analytic.filter==name;
                        if(ImGui::Selectable(name,selected))r.analytic.filter=name;
                        if(selected)ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                number("Cutoff (0, 0.5] cycles/sample",r.analytic.cutoff);
                number("Filter gain",r.analytic.gain);
                if(r.analytic.filter=="butterworth")number("Butterworth order",r.analytic.butterworth_order);
                if(r.analytic.filter=="kaiser")number("Kaiser beta",r.analytic.kaiser_beta);
                if(r.analytic.filter=="tukey")number("Tukey alpha",r.analytic.tukey_alpha);
            }else if(method==1){
                const char* algorithms[]={"sirt","ossart","cgls","tigre_sart","tigre_os_sart","tigre_sart_tv","tigre_os_sart_tv","sart","tigre_sirt"};
                bool listed=false;for(const auto* algorithm:algorithms)listed=listed || r.iterative.algorithm==algorithm;
                if(ImGui::BeginCombo("Algorithm",listed?r.iterative.algorithm.c_str():"Choose supported algorithm")){
                    for(const auto* algorithm:algorithms){
                        const bool selected=r.iterative.algorithm==algorithm;
                        if(ImGui::Selectable(algorithm,selected)){
                            r.iterative.algorithm=algorithm;
                            if(r.iterative.algorithm!="ossart" && r.iterative.algorithm!="tigre_os_sart" && r.iterative.algorithm!="tigre_os_sart_tv")r.iterative.subsets=1;
                        }
                        if(selected)ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                const bool internal_regularization=r.iterative.algorithm=="sirt" || r.iterative.algorithm=="sart" || r.iterative.algorithm=="ossart";
                if(!internal_regularization)r.iterative.regularization.plugin="none";
                if(!listed)ImGui::TextWrapped("This UI supports TV plugins with internal SIRT / SART / OS-SART only. Choose an algorithm from the list before reconstructing.");
                integer("Iterations",r.iterative.iterations);number("Relaxation",r.iterative.relaxation);
                if(r.iterative.algorithm=="ossart" || r.iterative.algorithm=="tigre_os_sart" || r.iterative.algorithm=="tigre_os_sart_tv")integer("Subsets",r.iterative.subsets);
                ImGui::Checkbox("Non negative",&r.iterative.non_negative);
                if(r.iterative.algorithm=="tigre_sart_tv" || r.iterative.algorithm=="tigre_os_sart_tv"){
                    ImGui::SeparatorText("TIGRE TV");
                    integer("TV iterations",r.iterative.tv_iterations);number("TV alpha",r.iterative.tv_alpha);
                }
                if(internal_regularization){
                    ImGui::SeparatorText("Internal regularization");
                    auto& tv=r.iterative.regularization;
                    int plugin=tv.plugin=="huber_tv_prox"?2:tv.plugin=="smoothed_tv"?1:0;
                    if(ImGui::Combo("Regularization",&plugin,"None\0Smoothed TV\0Huber TV Prox\0")){
                        tv.plugin=plugin==2?"huber_tv_prox":plugin==1?"smoothed_tv":"none";
                        if(plugin&&tv.strength==0)tv.strength=plugin==2?1e-4:.002;
                    }
                    if(plugin){
                        number("TV strength",tv.strength);if(plugin!=2)integer("TV inner steps",tv.inner_iterations);
                        int dims=tv.dimensions==2?0:1;
                        if(ImGui::Combo("TV dimensions",&dims,"2D slices\0Full 3D\0"))tv.dimensions=dims?3:2;
                        if(plugin!=2)number("TV strength reduction",tv.strength_reduction);
                        if(plugin==2){number("Huber delta",tv.huber_delta);ImGui::TextWrapped("Prox solver accuracy is managed automatically.");}
                        else number("TV epsilon",tv.epsilon);
                        ImGui::TextWrapped("Applied after every full angle sweep. Uses GPU volume buffers.");
                    }
                }
            }else{
                auto& s=r.statistical;
                ImGui::SeparatorText("Data fidelity");
                ImGui::TextUnformatted("Weighted least squares");
                int model=s.projection_model=="siddon"?1:0;
                if(ImGui::Combo("Projector",&model,"Joseph\0Siddon\0"))s.projection_model=model?"siddon":"joseph";
                number("Data scale",s.data_fidelity.scale);
                ImGui::Checkbox("Mean data term",&s.data_fidelity.normalize_by_ray_count);
                pathInput("Weights (optional)",s.data_fidelity.weights_file,L"Weights\0*.raw\0All files\0*.*\0\0");
                ImGui::TextWrapped("Empty weights file uses W=I. Weight RAW must match projection dimensions and contain nonnegative float32 values.");
                ImGui::SeparatorText("Regularization");
                auto& reg=s.regularization;
                int plugin=reg.plugin=="smoothed_tv"?1:(reg.plugin=="huber_tv"||reg.plugin=="huber_tv_prox")?2:reg.plugin=="tikhonov"?3:reg.plugin=="tv"?4:0;
                if(ImGui::Combo("Penalty",&plugin,"None\0Smoothed TV\0Huber TV\0Tikhonov\0Standard TV\0"))reg.plugin=plugin==1?"smoothed_tv":plugin==2?"huber_tv":plugin==3?"tikhonov":plugin==4?"tv":"none";
                if(plugin){number("Penalty weight",reg.strength);
                    if(plugin==1||plugin==2)number(plugin==1?"TV smooth scale":"Huber threshold",reg.shape);
                    if(plugin!=3){int dims=reg.dimensions==2?0:1;if(ImGui::Combo("Penalty dimensions",&dims,"2D slices\0Full 3D\0"))reg.dimensions=dims?3:2;}}
                else reg.strength=0;
                ImGui::SeparatorText("Optimizer");
                int update=(s.optimizer.type=="raw_gradient_descent"||s.optimizer.type=="gradient_descent")?1:0;
                if(ImGui::Combo("Data update",&update,"Regularized algebraic (SIRT)\0Gradient descent\0")){s.optimizer.type=update?"gradient_descent":"regularized_algebraic";}
                ImGui::TextWrapped(update?"One full-batch gradient update per iteration.":"One normalized full-view SIRT update per sweep. Zero penalty matches internal SIRT.");
                integer(update?"GD iterations":"Algebraic sweeps",s.optimizer.iterations);number(update?"GD step (0 = auto)":"Penalty step (0 = auto)",s.optimizer.step_size);
                if(update||(plugin!=2&&plugin!=4))ImGui::Checkbox(update?"Objective backtracking":"Penalty backtracking",&s.optimizer.backtracking);
                if(s.optimizer.backtracking&&(update||(plugin!=2&&plugin!=4))){integer("Maximum backtracking",s.optimizer.max_backtracking);number("Step shrink",s.optimizer.step_shrink);}
                number("Relative update tolerance",s.optimizer.relative_tolerance);integer("Minimum iterations",s.optimizer.minimum_iterations);
                ImGui::Checkbox("Non negative volume",&s.non_negative);
                ImGui::TextWrapped("Regularization integration is selected internally. FISTA is not yet available.");
                if(update&&plugin==4)ImGui::TextWrapped("Standard TV is available with Regularized algebraic.");
            }
            ImGui::SeparatorText("Output grid");
            integer("Recon X",g.reconstruction_volume_x);integer("Recon Y",g.reconstruction_volume_y);integer("Recon Z",g.reconstruction_volume_z);
            number("Recon voxel X",g.reconstruction_voxel_x_mm);number("Recon voxel Y",g.reconstruction_voxel_y_mm);number("Recon voxel Z",g.reconstruction_voxel_z_mm);
            pathInput("Input projection",r.input_projection_file,L"Projection\0*.raw\0All files\0*.*\0\0");
            std::string path=r.output_volume_file.generic_string();stringInput("Output volume",path);r.output_volume_file=path;
            ImGui::Checkbox("Save orthogonal slices",&r.save_slices);path=r.slice_prefix.generic_string();stringInput("Slice prefix",path);r.slice_prefix=path;
            ImGui::SeparatorText("Corrections");
            auto& pp=r.preprocessing;ImGui::Checkbox("Enable preprocessing",&pp.enabled);
            if(pp.enabled)ImGui::TextWrapped("Corrected projections are saved beside the input and reused when correction inputs and settings match.");
            if(r.type=="iterative" && r.iterative.algorithm=="tigre_sart")
                ImGui::TextWrapped("SART updates one view at a time. One iteration is a complete sweep over all views; large volumes can take several minutes per sweep.");
            if(pp.enabled){auto& bh=pp.beam_hardening;ImGui::Checkbox("Beam hardening correction",&bh.enabled);
                if(bh.enabled){ImGui::Checkbox("Fit water plate",&bh.fit_from_water_plate);integer("Polynomial order",bh.order);
                    number("Clamp min",bh.clamp_min);number("Clamp max",bh.clamp_max);
                    ImGui::TextWrapped("Clamp bounds apply to corrected line integrals, not geometric length.");
                    number("Reference (keV)",bh.reference_energy_keV);
                    if(bh.fit_from_water_plate){number("Water max (mm)",bh.water_plate_max_mm);integer("Water samples",bh.water_plate_samples);}
                    else{
                        if(bh.order>=1&&bh.order<=16)bh.coefficients.resize(size_t(bh.order)+1,0);
                        for(size_t i=0;i<bh.coefficients.size();++i){const auto label="Coefficient c"+std::to_string(i);number(label.c_str(),bh.coefficients[i]);}
                        pathInput("BHC spectrum",bh.spectrum_file,L"Spectrum\0*.spec;*.csv;*.txt\0All files\0*.*\0\0");
                    }}
                if(ImGui::Checkbox("Scatter correction",&pp.scatter.enabled) && pp.scatter.enabled){
                    if(pp.scatter.air_projection_file.empty())pp.scatter.air_projection_file=draft.projection.air_output_file;
                    if(pp.scatter.model_file.empty())pp.scatter.model_file=draft.projection.scatter.model_file;
                }
                if(pp.scatter.enabled){
                    auto& sc=pp.scatter;
                    pathInput("Correction air",sc.air_projection_file,L"Air projection\0*.raw\0All files\0*.*\0\0");
                    pathInput("Correction model",sc.model_file,L"Scatter model\0*.json\0All files\0*.*\0\0");
                    if(ImGui::Button("Use forward scatter inputs")){sc.air_projection_file=draft.projection.air_output_file;sc.model_file=draft.projection.scatter.model_file;}
                    int solver=sc.solver.type=="adaptive_experimental"?1:0;
                    if(ImGui::Combo("Solver",&solver,"Fixed\0Adaptive (experimental)\0"))sc.solver.type=solver?"adaptive_experimental":"fixed";
                    if(!solver){integer("Correction iterations",sc.correction.iterations);number("Damping",sc.correction.damping);}
                    else{
                        auto& a=sc.solver.adaptive;
                        integer("Adaptive iterations",a.iterations);number("Initial damping",a.initial_damping);number("Minimum damping",a.min_damping);number("Damping decay",a.damping_decay);
                        if(ImGui::CollapsingHeader("Transmission guard")){
                            number("Low quantile",a.transmission_guard.low_quantile);number("Risk threshold",a.transmission_guard.risk_threshold);number("Positivity margin",a.transmission_guard.positivity_margin);}
#ifndef YK_SPECTRAL_ADAPTIVE_SCATTER
                        ImGui::TextWrapped("This build requires MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER=ON to run adaptive correction.");
#endif
                    }
                    number("Scatter scale",sc.correction.scatter_scale);number("Mask threshold",sc.correction.object_mask_threshold);
                    ImGui::Checkbox("Scatter guidance",&sc.guide.enabled);
                    if(sc.guide.enabled){number("Guide sigma U (px)",sc.guide.sigma_u_pixels);number("Guide sigma V (px)",sc.guide.sigma_v_pixels);}
                    int kernel=sc.kernel.mode=="segmented"?1:0;
                    if(ImGui::Combo("Kernel evaluation",&kernel,"All bins\0Segmented\0"))sc.kernel.mode=kernel?"segmented":"all";
                    if(kernel)integer("Kernel segments",sc.kernel.segments);
                    ImGui::TextWrapped("Correction uses processed air and energy-compatible calibration. Paths are checked before reconstruction.");
                }
            }
            if(ImGui::Button("Run reconstruction",{-1,32}))reconstruct=true;
            ImGui::TextWrapped("For the current computed projection, reconstruction uses its saved scan geometry. Output grid and reconstruction settings use this page.");
        }
        void results(yk::visualization::LiveView& view){
            ImGui::TextWrapped("Use the permanent Projections / Reconstruction tabs above the right image area.");
            if(view.hasReconstruction()){
                if(ImGui::SliderInt("Axial Z",&recon_slices[2],0,recon_dimensions[2]-1))slices_changed=true;
                if(ImGui::SliderInt("Coronal Y",&recon_slices[1],0,recon_dimensions[1]-1))slices_changed=true;
                if(ImGui::SliderInt("Sagittal X",&recon_slices[0],0,recon_dimensions[0]-1))slices_changed=true;
                double width=window_high-window_low,level=(window_high+window_low)/2;
                ImGui::SetNextItemWidth(-185);
                const bool width_changed=ImGui::InputDouble("Window width (HU)",&width,10,100,"%.1f");
                ImGui::SetNextItemWidth(-185);
                const bool level_changed=ImGui::InputDouble("Window level (HU)",&level,10,100,"%.1f");
                if(width_changed||level_changed){window_low=level-width/2;window_high=level+width/2;window_changed=true;}
                ImGui::Text("Water reference: %.3f keV",ct_reference.energy_keV);
                ImGui::Text("mu water: %.7f mm^-1",ct_reference.mu_water_mm_inv);
                if(ImGui::Button("Auto window"))auto_window=true;
                ImGui::TextWrapped("CT display uses HU = 1000 * (mu / mu_water - 1). Reference energy matches BHC. Original attenuation values are retained; windowing only affects display.");
            }else ImGui::TextWrapped("Reconstruct first, or load the saved output volume below.");
            if(ImGui::Button("Load saved reconstruction"))load_reconstruction=true;
        }
        void draw(int width, int height, bool enabled, const std::string& status, yk::visualization::LiveView& view)
        {
            ImGui::SetNextWindowPos({ 0, 0 });
            ImGui::SetNextWindowSize({ float(width) * .30f, float(height) });
            ImGui::Begin("Projection setup", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::BeginDisabled(!enabled);
            if (ImGui::Button("Load TOML"))
                load = true;
            ImGui::SameLine();
            if (ImGui::Button("Save TOML"))
                save = true;
            ImGui::TextWrapped("Changes are drafts until Calculate is pressed.");
            ImGui::SetNextItemWidth(-130);
            if(ImGui::SliderFloat("Volume opacity",&opacity_scale,0,1,"%.3f",ImGuiSliderFlags_Logarithmic))opacity_changed=true;
            relative_mode=view.relativeGeometryMode();
            ImGui::BeginDisabled(!view.relativeGeometryAvailable());
            if(ImGui::Checkbox("Gantry fixed (relative)",&relative_mode))relative_changed=true;
            ImGui::EndDisabled();
            if(!view.relativeGeometryAvailable())ImGui::TextWrapped("Relative mode requires a rigid source-detector trajectory.");
            auto body=[&](auto draw_content){
                ImGui::BeginChild("Parameter content",{0,-140},false);
                if(!message.empty()){
                    ImGui::PushStyleColor(ImGuiCol_Text,ImVec4(1,.8f,.4f,1));ImGui::TextWrapped("%s",message.c_str());ImGui::PopStyleColor();ImGui::Separator();
                }
                draw_content();ImGui::EndChild();
            };
            if (ImGui::BeginTabBar("Settings")) {
                if (ImGui::BeginTabItem("Scan")) {
                    body([&]{scan();});
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Volume")) {
                    body([&]{volume();});
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Spectrum")) {
                    body([&]{energy();});
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Effects")) {
                    body([&]{effects();});
                    ImGui::EndTabItem();
                }
                const auto recon_flags=!view.hasReconstruction() && (std::getenv("YK_IMGUI_RECON_SMOKE")||std::getenv("YK_IMGUI_CHAIN_SMOKE"))?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None;
                if(ImGui::BeginTabItem("Recon",nullptr,recon_flags)){body([&]{reconstruction();});ImGui::EndTabItem();}
                if(ImGui::BeginTabItem("Results",nullptr,select_results?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None)){body([&]{results(view);});ImGui::EndTabItem();}
                select_results=false;
                ImGui::EndTabBar();
            }
            const float action_width=(ImGui::GetContentRegionAvail().x-ImGui::GetStyle().ItemSpacing.x)/2;
            if (ImGui::Button("Calculate all views", { action_width, 38 })) calculate = true;
            ImGui::SameLine();
            if(ImGui::Button("Run reconstruction", { action_width, 38 })) reconstruct=true;
            ImGui::EndDisabled();
            if (!enabled) {
            float progress=0;
            if (status.rfind("Computing ",0)==0) {
                std::istringstream in(status.substr(10));
                int completed=0,total=0;char slash=0;
                if(in>>completed>>slash>>total && total>0) progress=float(completed)/total;
            }
            if(reconstruction_running && progress_total>0)progress=float(progress_completed)/progress_total;
            ImGui::ProgressBar(progress,{-1,22},reconstruction_running && progress_total==0?"Working...":nullptr);
        }
        ImGui::TextWrapped("%s", status.c_str());

            ImGui::End();
            if(drawer_open){
            ImGui::SetNextWindowPos({float(width)*.65f,0});
            ImGui::SetNextWindowSize({float(width)*.35f,56});
            ImGui::Begin("Image pages",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoScrollbar);
            if(ImGui::Button("Close")){drawer_open=false;drawer_changed=true;}
            ImGui::SameLine();
            if(ImGui::BeginTabBar("Image tabs")){
                const bool projections=ImGui::BeginTabItem("Projections",nullptr,select_image_page && result_mode==0?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None);
                if(ImGui::IsItemClicked() && result_mode!=0){result_mode=0;result_changed=true;}
                if(projections)ImGui::EndTabItem();
                const bool reconstruction=ImGui::BeginTabItem("Reconstruction",nullptr,select_image_page && result_mode==1?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None);
                if(ImGui::IsItemClicked() && result_mode!=1){result_mode=1;result_changed=true;}
                if(reconstruction)ImGui::EndTabItem();
                ImGui::EndTabBar();
            }
            select_image_page=false;
            ImGui::End();
            if(result_mode==1 && view.hasReconstruction()){
                ImGui::SetNextWindowPos({float(width)*.65f,float(height)-166});ImGui::SetNextWindowSize({float(width)*.35f,166});
                ImGui::Begin("Reconstruction drawer controls",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoScrollbar);
                ImGui::BeginDisabled(!enabled);
                const float cell=(ImGui::GetContentRegionAvail().x-2*ImGui::GetStyle().ItemSpacing.x)/3;
                ImGui::SetNextItemWidth(cell-24);if(ImGui::SliderInt("X",&recon_slices[0],0,recon_dimensions[0]-1))slices_changed=true;
                ImGui::SameLine();ImGui::SetNextItemWidth(cell-24);if(ImGui::SliderInt("Y",&recon_slices[1],0,recon_dimensions[1]-1))slices_changed=true;
                ImGui::SameLine();ImGui::SetNextItemWidth(cell-24);if(ImGui::SliderInt("Z",&recon_slices[2],0,recon_dimensions[2]-1))slices_changed=true;
                double ww=window_high-window_low,wl=(window_high+window_low)/2;
                const float half=(ImGui::GetContentRegionAvail().x-ImGui::GetStyle().ItemSpacing.x)/2;
                ImGui::SetNextItemWidth(half-42);const bool cw=ImGui::InputDouble("WW",&ww,0,0,"%.1f");
                ImGui::SameLine();ImGui::SetNextItemWidth(half-42);const bool cl=ImGui::InputDouble("WL",&wl,0,0,"%.1f");
                if(cw||cl){window_low=wl-ww/2;window_high=wl+ww/2;window_changed=true;}
                if(ImGui::Button("Auto window"))auto_window=true;ImGui::SameLine();ImGui::Text("%.1f keV | HU",ct_reference.energy_keV);
                ImGui::EndDisabled();ImGui::End();
            }
            }else{
                ImGui::SetNextWindowPos({float(width)-310,12});ImGui::SetNextWindowSize({298,98});
                ImGui::Begin("Image drawer handle",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoScrollbar);
                if(ImGui::Button("Projection images",{-1,32})){drawer_open=true;drawer_changed=true;result_mode=0;result_changed=true;select_image_page=true;}
                if(ImGui::Button("Reconstruction images",{-1,32})){drawer_open=true;drawer_changed=true;result_mode=1;result_changed=true;select_image_page=true;}
                ImGui::End();
            }
            ImGui::SetNextWindowPos({ float(width) * .30f, float(height) * .90f });
            ImGui::SetNextWindowSize({ float(width) * .70f, float(height) * .10f });
            ImGui::Begin("Scan browsing", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::BeginDisabled(!enabled || view.cachedFrameCount() == 0);
            int frame = view.displayedFrame();
            ImGui::SetNextItemWidth(-110);
            if (ImGui::SliderInt("View", &frame, 0, std::max(0, int(view.cachedFrameCount()) - 1)))
                requested_frame = frame;
            ImGui::EndDisabled();
            ImGui::Text("Drag 3D view to rotate. B: beam, T: trajectory.");
            ImGui::End();
        }
    };
}
bool runImGuiProjection(const SimulationConfig& initial, const std::filesystem::path& output, std::string& error)
{
    try {
        UiState ui;
        ui.draft = initial;
        ui.auto_reconstruct=initial.runsReconstruction();
        ui.opacity_scale=float(initial.visualization.volume_opacity_scale);
        ui.draft.projection.output_file = output;

        auto current = ui.draft;
        std::vector<std::uint8_t> labels;
        auto& options = initial.visualization;
        yk::visualization::LiveView view({ options.texture_max_size, options.max_fps, false, false, options.snapshot_directory.string() });
        if (!view.installGui([&](int w, int h, bool enabled, const std::string& status) { ui.draw(w, h, enabled, status, view); }))
            throw std::runtime_error("Build with BUILD_VTK_VISUALIZATION=ON and BUILD_IMGUI_GUI=ON");
        SpectralProjectionPipeline pipeline(primary_detail::makeCudaProjectionSourceRegistry().create("projection.source.multi_spec_fp"));
        pipeline.setVisualization(&view);
        XcomAttenuationProvider attenuation({});
        std::optional<SimulationConfig> computed_config, reconstructed_config;
        auto displayReconstruction=[&](const SimulationConfig& config){
            auto volume=loadReconstructionDisplay(config.reconstruction.output_volume_file,config.geometry.parameters);
            const auto reference=resolveWaterReference(config);
            ui.ct_reference=reference;
            ui.recon_dimensions=volume.dimensions;
            for(int i=0;i<3;++i)ui.recon_slices[i]=volume.dimensions[i]/2;
            view.setReconstruction(std::move(volume));
            view.setReconstructionCtReference(reference.mu_water_mm_inv);
            const auto window=view.reconstructionWindow();ui.window_low=window[0];ui.window_high=window[1];
            view.setImageDrawer(true);ui.drawer_open=true;ui.drawer_changed=false;ui.result_mode=1;ui.select_image_page=true;ui.result_changed=false;ui.select_results=true;reconstructed_config=config;
        };
        auto runReconstruction=[&](const SimulationConfig& projection_config) {
            auto config=projection_config;
            config.reconstruction=ui.draft.reconstruction;
            const auto& grid=ui.draft.geometry.parameters;auto& g=config.geometry.parameters;
            g.reconstruction_volume_x=grid.reconstruction_volume_x;g.reconstruction_volume_y=grid.reconstruction_volume_y;g.reconstruction_volume_z=grid.reconstruction_volume_z;
            g.reconstruction_voxel_x_mm=grid.reconstruction_voxel_x_mm;g.reconstruction_voxel_y_mm=grid.reconstruction_voxel_y_mm;g.reconstruction_voxel_z_mm=grid.reconstruction_voxel_z_mm;
            g.reconstruction_offset_x_mm=grid.reconstruction_offset_x_mm;g.reconstruction_offset_y_mm=grid.reconstruction_offset_y_mm;g.reconstruction_offset_z_mm=grid.reconstruction_offset_z_mm;
            config.workflow.mode=WorkflowMode::Reconstruct;
            config.projection.scatter.enabled=false; // Forward scatter is not executed by reconstruction.
            if(config.reconstruction.output_volume_file.empty())throw std::runtime_error("Set a reconstruction output volume path");
            const auto validation=std::filesystem::path(output.string()+".recon-validation.toml");
            saveProjectionUiConfig(config,validation,WorkflowMode::Reconstruct);
            auto checked=loadConfig(validation);checked.source_config_file=projection_config.source_config_file;
            for(const auto& path:{checked.reconstruction.output_volume_file,checked.reconstruction.slice_prefix})
                if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
            view.setEditingEnabled(false);view.setStatus("Running reconstruction...");
            std::mutex progress_mutex;TaskProgress progress_state;
            std::vector<TaskProgress> progress_history;
            ui.reconstruction_running=true;ui.progress_completed=ui.progress_total=0;
            auto task=std::async(std::launch::async,[checked,&progress_mutex,&progress_state,&progress_history]{std::string diagnostic;
                std::string logged_phase;
                const auto callback=[&](TaskProgress value){
                    {std::lock_guard<std::mutex> lock(progress_mutex);progress_state=value;progress_history.push_back(value);}
                    // Use the existing completion callback; logging must not add GPU synchronization.
                    if(value.phase!=logged_phase || value.phase=="Iterative updates"){
                        std::ostringstream line;line<<"[reconstruction] "<<value.phase;
                        if(value.total>0)line<<" "<<value.completed<<"/"<<value.total;
                        SimLogger::instance().info(line.str());logged_phase=value.phase;
                    }
                };
                const bool ok=reconstructWithLibraryFdk(checked,checked.reconstruction.input_projection_file,checked.reconstruction.output_volume_file,checked.reconstruction.slice_prefix,diagnostic,callback);
                return std::make_pair(ok,diagnostic);});
            const auto started=std::chrono::steady_clock::now();
            while(task.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready){
                TaskProgress value;{std::lock_guard<std::mutex> lock(progress_mutex);value=progress_state;}
                ui.progress_completed=value.completed;ui.progress_total=value.total;
                std::ostringstream text;text<<value.phase;
                if(value.total>0)text<<" "<<value.completed<<" / "<<value.total<<" ("<<int(100.0*value.completed/value.total)<<"%)";
                text<<" | "<<std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-started).count()<<" s";
                view.setStatus(text.str());view.pollGui();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            const auto result=task.get();ui.reconstruction_running=false;view.setEditingEnabled(true);
            if(!result.first)throw std::runtime_error(result.second);
            if(std::getenv("YK_IMGUI_RECON_SMOKE")||std::getenv("YK_IMGUI_CHAIN_SMOKE")){
                bool reported=false;for(const auto& event:progress_history){
                    std::cout<<"progress "<<event.phase<<" "<<event.completed<<"/"<<event.total<<std::endl;
                    if(event.completed>0 && (std::string(event.phase)=="FDK views"||std::string(event.phase)=="Iterative updates"||std::string(event.phase)=="Full-volume reconstruction"))reported=true;
                }
                if(!reported)throw std::runtime_error("reconstruction did not report completed units");
            }

            displayReconstruction(checked);
            ui.message="Reconstruction displayed and saved: "+checked.reconstruction.output_volume_file.string();
            view.setStatus("Reconstruction complete. Projection browsing is available.");
            saveProjectionUiConfig(checked,checked.reconstruction.output_volume_file.string()+".toml",WorkflowMode::Reconstruct);
        };
        auto loadResources=[&](const SimulationConfig& config) {
            view.setEditingEnabled(false);
            view.setStatus("Loading TOML volume, materials, spectrum and trajectory...");
            auto loaded_labels=loadLabelVolume(config.projection.label_volume);
            validateProjectionUiInputs(config,loaded_labels);
            auto projection=config.projection;projection.spectrum.export_file.clear();
            auto spectrum=loadProjectionSpectrum(projection);
            // Resolve material identities and attenuation now; no projection kernel is launched.
            SpectralTransmissionModel material_check(projection.materials,spectrum,attenuation,projection.detector_response);
            const auto frames=buildDisplayGeometry(config);
            yk::visualization::LabelVolume volume;
            const auto& g=config.geometry.parameters;
            volume.dimensions={g.volume_x,g.volume_y,g.volume_z};volume.spacing={g.voxel_x_mm,g.voxel_y_mm,g.voxel_z_mm};
            volume.labels=loaded_labels;
            constexpr double colors[][3]={{.75,.85,1},{1,.65,.25},{.85,.35,.4},{.4,.85,.6},{.8,.6,.95},{.85,.85,.35}};
            for(size_t i=0;i<projection.materials.size();++i){const auto& m=projection.materials[i];const auto* color=colors[i%6];
                volume.materials.push_back({m.label,m.name,{color[0],color[1],color[2],m.density_g_cm3<.01?0.:.15}});}
            view.setVolumeOpacityScale(config.visualization.volume_opacity_scale);
            view.setVolume(volume);view.setTrajectory(frames);view.setGeometryPreview(frames.front());
            ui.result_mode=0;ui.select_image_page=true;
            ui.histogram(loaded_labels);ui.spectrum(spectrum);labels=std::move(loaded_labels);
            ui.requested_frame=-1;
            view.setEditingEnabled(true);
            if(config.projection.scatter.enabled && !std::filesystem::is_regular_file(config.projection.scatter.model_file)) {
                ui.message="Missing scatter model: "+config.projection.scatter.model_file.string()+". Set the correct path in Effects, or explicitly disable Scatter before calculating.";
                view.setStatus("Resources loaded. Scatter model needs attention.");
                std::cerr<<ui.message<<std::endl;
            } else view.setStatus("Resources loaded. Press Calculate to generate projections.");
            if(config.visualization.auto_load_reconstruction && std::filesystem::is_regular_file(config.reconstruction.output_volume_file)) {
                try{displayReconstruction(config);ui.drawer_open=false;ui.drawer_changed=false;view.setImageDrawer(false);view.setStatus("Saved reconstruction loaded (HU). Open the image drawer to inspect.");}
                catch(const std::exception& e){ui.message=std::string("Saved reconstruction could not be loaded: ")+e.what();}
            }
        };
        bool success = true; // Closing a loaded setup without calculating is normal.
        const bool smoke = std::getenv("YK_IMGUI_SMOKE") != nullptr;
        const bool chain_smoke=std::getenv("YK_IMGUI_CHAIN_SMOKE")!=nullptr;
        ui.calculate = smoke || chain_smoke;
        try {loadResources(ui.draft);}
        catch(const std::exception& e){ui.message=e.what();view.setEditingEnabled(true);view.setStatus("Could not load TOML resources. Correct paths or load another TOML.");if(smoke||chain_smoke)throw;}
        if(ui.message.empty())ui.message=initial.source_config_file.string();
        if(std::getenv("YK_IMGUI_RELATIVE_SMOKE")) {
            if(!view.relativeGeometryAvailable())throw std::runtime_error("Rigid gantry relative mode unavailable");
            view.setRelativeGeometryMode(true); // Loaded preview, with no cached projections.
            view.setRelativeGeometryMode(false);
            const auto frames=buildDisplayGeometry(ui.draft);
            const std::vector<float> pixels(16,1.f);
            view.update(frames.front(),4,4,pixels,"Relative geometry smoke",true);
            view.update(frames.back(),4,4,pixels,"Relative geometry smoke",true);
            view.setRelativeGeometryMode(true);
            if(!view.selectFrame(frames.front().view) || !view.selectFrame(frames.back().view))
                throw std::runtime_error("Relative cached frame navigation failed");
            view.setRelativeGeometryMode(false);
            view.setRelativeGeometryMode(true);
            view.saveSnapshot();return true;
        }
        if(std::getenv("YK_IMGUI_LOAD_SMOKE")) {
            auto trial=initial;
            const GeometryKind kinds[]={GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::PlanarSourceOrbit,GeometryKind::NikonTilted};
            for(auto kind:kinds){
                trial.geometry.kind=kind;auto& g=trial.geometry.parameters;
                g.pitch_mm_per_turn=100;g.planar_detector_y_mm=g.sdd_mm-g.sid_mm;
                g.source_orbit_radius_mm=g.sid_mm;g.reconstruction_detector_u=g.detector_u;g.reconstruction_detector_v=g.detector_v;g.nikon_tilt_rad=15/degrees;
                loadResources(trial);
                if(view.cachedFrameCount()!=0 || labels.empty() || ui.spectrum_plot.empty())
                    throw std::runtime_error("TOML resource load did not complete independently of projection");
            }
            view.saveSnapshot();
            return true;
        }
        if(const auto* probe=std::getenv("YK_IMGUI_RECON_SMOKE")){
            if(std::string(probe)=="sirt"){
                ui.draft.reconstruction.type="iterative";ui.draft.reconstruction.iterative.algorithm="sirt";
                ui.draft.reconstruction.iterative.iterations=2;ui.draft.reconstruction.iterative.subsets=1;
                ui.draft.reconstruction.output_volume_file=ui.draft.reconstruction.output_volume_file.parent_path()/"reconstruction-sirt.raw";
                ui.draft.reconstruction.save_slices=false;
            }
            runReconstruction(ui.draft);view.saveSnapshot();return true;
        }
        int smoke_mode = 0;
        while (view.isOpen()) {
            view.pollGui();
            try {
                if(ui.relative_changed){ui.relative_changed=false;view.setRelativeGeometryMode(ui.relative_mode);}
                if(ui.drawer_changed){ui.drawer_changed=false;view.setImageDrawer(ui.drawer_open);}
                if(ui.result_changed){ui.result_changed=false;view.showReconstruction(ui.result_mode==1);}
                if(ui.slices_changed){ui.slices_changed=false;view.setReconstructionSlices(ui.recon_slices);}
                if(ui.window_changed){ui.window_changed=false;view.setReconstructionWindow(ui.window_low,ui.window_high);}
                if(ui.auto_window){ui.auto_window=false;const auto indices=ui.recon_slices;
                    displayReconstruction(reconstructed_config?*reconstructed_config:ui.draft);ui.recon_slices=indices;view.setReconstructionSlices(indices);}
                if(ui.load_reconstruction){ui.load_reconstruction=false;displayReconstruction(ui.draft);view.setStatus("Saved reconstruction loaded and displayed.");}
                if(ui.opacity_changed){ui.opacity_changed=false;ui.draft.visualization.volume_opacity_scale=ui.opacity_scale;view.setVolumeOpacityScale(ui.opacity_scale);}
                if(ui.reconstruct){ui.reconstruct=false;
                    auto config=ui.draft;
                    if(computed_config && config.reconstruction.input_projection_file.lexically_normal()==computed_config->projection.output_file.lexically_normal())config.geometry=computed_config->geometry;
                    runReconstruction(config);
                }
                if (ui.load) {
                    ui.load = false;
                    const auto file = chooseFile(false, L"Configuration\0*.toml\0All files\0*.*\0\0");
                    if (!file.empty()) {
                        auto loaded = loadConfig(file, false);
                        if (loaded.geometry.kind == GeometryKind::CylCbct || loaded.geometry.kind == GeometryKind::CylHelical)
                            throw std::runtime_error("Choose one of the four flat-detector scans");
                        ui.auto_reconstruct=loaded.runsReconstruction();
                        loaded.workflow.mode = WorkflowMode::Project;
                        loaded.visualization.enabled = true;
                        loaded.visualization.interactive = true;
                        loaded.visualization.offscreen = false;
                        loaded.visualization.hold_open = true;
                        ui.draft = std::move(loaded);
                        ui.opacity_scale=float(ui.draft.visualization.volume_opacity_scale);
                        computed_config.reset();
                        ui.message.clear();
                        try {loadResources(ui.draft);}
                        catch(const std::exception& e){
                            view.setEditingEnabled(true);
                            view.setStatus("TOML parameters loaded; resource loading failed.");
                            ui.message=e.what();
                            std::cerr<<"TOML resource load failed: "<<file<<": "<<e.what()<<std::endl;
                        }
                        if(ui.message.empty())ui.message = "Loaded: " + file.string();
                    }
                }
                if (ui.save) {
                    ui.save = false;
                    const auto file = chooseFile(true, L"Configuration\0*.toml\0All files\0*.*\0\0");
                    if (!file.empty()) {
                        saveProjectionUiConfig(ui.draft, file,ui.auto_reconstruct?WorkflowMode::ProjectAndReconstruct:WorkflowMode::Project);
                        ui.message = "Saved configuration: " + file.string();
                    }
                }
                if (ui.preview || ui.export_spectrum) {
                    const bool exporting = ui.export_spectrum;
                    ui.preview = false;
                    ui.export_spectrum = false;
                    view.setEditingEnabled(false);
                    view.setStatus("Generating / loading spectrum...");
                    auto projection = ui.draft.projection;
                    projection.spectrum.export_file.clear();
                    const auto spectrum = loadProjectionSpectrum(projection);
                    ui.spectrum(spectrum);
                    if (exporting) {
                        const auto file = chooseFile(true, L"Spectrum\0*.spec\0All files\0*.*\0\0");
                        if (!file.empty())
                            saveSpectrum(spectrum, file);
                    }
                    ui.message = "Spectrum preview updated. Projection results still use the last calculated settings.";
                    view.setEditingEnabled(true);
                    view.setStatus("Ready");
                }
                if (ui.requested_frame >= 0) {
                    const int frame = ui.requested_frame;
                    ui.requested_frame = -1;
                    view.selectFrame(frame);
                }
                if (ui.calculate) {
                    ui.calculate = false;
                    view.setEditingEnabled(false);
                    view.setStatus("Validating inputs and preparing spectrum...");
                    ui.message.clear();
                    auto trial = ui.draft;
                    trial.workflow.mode = WorkflowMode::Project;
                    trial.visualization.enabled = true;
                    trial.visualization.interactive = true;
                    trial.visualization.offscreen = false;
                    trial.visualization.hold_open = true;
                    const auto validation_file = std::filesystem::absolute(std::filesystem::path(output.string() + ".imgui-validation.toml"));
                    saveProjectionUiConfig(trial, validation_file);
                    auto checked = loadConfig(validation_file);
                    checked.source_config_file = trial.source_config_file;
                    auto trial_labels = loadLabelVolume(checked.projection.label_volume);
                    validateProjectionUiInputs(checked, trial_labels);
                    auto spectrum = loadProjectionSpectrum(checked.projection);
                    if (spectrum.size() > 256)
                        throw std::runtime_error("CUDA preview supports at most 256 energy bins; increase bin width");
                    if(ui.auto_reconstruct && !smoke){
                        auto chain=checked;chain.reconstruction=ui.draft.reconstruction;
                        chain.reconstruction.input_projection_file=checked.projection.output_file;
                        const auto chain_validation=std::filesystem::path(output.string()+".chain-validation.toml");
                        saveProjectionUiConfig(chain,chain_validation,WorkflowMode::ProjectAndReconstruct);
                        loadConfig(chain_validation);
                    }
                    SpectralTransmissionModel model(checked.projection.materials, spectrum, attenuation, checked.projection.detector_response);
                    ui.spectrum(spectrum);
                    ui.histogram(trial_labels);
                    current = std::move(checked);
                    labels = std::move(trial_labels);
                    success = pipeline.run(current, labels, model, current.projection.output_file, error);
                    if (!view.isOpen())
                        return success;
                    view.setEditingEnabled(true);
                    view.setStatus(success ? "Ready: all views cached." : "Calculation failed: " + error);
                    if (success) {
                        computed_config=current;
                        ui.draft.reconstruction.input_projection_file=current.projection.output_file;
                        const auto saved = std::filesystem::path(current.projection.output_file.string() + ".imgui.toml");
                        saveProjectionUiConfig(current, saved,ui.auto_reconstruct && !smoke?WorkflowMode::ProjectAndReconstruct:WorkflowMode::Project);
                        if(ui.auto_reconstruct && !smoke){
                            auto reconstruction=current;reconstruction.reconstruction.input_projection_file=current.projection.output_file;
                            ui.draft.reconstruction.input_projection_file=current.projection.output_file;
                            runReconstruction(reconstruction);
                        }
                    }
                    if(chain_smoke){view.saveSnapshot();return success;}
                    if (smoke) {
                        view.saveSnapshot();
                        if (!success)
                            return false;
                        if (++smoke_mode == 4)
                            return true;
                        ui.draft.geometry.kind = smoke_mode == 1 ? GeometryKind::FlatHelical : smoke_mode == 2 ? GeometryKind::PlanarSourceOrbit
                                                                                                               : GeometryKind::NikonTilted;
                        auto& g = ui.draft.geometry.parameters;
                        g.pitch_mm_per_turn = 100;
                        g.planar_detector_y_mm = g.sdd_mm - g.sid_mm;
                        g.source_orbit_radius_mm = g.sid_mm;
                        g.reconstruction_detector_u = g.detector_u;
                        g.reconstruction_detector_v = g.detector_v;
                        g.nikon_tilt_rad = 15 / degrees;
                        ui.calculate = true;
                    }
                }
            } catch (const std::exception& e) {
                ui.message = e.what();
                view.setEditingEnabled(true);
                view.setStatus("Operation failed. See the message above.");
                if (smoke) {
                    error = e.what();
                    return false;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return success;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
bool launchImGuiProjection(const std::filesystem::path& config_file,std::string& error) {
    try {
        const auto file=config_file.empty()?chooseFile(false,L"Configuration\0*.toml\0All files\0*.*\0\0"):config_file;
        if(file.empty()) return true;
        auto config=loadConfig(file,false);
        if(config.geometry.kind==GeometryKind::CylCbct||config.geometry.kind==GeometryKind::CylHelical)
            throw std::runtime_error("Choose one of the four flat-detector scan types");
        config.visualization.enabled=true;
        config.visualization.interactive=true;
        config.visualization.offscreen=false;
        config.visualization.hold_open=true;
        return runImGuiProjection(config,config.projection.output_file,error);
    } catch(const std::exception& e) {error=e.what();return false;}
}

}
