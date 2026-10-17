#include "ui/VolumePanel.hpp"
#include "ui/EffectsPanel.hpp"
#include "ui/SpectrumPanel.hpp"
#include "ui/ScanPanel.hpp"
#include "ui/ReconstructionPanel.hpp"
#include "ui/ReconstructionViewerPanel.hpp"
#include "config/ReconstructionGeometry.hpp"
#include "calibration/GeometryCalibrationStage.hpp"
#include "cbct_calibration/io.hpp"
#include "examples/DataExamples.hpp"
#include "io/MatLabelVolume.hpp"
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
    using namespace ui;
    struct UiState {
        SimulationConfig draft;
        std::vector<float> spectrum_plot;
        std::string message;
        bool calculate = false, load = false, save = false, preview = false, export_spectrum = false;
        bool calibration_window=false,calibrate_geometry=false,nominal_reconstruction=false,corrected_reconstruction=false;
        calibration::Context calibration_context;
        calibration::Data calibration_data;
        std::optional<SimulationConfig> calibration_input;
        std::string calibration_message, known_example_id;
        double nominal_sid=440,nominal_sdd=770;
        bool create_example=false; int example_index=0;
        bool example_window=false, help_window=false;
        int requested_settings=-1;
        std::vector<examples::Parameter> example_parameters;
        std::string example_error;
        void bindCalibrationExample(const std::string& id,const std::vector<examples::Parameter>& parameters){
            known_example_id=id;
            if(id.rfind("bead-",0)!=0)return;
            std::map<std::string,double> values;for(const auto& p:parameters)values[p.id]=p.value;
            auto& c=calibration_context.config;
            using Kind=cbct::calibration::PipelineConfig::PhantomKind;
            c.phantom_kind=id=="bead-line"?Kind::SingleRow:id=="bead-spiral"?Kind::Spiral:Kind::DoubleRing;
            if(c.phantom_kind==Kind::SingleRow){c.row_beads=int(values.at("count"));c.row_spacing_mm=values.at("spacing");}
            else if(c.phantom_kind==Kind::Spiral){
                c.spiral_beads=int(values.at("count"));c.spiral_radius_mm=values.at("radius");
                c.spiral_spacing_mm=values.at("spacing");c.spiral_step_deg=values.at("angle_step");
                c.spiral_marker_mm={values.at("radius"),0,values.at("marker_z")};
            }else{
                auto& f=c.phantom;
                f.beads_per_ring=int(values.at("count"));f.upper_radius_mm=values.at("radius");f.lower_radius_mm=values.at("lower_radius");
                f.ring_half_spacing_mm=values.at("separation")/2;f.lower_phase_deg=values.at("phase");
                f.markers_for_numbering_only=id=="bead-cylinder-marker";
                f.marker_points_mm={{values.at("radius"),0,values.at("marker_z")}};
                if(id=="bead-cylinder-marker")f.marker_points_mm.push_back({values.at("radius"),0,-values.at("marker_z")});
                else if(id=="bead-sync-6" || id=="bead-sync-12"){
                    double a=values.at("marker2_angle")/degrees;
                    f.marker_points_mm.emplace_back(values.at("radius")*std::cos(a),values.at("radius")*std::sin(a),values.at("marker2_z"));
                }
            }
            nominal_sid=draft.geometry.parameters.sid_mm;nominal_sdd=draft.geometry.parameters.sdd_mm;
        }
        void chooseExample(int index){
            example_index=index;
            example_parameters=examples::clientExamples().create(builtin_examples[index].id)->parameters();
            example_error.clear();example_window=!example_parameters.empty();
            create_example=example_parameters.empty();
        }
        const std::vector<pipeline::PluginInfo> builtin_examples=examples::clientExamples().list();
        bool reconstruction_geometry_initialized=false;
        bool reconstruct=false, auto_reconstruct=false, opacity_changed=false;
        bool scan_type_changed=false;
        float opacity_scale=.03f;
        int result_mode=0;
        bool relative_mode=false,relative_changed=false;
        bool drawer_open=false,drawer_changed=false;
        bool select_image_page=true;
        bool result_changed=false,load_reconstruction=false,browse_reconstruction=false;
        ui::ReconstructionViewerPanel viewer;
        bool select_results=false;
        WaterReference ct_reference;
        bool reconstruction_running=false;
        int progress_completed=0,progress_total=0;


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
        void scan(){scan_type_changed=ui::ScanPanel::draw(draft)||scan_type_changed;}
        ui::VolumePanel volume_panel;
        void volume(){volume_panel.draw(draft,counts);}
        void energy(){ui::SpectrumPanel::draw(draft,spectrum_plot,preview,export_spectrum);}
        void effects(){ui::EffectsPanel::draw(draft);}
        void reconstruction() {
            ui::ReconstructionPanel::draw(draft,auto_reconstruct,reconstruction_geometry_initialized);
        }
        void results(yk::visualization::LiveView& view){
            ImGui::TextWrapped("Open Projections or Reconstruction from the View menu.");
            if(view.hasReconstruction()){
                ImGui::Text("Grid: %d x %d x %d",viewer.recon_dimensions[0],viewer.recon_dimensions[1],viewer.recon_dimensions[2]);
                ImGui::Text("Voxel: %.3f / %.3f / %.3f mm",viewer.spacing[0],viewer.spacing[1],viewer.spacing[2]);
                if(ImGui::Button("Open MPR workspace")){drawer_open=true;drawer_changed=true;result_mode=1;result_changed=true;select_image_page=true;viewer.expanded=true;viewer.layout_changed=true;}
                ImGui::TextWrapped("Four linked views, oblique plane, HU window presets and mouse controls are available in the image viewer.");
                ImGui::TextWrapped("CT display uses HU = 1000 * (mu / mu_water - 1). Reference energy matches BHC. Original attenuation values are retained; windowing only affects display.");
            }else ImGui::TextWrapped("Reconstruct first, or load the saved output volume below.");
            if(ImGui::Button("Load saved reconstruction"))load_reconstruction=true;
        }
        void draw(int width, int height, bool enabled, const std::string& status, yk::visualization::LiveView& view)
        {
            float menu_height=ImGui::GetFrameHeight();
            if(ImGui::BeginMainMenuBar()){
                menu_height=ImGui::GetWindowHeight();
                if(ImGui::BeginMenu("File")){
                    if(ImGui::MenuItem("Open configuration...",nullptr,false,enabled))load=true;
                    if(ImGui::MenuItem("Save configuration...",nullptr,false,enabled))save=true;
                    ImGui::Separator();
                    if(ImGui::MenuItem("Volume and materials...",nullptr,false,enabled))requested_settings=1;
                    if(ImGui::MenuItem("Load reconstruction...",nullptr,false,enabled))browse_reconstruction=true;
                    if(ImGui::MenuItem("Export spectrum...",nullptr,false,enabled))export_spectrum=true;
                    ImGui::EndMenu();
                }
                if(ImGui::BeginMenu("Settings",enabled)){
                    const char* pages[]={"Scan geometry","Volume and materials","Spectrum","Physical effects","Reconstruction","Results"};
                    for(int i=0;i<6;++i)if(ImGui::MenuItem(pages[i]))requested_settings=i;
                    ImGui::EndMenu();
                }
                if(ImGui::BeginMenu("Compute",enabled)){
                    if(ImGui::MenuItem("Forward projection"))calculate=true;
                    if(ImGui::MenuItem("Reconstruction"))reconstruct=true;
                    if(ImGui::MenuItem("Preview spectrum"))preview=true;
                    ImGui::Separator();
                    ImGui::MenuItem("Reconstruct after projection",nullptr,&auto_reconstruct);
                    ImGui::EndMenu();
                }
                if(ImGui::BeginMenu("View",enabled)){
                    if(ImGui::MenuItem("Projection images",nullptr,drawer_open&&result_mode==0)){drawer_open=true;drawer_changed=true;result_mode=0;result_changed=true;select_image_page=true;}
                    if(ImGui::MenuItem("Reconstruction images",nullptr,drawer_open&&result_mode==1)){drawer_open=true;drawer_changed=true;result_mode=1;result_changed=true;select_image_page=true;}
                    if(ImGui::MenuItem("MPR workspace",nullptr,drawer_open&&result_mode==1&&viewer.expanded,view.hasReconstruction())){drawer_open=true;drawer_changed=true;result_mode=1;result_changed=true;select_image_page=true;viewer.expanded=true;viewer.layout_changed=true;}
                    if(ImGui::MenuItem("Close image drawer",nullptr,false,drawer_open)){drawer_open=false;drawer_changed=true;}
                    relative_mode=view.relativeGeometryMode();
                    if(ImGui::MenuItem("Gantry fixed",nullptr,&relative_mode,view.relativeGeometryAvailable()))relative_changed=true;
                    ImGui::EndMenu();
                }
                if(ImGui::BeginMenu("Example",enabled)){
                    if(ImGui::MenuItem("Phantom calibration (experimental)..."))calibration_window=true;
                    ImGui::Separator();
                    for(const auto* category:{"Water phantoms","Geometry calibration","ICRP145","BrainWeb"}){
                        if(ImGui::BeginMenu(category)){
                            for(int i=0;i<int(builtin_examples.size());++i){
                                const auto& item=builtin_examples[i];
                                const auto group=item.id.rfind("water-",0)==0?"Water phantoms":item.id.rfind("bead-",0)==0?"Geometry calibration":item.category.c_str();
                                if(std::string_view(category)==group && ImGui::MenuItem(item.name.c_str()))chooseExample(i);
                            }
                            ImGui::EndMenu();
                        }
                    }
                    ImGui::EndMenu();
                }
                if(ImGui::BeginMenu("Help")){
                    if(ImGui::MenuItem("Workflow guide"))help_window=true;
                    ImGui::EndMenu();
                }
                ImGui::EndMainMenuBar();
            }
            if(example_window){
                ImGui::SetNextWindowSize({620,650},ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowPos({float(width)*.5f,float(height)*.5f},ImGuiCond_Appearing,{.5f,.5f});
                if(ImGui::Begin("Phantom parameters",&example_window)){
                    ImGui::BeginDisabled(!enabled);
                    ImGui::TextWrapped("%s",builtin_examples[example_index].name.c_str());
                    ImGui::TextWrapped("Generates a new setup in memory. Current projections and reconstruction will be cleared.");
                    const auto preset=examples::clientExamples().create(builtin_examples[example_index].id)->scanPreset();
                    ImGui::TextWrapped("Scan preset: %s",preset.name.c_str());
                    ImGui::Text("SID %.0f / SDD %.0f mm | %d views (%d / turn)",preset.sid_mm,preset.sdd_mm,preset.views,preset.views_per_turn);
                    ImGui::Text("Detector %d x %d | %.3f x %.3f mm | %s + Hann",preset.detector_u,preset.detector_v,preset.pixel_u_mm,preset.pixel_v_mm,preset.reconstruction.c_str());
                    if(preset.pitch_mm_per_turn>0)ImGui::Text("Travel %.0f mm / turn | start Z %.0f mm",preset.pitch_mm_per_turn,preset.start_z_mm);
                    ImGui::TextWrapped("120 kVp, 2 mm Al. Noise and scatter off. Generate applies ALL setup parameters.");
                    ImGui::BeginChild("Phantom fields",{0,-110},true);
                    std::string group;
                    for(auto& parameter:example_parameters){
                        if(group!=parameter.group){group=parameter.group;ImGui::SeparatorText(group.c_str());}
                        ImGui::PushID(parameter.id.c_str());
                        ImGui::SetNextItemWidth(150);
                        if(parameter.integer){int value=int(parameter.value);if(ImGui::InputInt(parameter.name.c_str(),&value))parameter.value=value;}
                        else ImGui::InputDouble(parameter.name.c_str(),&parameter.value,0,0,"%.3f");
                        ImGui::PopID();
                    }
                    ImGui::EndChild();
                    if(!example_error.empty()){ImGui::TextWrapped("%s",example_error.c_str());}
                    if(ImGui::Button("Generate and load",{170,32})){create_example=true;example_error.clear();}
                    ImGui::SameLine();
                    if(ImGui::Button("Restore defaults"))chooseExample(example_index);
                    ImGui::SameLine();
                    if(ImGui::Button("Cancel"))example_window=false;
                    ImGui::EndDisabled();
                }
                ImGui::End();
            }
            if(calibration_window){
                ImGui::SetNextWindowSize({690,720},ImGuiCond_FirstUseEver);
                if(ImGui::Begin("Phantom calibration (experimental)",&calibration_window)){
                    ImGui::BeginDisabled(!enabled);
                    ImGui::TextWrapped("Set non-ideal detector/source offsets and tilts in Settings > Scan, then Compute > Forward projection. Calibration below uses that saved projection stack.");
                    ImGui::TextWrapped("Full circular flat CBCT. Uses measured negative-log projections and known phantom dimensions.");
                    auto& c=calibration_context.config;
                    using Kind=cbct::calibration::PipelineConfig::PhantomKind;
                    int kind=int(c.phantom_kind);
                    if(ImGui::Combo("Phantom family",&kind,"Double rings\0Marked spiral\0Single row (limited)\0")){
                        c.phantom_kind=Kind(kind);calibration_data={};calibration_input.reset();
                    }
                    if(c.phantom_kind==Kind::SingleRow){
                        integer("Bead count",c.row_beads);number("Spacing (mm)",c.row_spacing_mm);
                        ImGui::TextWrapped("Five-parameter fit: SID, SDD, detector U/V and in-plane rotation. Out-of-plane detector tilt must be negligible. The line must be off the rotation axis. Nominal distances initialize the fit.");
                    }else if(c.phantom_kind==Kind::Spiral){
                        integer("Bead count",c.spiral_beads);number("Radius (mm)",c.spiral_radius_mm);
                        number("Axial spacing (mm)",c.spiral_spacing_mm);number("Angular step (deg)",c.spiral_step_deg);
                        number("Marker X (mm)",c.spiral_marker_mm.x());number("Marker Y (mm)",c.spiral_marker_mm.y());number("Marker Z (mm)",c.spiral_marker_mm.z());
                    }else{
                    ImGui::SeparatorText("Known phantom dimensions");
                    ImGui::Checkbox("Use markers for numbering only",&c.phantom.markers_for_numbering_only);
                    ImGui::InputInt("Beads per ring",&c.phantom.beads_per_ring);
                    number("Upper radius (mm)",c.phantom.upper_radius_mm);number("Lower radius (mm)",c.phantom.lower_radius_mm);
                    number("Half ring separation (mm)",c.phantom.ring_half_spacing_mm);number("Lower phase (deg)",c.phantom.lower_phase_deg);
                    int markers=int(c.phantom.marker_points_mm.size());
                    if(ImGui::SliderInt("Markers",&markers,0,2))c.phantom.marker_points_mm.resize(markers,cbct::calibration::Point3(50,0,70));
                    for(size_t i=0;i<c.phantom.marker_points_mm.size();++i){
                        ImGui::PushID(int(i));ImGui::Text("Marker %d",int(i)+1);
                        number("X (mm)",c.phantom.marker_points_mm[i].x());number("Y (mm)",c.phantom.marker_points_mm[i].y());number("Z (mm)",c.phantom.marker_points_mm[i].z());
                        ImGui::PopID();
                    }
                    }
                    ImGui::SeparatorText("Detection and fit");
                    ImGui::InputFloat("Threshold (-log)",&c.tracker.threshold,0,0,"%.2f");
                    ImGui::InputInt("Minimum component pixels",&c.tracker.min_pixels);
                    number("Tracking gate (pixels)",c.tracker.gate_radius_px);
                    ImGui::InputInt("Maximum fit iterations",&c.joint_max_iterations);
                    number("Accepted RMSE (pixels)",calibration_context.maximum_rmse_px);
                    number("Nominal SID (mm)",nominal_sid);number("Nominal SDD (mm)",nominal_sdd);
                    if(ImGui::Button("Calibrate current projections"))calibrate_geometry=true;
                    if(!calibration_message.empty())ImGui::TextWrapped("%s",calibration_message.c_str());
                    if(calibration_data.result){
                        const auto& r=*calibration_data.result;
                        ImGui::TextWrapped("Model: %s",r.model.c_str());
                        ImGui::Text("Last fit: RMSE %.4f px | max %.4f px | %s",r.joint_rmse_px,r.max_point_error_px,r.joint_converged?"converged":"NOT converged");
                        ImGui::Text("SID %.3f / SDD %.3f | U %.3f / V %.3f mm",r.machine.sid_mm,r.machine.sdd_mm,r.machine.offset_u_mm,r.machine.offset_v_mm);
                        ImGui::TextWrapped("Correction uses fitted geometry, not pixel warping. Source and normal offsets are gauge-fixed. Corrected volume is in the fitted scanner frame.");
                    }
                    ImGui::BeginDisabled(!calibration_input);
                    if(ImGui::Button("Reconstruct with nominal geometry"))nominal_reconstruction=true;
                    ImGui::EndDisabled();
                    ImGui::BeginDisabled(!calibration_data.corrected);
                    if(ImGui::Button("Reconstruct with calibrated geometry"))corrected_reconstruction=true;
                    ImGui::EndDisabled();
                    ImGui::EndDisabled();
                }
                ImGui::End();
            }
            if(help_window){
                ImGui::SetNextWindowSize({520,300},ImGuiCond_FirstUseEver);
                if(ImGui::Begin("Workflow guide",&help_window)){
                    ImGui::TextWrapped("1. File > Open configuration, or Example > choose a built-in phantom.");
                    ImGui::TextWrapped("2. Settings selects the parameter workspace: scan, materials, spectrum, effects or reconstruction.");
                    ImGui::TextWrapped("3. Compute > Forward projection. Geometry and parameter edits take effect on the next computation.");
                    ImGui::TextWrapped("4. Compute > Reconstruction. View opens projections or reconstructed slices in the image drawer.");
                    ImGui::TextWrapped("Built-in phantoms use memory resources. File > Save configuration explicitly exports TOML and the label volume.");
                }
                ImGui::End();
            }
            ImGui::SetNextWindowPos({ 0, menu_height });
            ImGui::SetNextWindowSize({ float(width) * .30f, float(height)-menu_height });
            ImGui::Begin("Parameter workspace", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::BeginDisabled(!enabled);
            if(ImGui::CollapsingHeader("View settings")){
            ImGui::SetNextItemWidth(-130);
            if(ImGui::SliderFloat("Volume opacity",&opacity_scale,0,1,"%.3f",ImGuiSliderFlags_Logarithmic))opacity_changed=true;
            relative_mode=view.relativeGeometryMode();
            ImGui::BeginDisabled(!view.relativeGeometryAvailable());
            if(ImGui::Checkbox("Gantry fixed (relative)",&relative_mode))relative_changed=true;
            ImGui::EndDisabled();
            if(!view.relativeGeometryAvailable())ImGui::TextWrapped("Relative mode requires a rigid source-detector trajectory.");
            }
            ImGui::Spacing();
            auto body=[&](auto draw_content){
                ImGui::BeginChild("Parameter content",{0,-140},false);
                if(!message.empty()){
                    ImGui::PushStyleColor(ImGuiCol_Text,ImVec4(1,.8f,.4f,1));ImGui::TextWrapped("%s",message.c_str());ImGui::PopStyleColor();ImGui::Separator();
                }
                draw_content();ImGui::EndChild();
            };
            if (ImGui::BeginTabBar("Settings")) {
                if (ImGui::BeginTabItem("Scan",nullptr,requested_settings==0?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None)) {
                    body([&]{scan();});
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Volume",nullptr,requested_settings==1?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None)) {
                    body([&]{volume();});
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Spectrum",nullptr,requested_settings==2?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None)) {
                    body([&]{energy();});
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Effects",nullptr,requested_settings==3?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None)) {
                    body([&]{effects();});
                    ImGui::EndTabItem();
                }
                const auto recon_flags=!view.hasReconstruction() && (std::getenv("YK_IMGUI_RECON_SMOKE")||std::getenv("YK_IMGUI_CHAIN_SMOKE"))?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None;
                if(ImGui::BeginTabItem("Recon",nullptr,requested_settings==4?ImGuiTabItemFlags_SetSelected:recon_flags)){body([&]{reconstruction();});ImGui::EndTabItem();}
                if(ImGui::BeginTabItem("Results",nullptr,(select_results||requested_settings==5)?ImGuiTabItemFlags_SetSelected:ImGuiTabItemFlags_None)){body([&]{results(view);});ImGui::EndTabItem();}
                select_results=false;requested_settings=-1;
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
            const float image_left=viewer.expanded&&result_mode==1?.30f:.65f;
            ImGui::SetNextWindowPos({float(width)*image_left,menu_height});
            ImGui::SetNextWindowSize({float(width)*(1-image_left),56});
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
                viewer.imageInteractions(view,enabled);
                ImGui::SetNextWindowPos({float(width)*image_left,float(height)-354});ImGui::SetNextWindowSize({float(width)*(1-image_left),354});
                ImGui::Begin("Reconstruction drawer controls",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize);
                ImGui::BeginDisabled(!enabled);
                viewer.controls(view,ct_reference);
                ImGui::EndDisabled();ImGui::End();
            }
            }else{
                ImGui::SetNextWindowPos({float(width)-310,menu_height+12});ImGui::SetNextWindowSize({298,98});
                ImGui::Begin("Image drawer handle",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoScrollbar);
                if(ImGui::Button("Projection images",{-1,32})){drawer_open=true;drawer_changed=true;result_mode=0;result_changed=true;select_image_page=true;}
                if(ImGui::Button("Reconstruction images",{-1,32})){drawer_open=true;drawer_changed=true;result_mode=1;result_changed=true;select_image_page=true;}
                ImGui::End();
            }
            if(!(drawer_open&&result_mode==1&&viewer.expanded)){
            ImGui::SetNextWindowPos({ float(width) * .30f, float(height) * .90f });
            ImGui::SetNextWindowSize({ float(width) * (drawer_open?.35f:.70f), float(height) * .10f });
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
        }
    };
}
bool runImGuiProjection(const SimulationConfig& initial, const std::filesystem::path& output, std::string& error, const std::vector<std::uint8_t>* initial_labels, const std::string& example_id)
{
    try {
        UiState ui;
        ui.draft = initial;
        if(!example_id.empty())ui.bindCalibrationExample(example_id,examples::clientExamples().create(example_id)->parameters());
        ui.auto_reconstruct=initial.runsReconstruction();
        ui.reconstruction_geometry_initialized=initial.reconstruction.geometry_override.enabled || !initial.source_config_file.empty();
        ui.opacity_scale=float(initial.visualization.volume_opacity_scale);
        ui.draft.projection.output_file = output;

        std::vector<std::uint8_t> builtin_labels = initial_labels ? *initial_labels : std::vector<std::uint8_t>{};
        auto builtin_grid = initial.geometry.parameters;
        auto readLabels = [&](const SimulationConfig& config) {
            const auto& g=config.geometry.parameters;
            if(!config.projection.label_volume.empty()) return readClientLabels(config.projection.label_volume,{g.volume_x,g.volume_y,g.volume_z});
            if(builtin_labels.empty()) throw std::runtime_error("Choose a built-in example or load a volume");
            if(g.volume_x!=builtin_grid.volume_x || g.volume_y!=builtin_grid.volume_y || g.volume_z!=builtin_grid.volume_z)
                throw std::runtime_error("Built-in volume dimensions are fixed; reload the example to restore them");
            return builtin_labels;
        };
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
            ui.viewer.reset(volume);
            view.setReconstruction(std::move(volume));
            view.setReconstructionCtReference(reference.mu_water_mm_inv);
            const auto window=view.reconstructionWindow();ui.viewer.window_low=window[0];ui.viewer.window_high=window[1];
            view.setImageDrawer(true);ui.drawer_open=true;ui.drawer_changed=false;ui.result_mode=1;ui.select_image_page=true;ui.result_changed=false;ui.select_results=true;reconstructed_config=config;
        };
        auto runReconstruction=[&](const SimulationConfig& projection_config, const char* output_suffix="") {
            auto config=projection_config;
            config.reconstruction=ui.draft.reconstruction;
            if(*output_suffix){
                config.reconstruction.geometry_override.enabled=false;
                config.reconstruction.input_projection_file=projection_config.projection.output_file;
                config.reconstruction.output_volume_file=projection_config.projection.output_file.string()+output_suffix+".raw";
                config.reconstruction.slice_prefix=projection_config.projection.output_file.string()+output_suffix;
            }
            const auto& grid=ui.draft.geometry.parameters;auto& g=config.geometry.parameters;
            g.reconstruction_volume_x=grid.reconstruction_volume_x;g.reconstruction_volume_y=grid.reconstruction_volume_y;g.reconstruction_volume_z=grid.reconstruction_volume_z;
            g.reconstruction_voxel_x_mm=grid.reconstruction_voxel_x_mm;g.reconstruction_voxel_y_mm=grid.reconstruction_voxel_y_mm;g.reconstruction_voxel_z_mm=grid.reconstruction_voxel_z_mm;
            g.reconstruction_offset_x_mm=grid.reconstruction_offset_x_mm;g.reconstruction_offset_y_mm=grid.reconstruction_offset_y_mm;g.reconstruction_offset_z_mm=grid.reconstruction_offset_z_mm;
            config.workflow.mode=WorkflowMode::Reconstruct;
            config.projection.scatter.enabled=false; // Forward scatter is not executed by reconstruction.
            if(config.reconstruction.output_volume_file.empty())throw std::runtime_error("Set a reconstruction output volume path");
            const auto validation=std::filesystem::path(output.string()+".recon-validation.toml");
            auto checked=config;
            if(!config.projection.label_volume.empty()){
                saveProjectionUiConfig(config,validation,WorkflowMode::Reconstruct);
                checked=loadConfig(validation);checked.source_config_file=projection_config.source_config_file;
            }
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
            if(!checked.projection.label_volume.empty()) saveProjectionUiConfig(checked,checked.reconstruction.output_volume_file.string()+".toml",WorkflowMode::Reconstruct);
        };
        auto loadResources=[&](const SimulationConfig& config) {
            view.setEditingEnabled(false);
            view.setStatus("Loading volume, materials, spectrum and trajectory...");
            auto loaded_labels=readLabels(config);
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
        auto refreshScanPreview=[&]{
            const auto frames=buildDisplayGeometry(ui.draft);
            if(computed_config && ui.draft.reconstruction.input_projection_file.lexically_normal()==computed_config->projection.output_file.lexically_normal())
                ui.draft.reconstruction.input_projection_file.clear();
            computed_config.reset();reconstructed_config.reset();
            view.setTrajectory(frames);view.setGeometryPreview(frames.front());
            ui.requested_frame=-1;ui.shown_frame=0;
            ui.calibration_data={};ui.calibration_input.reset();ui.calibration_message.clear();
            ui.message="Scan type changed. Geometry preview updated; calculate new projections before reconstruction.";
            view.setStatus("Geometry ready. Press Calculate to generate projections.");
        };
        bool success = true; // Closing a loaded setup without calculating is normal.
        const bool smoke = std::getenv("YK_IMGUI_SMOKE") != nullptr;
        const bool chain_smoke=std::getenv("YK_IMGUI_CHAIN_SMOKE")!=nullptr;
        const bool calibration_smoke=std::getenv("YK_IMGUI_CALIBRATION_SMOKE")!=nullptr;
        ui.calculate = smoke || chain_smoke || calibration_smoke;
        if(calibration_smoke){
            ui.calibration_window=true;
            auto& g=ui.draft.geometry.parameters;
            g.reconstruction_volume_x=g.reconstruction_volume_y=256;g.reconstruction_volume_z=320;
            g.reconstruction_voxel_x_mm=g.reconstruction_voxel_y_mm=g.reconstruction_voxel_z_mm=.6;
        }
        try {loadResources(ui.draft);}
        catch(const std::exception& e){ui.message=e.what();view.setEditingEnabled(true);view.setStatus("Could not load TOML resources. Correct paths or load another TOML.");if(smoke||chain_smoke)throw;}
        if(ui.message.empty())ui.message=initial.source_config_file.string();
        if(const auto* mpr_file=std::getenv("YK_IMGUI_MPR_VOLUME")){
            auto config=ui.draft;config.reconstruction.output_volume_file=mpr_file;
            displayReconstruction(config);ui.viewer.expanded=true;view.setReconstructionWorkspace(true);
            if(std::getenv("YK_IMGUI_MPR_SMOKE")){
                ui.viewer.mpr.indices=ui.viewer.recon_slices;yk::visualization::setMprRotation(ui.viewer.mpr,{20,-15,35});
                view.setReconstructionMpr(ui.viewer.mpr);view.setReconstructionWindow(-160,240);
                ui.viewer.window_low=-160;ui.viewer.window_high=240;
                for(int i=0;i<3;++i)view.pollGui();view.saveSnapshot();return true;
            }
        }
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
            const GeometryKind kinds[]={GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::PlanarSourceOrbit,GeometryKind::NikonTilted};
            for(auto kind:kinds){
                const auto old=buildDisplayGeometry(ui.draft);
                if(view.relativeGeometryAvailable())view.setRelativeGeometryMode(true);
                const std::vector<float> pixels(16,1.f);
                view.update(old.front(),4,4,pixels,"Old computed frame",true);
                view.update(old.back(),4,4,pixels,"Old computed frame",true);
                yk::visualization::FloatVolume volume;volume.dimensions={2,2,2};volume.spacing={1,1,1};volume.values.assign(8,.02f);
                view.setReconstruction(std::move(volume));
                computed_config=ui.draft;reconstructed_config=ui.draft;
                ui.draft.reconstruction.input_projection_file=ui.draft.projection.output_file;
                setProjectionUiScanType(ui.draft,kind);
                ui.draft.geometry.parameters.nikon_tilt_rad=15/degrees;
                refreshScanPreview();
                if(computed_config || reconstructed_config || view.hasReconstruction() || view.cachedFrameCount()!=0 || !ui.draft.reconstruction.input_projection_file.empty())
                    throw std::runtime_error("Scan switching retained old projection/reconstruction state");
                loadResources(ui.draft);
                const auto& g=ui.draft.geometry.parameters;
                const auto orbit=buildDisplayGeometry(ui.draft);
                const double travel=orbit.back().source[2]-orbit.front().source[2];
                if(kind==GeometryKind::FlatCbct && std::abs(travel)>1e-4)throw std::runtime_error("Circular preview consumed stale helical pitch");
                if(kind==GeometryKind::FlatHelical && g.views>1 && std::abs(travel)<1e-4)throw std::runtime_error("Helical preview has no Z travel");
                if(view.cachedFrameCount()!=0 || labels.empty() || ui.spectrum_plot.empty())
                    throw std::runtime_error("TOML resource load did not complete independently of projection");
            }
            const auto external=std::filesystem::path(ui.draft.projection.output_file.string()+".external");
            computed_config=ui.draft;ui.draft.reconstruction.input_projection_file=external;
            refreshScanPreview();
            if(ui.draft.reconstruction.input_projection_file!=external)throw std::runtime_error("Scan preview erased independent external projection input");
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
                if(ui.scan_type_changed){
                    ui.scan_type_changed=false;
                    refreshScanPreview();
                }
                if(ui.relative_changed){ui.relative_changed=false;view.setRelativeGeometryMode(ui.relative_mode);}
                if(ui.drawer_changed){ui.drawer_changed=false;view.setImageDrawer(ui.drawer_open);}
                if(ui.result_changed){ui.result_changed=false;view.showReconstruction(ui.result_mode==1);}
                if(ui.viewer.layout_changed){ui.viewer.layout_changed=false;view.setReconstructionWorkspace(ui.viewer.expanded,ui.viewer.single_panel);}
                if(ui.viewer.slices_changed){ui.viewer.slices_changed=false;ui.viewer.mpr.indices=ui.viewer.recon_slices;view.setReconstructionMpr(ui.viewer.mpr);}
                if(ui.viewer.window_changed){ui.viewer.window_changed=false;view.setReconstructionWindow(ui.viewer.window_low,ui.viewer.window_high);}
                if(ui.viewer.auto_window){ui.viewer.auto_window=false;view.resetReconstructionWindow();const auto range=view.reconstructionWindow();ui.viewer.window_low=range[0];ui.viewer.window_high=range[1];}
                if(ui.load_reconstruction){ui.load_reconstruction=false;displayReconstruction(ui.draft);view.setStatus("Saved reconstruction loaded and displayed.");}
                if(ui.browse_reconstruction){ui.browse_reconstruction=false;
                    const auto path=ui::chooseFile(false,L"Float32 volume (*.raw)\0*.raw\0All files\0*.*\0\0");
                    if(!path.empty()){auto config=ui.draft;config.reconstruction.output_volume_file=path;displayReconstruction(config);
                        view.setStatus("Reconstruction loaded. Grid metadata is used when present; HU reference follows the current configuration.");}}
                if(ui.opacity_changed){ui.opacity_changed=false;ui.draft.visualization.volume_opacity_scale=ui.opacity_scale;view.setVolumeOpacityScale(ui.opacity_scale);}
                if(ui.reconstruct){ui.reconstruct=false;
                    auto config=ui.draft;
                    if(config.reconstruction.input_projection_file.empty())throw std::runtime_error("Calculate new projections or select a reconstruction input file first.");
                    if(computed_config && config.reconstruction.input_projection_file.lexically_normal()==computed_config->projection.output_file.lexically_normal())config.geometry=computed_config->geometry;
                    runReconstruction(config);
                }
                if(ui.calibrate_geometry){
                    ui.calibrate_geometry=false;ui.calibration_data={};ui.calibration_input.reset();ui.calibration_message.clear();
                    if(!ui.known_example_id.empty() && ui.known_example_id.rfind("bead-",0)!=0)
                        throw std::runtime_error("Choose a bead calibration phantom, not a water cylinder.");
                    auto input=computed_config?*computed_config:ui.draft;
                    auto context=ui.calibration_context;const auto& g=input.geometry.parameters;
                    context.full_circle=input.geometry.kind==GeometryKind::FlatCbct && g.views==g.views_per_turn;
                    if(!context.full_circle)throw std::runtime_error("Calibration requires a single full circular flat scan");
                    context.config.tracker.raw_path=input.projection.output_file;
                    context.config.tracker.views=g.views;context.config.tracker.image={g.detector_v,g.detector_u};context.config.pixel={g.pixel_u_mm,g.pixel_v_mm};
                    context.nominal=calibration::nominalGeometry(input.geometry);
                    context.nominal.parameters.sid_mm=ui.nominal_sid;context.nominal.parameters.sdd_mm=ui.nominal_sdd;
                    context.config.initial_sid_mm=ui.nominal_sid;context.config.initial_sdd_mm=ui.nominal_sdd;
                    ui.calibration_input=input;ui.calibration_context=context;
                    view.setEditingEnabled(false);view.setStatus("Detecting beads and fitting geometry...");
                    auto task=std::async(std::launch::async,[context]{
                        calibration::Data data;calibration::GeometryCalibrationStage stage;
                        stage.process(data,context);data.domain=stage.contract().output_domain;return data;
                    });
                    const auto started=std::chrono::steady_clock::now();
                    while(task.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready){
                        view.setStatus("Geometry calibration: "+std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-started).count())+" s");
                        view.pollGui();std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    ui.calibration_data=task.get();ui.calibration_input=input;
                    ui.calibration_context=context;
                    const auto report=std::filesystem::path(input.projection.output_file.string()+".geometry-calibration.json");
                    cbct::calibration::writeCalibrationJson(report,*ui.calibration_data.result);
                    ui.calibration_message="Report: "+report.string();
                    try{calibration::GeometryCorrectionStage correction;correction.process(ui.calibration_data,context);ui.calibration_data.domain=correction.contract().output_domain;}
                    catch(const std::exception& e){ui.calibration_message+=" | ";ui.calibration_message+=e.what();}
                    view.setEditingEnabled(true);view.setStatus("Geometry calibration finished. Review the fit before reconstruction.");
                    if(calibration_smoke){
                        std::cout<<"calibration_rmse_px="<<ui.calibration_data.result->joint_rmse_px<<" converged="<<ui.calibration_data.result->joint_converged<<std::endl;
                        if(!ui.calibration_data.corrected)throw std::runtime_error(ui.calibration_message);
                        auto nominal=input;nominal.geometry=context.nominal;runReconstruction(nominal,".nominal");
                        auto corrected=input;corrected.geometry=*ui.calibration_data.corrected;runReconstruction(corrected,".calibrated");
                        return true;
                    }
                }
                if(ui.nominal_reconstruction || ui.corrected_reconstruction){
                    const bool corrected=ui.corrected_reconstruction;ui.nominal_reconstruction=ui.corrected_reconstruction=false;
                    if(!ui.calibration_input || (corrected&&!ui.calibration_data.corrected))throw std::runtime_error("Run a valid calibration first");
                    auto input=*ui.calibration_input;
                    input.geometry=corrected?*ui.calibration_data.corrected:ui.calibration_context.nominal;
                    runReconstruction(input,corrected?".calibrated":".nominal");
                }
                if (ui.load || ui.create_example) {
                    ui.load = false;
                    const bool create=ui.create_example;ui.create_example=false;
                    std::filesystem::path file;
                    std::optional<examples::ExampleResult> generated;
                    if(create){
                        const auto id=ui.builtin_examples[ui.example_index].id;
                        const auto stamp=std::chrono::system_clock::now().time_since_epoch().count();
                        const auto directory=std::filesystem::current_path()/"out"/"examples"/(id+"-"+std::to_string(stamp));
                                                examples::ExampleRequest request{directory};
                        for(const auto& p:ui.example_parameters)request.parameters[p.id]=p.value;
                        try{generated=examples::clientExamples().create(id)->generate(request);}
                        catch(const std::exception& e){ui.example_error=e.what();throw;}
                    }else file=chooseFile(false, L"Configuration\0*.toml\0All files\0*.*\0\0");
                    if (generated || !file.empty()) {
                        auto loaded = generated ? generated->config : loadConfig(file, false);
                        if(generated){builtin_grid=loaded.geometry.parameters;builtin_labels=std::move(generated->labels);}
                        else builtin_labels.clear();
                        if (loaded.geometry.kind == GeometryKind::CylCbct || loaded.geometry.kind == GeometryKind::CylHelical)
                            throw std::runtime_error("Choose one of the four flat-detector scans");
                        ui.auto_reconstruct=loaded.runsReconstruction();
                        loaded.workflow.mode = WorkflowMode::Project;
                        loaded.visualization.enabled = true;
                        loaded.visualization.interactive = true;
                        loaded.visualization.offscreen = false;
                        loaded.visualization.hold_open = true;
                        ui.draft = std::move(loaded);
                        ui.reconstruction_geometry_initialized=!generated || ui.draft.reconstruction.geometry_override.enabled;
                        ui.opacity_scale=float(ui.draft.visualization.volume_opacity_scale);
                        computed_config.reset();reconstructed_config.reset();view.invalidateResults();
                        ui.known_example_id=generated?ui.builtin_examples[ui.example_index].id:"";
                        ui.calibration_data={};ui.calibration_input.reset();ui.calibration_message.clear();
                        if(generated)ui.bindCalibrationExample(ui.builtin_examples[ui.example_index].id,ui.example_parameters);
                        ui.message.clear();
                        try {loadResources(ui.draft);}
                        catch(const std::exception& e){
                            view.setEditingEnabled(true);
                            view.setStatus("TOML parameters loaded; resource loading failed.");
                            ui.message=e.what();
                            std::cerr<<"TOML resource load failed: "<<file<<": "<<e.what()<<std::endl;
                        }
                        if(ui.message.empty() && generated){ui.example_window=false;ui.requested_settings=0;}
                        if(ui.message.empty())ui.message = generated ? "Example and scan preset applied: "+examples::clientExamples().create(ui.builtin_examples[ui.example_index].id)->scanPreset().name : "Loaded: " + file.string();
                    }
                }
                if (ui.save) {
                    ui.save = false;
                    const auto file = chooseFile(true, L"Configuration\0*.toml\0All files\0*.*\0\0");
                    if (!file.empty()) {
                        auto exported=ui.draft;
                        if(exported.projection.label_volume.empty()){
                            auto raw=file;raw.replace_extension(".labels-"+std::to_string(std::chrono::system_clock::now().time_since_epoch().count())+".mat");
                            const auto data=readLabels(exported);
                            const auto& g=exported.geometry.parameters;
                            writeMatLabels(raw,{g.volume_x,g.volume_y,g.volume_z},data);
                            exported.projection.label_volume=std::filesystem::absolute(raw);
                        }
                        saveProjectionUiConfig(exported, file,ui.auto_reconstruct?WorkflowMode::ProjectAndReconstruct:WorkflowMode::Project);
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
                    ui.calibration_data={};ui.calibration_input.reset();ui.calibration_message.clear();
                    auto trial = ui.draft;
                    trial.workflow.mode = WorkflowMode::Project;
                    trial.visualization.enabled = true;
                    trial.visualization.interactive = true;
                    trial.visualization.offscreen = false;
                    trial.visualization.hold_open = true;
                    const auto validation_file = std::filesystem::absolute(std::filesystem::path(output.string() + ".imgui-validation.toml"));
                    auto checked = trial;
                    if(!trial.projection.label_volume.empty()){
                        saveProjectionUiConfig(trial, validation_file);
                        checked=loadConfig(validation_file);
                    }
                    checked.source_config_file = trial.source_config_file;
                    auto trial_labels = readLabels(checked);
                    validateProjectionUiInputs(checked, trial_labels);
                    auto spectrum = loadProjectionSpectrum(checked.projection);
                    if (spectrum.size() > 256)
                        throw std::runtime_error("CUDA preview supports at most 256 energy bins; increase bin width");
                    if(ui.auto_reconstruct && !smoke && !checked.projection.label_volume.empty()){
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
                        if(!current.projection.label_volume.empty()) saveProjectionUiConfig(current, saved,ui.auto_reconstruct && !smoke?WorkflowMode::ProjectAndReconstruct:WorkflowMode::Project);
                        if(ui.auto_reconstruct && !smoke){
                            auto reconstruction=current;reconstruction.reconstruction.input_projection_file=current.projection.output_file;
                            ui.draft.reconstruction.input_projection_file=current.projection.output_file;
                            runReconstruction(reconstruction);
                        }
                    }
                    if(calibration_smoke){if(!success)throw std::runtime_error(error);ui.calibrate_geometry=true;}
                    if(chain_smoke){view.saveSnapshot();return success;}
                    if (smoke) {
                        view.saveSnapshot();
                        if (!success)
                            return false;
                        if (++smoke_mode == 6)
                            return true;
                        const GeometryKind sequence[]={GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::FlatCbct,
                            GeometryKind::FlatHelical,GeometryKind::PlanarSourceOrbit,GeometryKind::NikonTilted};
                        setProjectionUiScanType(ui.draft,sequence[smoke_mode]);
                        ui.scan_type_changed=true;
                        auto& g = ui.draft.geometry.parameters;
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
                if(ui.calibration_window)ui.calibration_message=e.what();
                view.setEditingEnabled(true);
                view.setStatus("Operation failed. See the message above.");
                if (smoke || calibration_smoke) {
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
