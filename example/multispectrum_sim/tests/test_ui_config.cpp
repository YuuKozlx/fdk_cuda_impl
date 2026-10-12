#include "ProjectionUiConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    const auto dir=std::filesystem::current_path()/"ui-config-tests";
    std::filesystem::create_directories(dir);
    SimulationConfig c;c.workflow.mode=WorkflowMode::Project;c.visualization.enabled=true;c.visualization.interactive=true;
    auto& g=c.geometry.parameters;g.volume_x=2;g.volume_y=2;g.volume_z=2;g.views=12;g.views_per_turn=12;
    g.detector_u=64;g.detector_v=48;g.source_offset_x_mm=4.5;g.phantom_rotation_z_rad=.3;
    g.pitch_mm_per_turn=80;g.nikon_tilt_rad=.2;g.planar_detector_y_mm=500;g.source_orbit_radius_mm=500;
    g.reconstruction_detector_u=64;g.reconstruction_detector_v=48;
    c.projection.label_volume=dir/"labels.raw";c.projection.spectrum.source="generated";
    c.projection.spectrum.filters={{"Al",2}};c.projection.output_file=dir/"output.raw";
    c.projection.quantum_noise.enabled=false;
    c.projection.batching.views=3;c.projection.batching.max_buffer_mib=32;
    c.projection.materials={{1,"water","H2O","",1}};
    std::vector<std::uint8_t> labels={0,1,1,0,1,0,1,0};
    std::ofstream raw(c.projection.label_volume,std::ios::binary);raw.write(reinterpret_cast<const char*>(labels.data()),labels.size());raw.close();
    const GeometryKind kinds[]={GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::PlanarSourceOrbit,GeometryKind::NikonTilted};
    for(auto kind:kinds){c.geometry.kind=kind;const auto path=dir/"saved.toml";
        saveProjectionUiConfig(c,path);const auto loaded=loadConfig(path);validateProjectionUiInputs(loaded,labels);
        require(loaded.geometry.kind==kind,"scan type roundtrip");
        require(loaded.geometry.parameters.source_offset_x_mm==4.5,"source pose roundtrip");
        require(loaded.geometry.parameters.phantom_rotation_z_rad==.3,"volume pose roundtrip");
        require(loaded.projection.spectrum.source=="generated"&&loaded.projection.spectrum.filters.size()==1,"spectrum roundtrip");
        require(loaded.projection.label_volume==c.projection.label_volume,"file location roundtrip");
        require(loaded.projection.sampling.samples_per_pixel==1,"preview sampling");
        require(loaded.projection.batching.views==3&&loaded.projection.batching.max_buffer_mib==32,"batch limits roundtrip");
        require(loaded.workflow.mode==WorkflowMode::Project&&loaded.visualization.interactive,"workflow roundtrip");}
    c.projection.focal_spot.normal_size_u_mm=.4;c.projection.focal_spot.normal_size_v_mm=.3;
    c.projection.focal_spot.defocus_size_u_mm=1;c.projection.focal_spot.defocus_size_v_mm=1.2;
    c.projection.focal_spot.epsilon=.01;
    for(bool normal:{false,true})for(bool defocus:{false,true}){
        c.projection.focal_spot.enabled=normal;c.projection.focal_spot.defocus_enabled=defocus;
        const auto path=dir/"independent-focus.toml";saveProjectionUiConfig(c,path);const auto restored=loadConfig(path);
        require(restored.projection.focal_spot.enabled==normal&&restored.projection.focal_spot.defocus_enabled==defocus,"independent focus switches roundtrip");
        require(restored.projection.focal_spot.epsilon==.01&&restored.projection.focal_spot.normal_size_u_mm==.4,"disabled focus erased parameters");
    }
    MaterialSpec picked{7,"old-name","Al","",2.7};
    selectBuiltinMaterial(picked,"water");
    const auto* water=findBuiltinMaterial("water");
    require(picked.label==7 && picked.formula.empty() && picked.preset==water->id && picked.name==water->display_name && picked.density_g_cm3==water->density_g_cm3,"builtin selection did not populate material");
    bool unknown=false;try{selectBuiltinMaterial(picked,"missing-material");}catch(const std::exception&){unknown=true;}
    require(unknown && picked.preset==water->id,"invalid selection damaged material");
    c.projection.materials[0]=picked;c.projection.materials[0].label=1;
    saveProjectionUiConfig(c,dir/"picked.toml");
    const auto selected=loadConfig(dir/"picked.toml");
    require(selected.projection.materials[0].preset==water->id && selected.projection.materials[0].formula.empty(),"selected preset did not survive save/load");
    XcomAttenuationProvider provider({});std::vector<double> preset_mu,formula_mu;std::string error;
    const std::vector<double> energies={30,60,100};
    require(provider.query(picked,energies,preset_mu,error),"selected preset did not reach attenuation provider");
    MaterialSpec formula{1,"water","H2O","",1};require(provider.query(formula,energies,formula_mu,error),"water formula query failed");
    for(size_t i=0;i<energies.size();++i)require(std::abs(preset_mu[i]-formula_mu[i])/formula_mu[i]<2e-4,"selected water attenuation mismatch");
    c.geometry.kind=GeometryKind::FlatCbct;
    c.reconstruction.type="iterative";c.reconstruction.iterative.algorithm="sirt";c.reconstruction.iterative.iterations=2;
    c.reconstruction.iterative.regularization.plugin="huber_tv_prox";
    c.reconstruction.iterative.regularization.strength=.004;
    c.reconstruction.iterative.regularization.huber_delta=.002;
    c.reconstruction.iterative.regularization.dimensions=2;
    saveProjectionUiConfig(c,dir/"tv-plugin.toml");
    const auto tv_config=loadConfig(dir/"tv-plugin.toml");
    require(tv_config.reconstruction.iterative.regularization.plugin=="huber_tv_prox" &&
        tv_config.reconstruction.iterative.regularization.strength==.004 &&
        tv_config.reconstruction.iterative.regularization.huber_delta==.002 &&
        tv_config.reconstruction.iterative.regularization.dimensions==2,"TV plugin roundtrip");
    c.reconstruction.iterative.regularization.plugin="none";
    c.reconstruction.type="iterative";c.reconstruction.iterative.algorithm="tigre_os_sart";
    c.reconstruction.iterative.subsets=3;
    saveProjectionUiConfig(c,dir/"tigre-subsets.toml");
    const auto subset_config=loadConfig(dir/"tigre-subsets.toml");
    require(subset_config.reconstruction.iterative.subsets==3,"TIGRE subset-count roundtrip");
    c.reconstruction.iterative.algorithm="sirt";c.reconstruction.iterative.subsets=1;
    c.geometry.kind=GeometryKind::FlatHelical;c.geometry.parameters.pitch_mm_per_turn=4;
    c.reconstruction.type="analytic";c.reconstruction.analytic.pipeline="wfbp";c.reconstruction.analytic.filter="ramlak";
    c.reconstruction.analytic.wfbp_cutoff=.8;c.reconstruction.analytic.wfbp_apodization=.4;
    saveProjectionUiConfig(c,dir/"flat-helical-wfbp.toml",WorkflowMode::ProjectAndReconstruct);
    const auto wfbp=loadConfig(dir/"flat-helical-wfbp.toml");
    require(wfbp.geometry.kind==GeometryKind::FlatHelical&&wfbp.reconstruction.analytic.pipeline=="wfbp"&&wfbp.reconstruction.analytic.wfbp_cutoff==.8&&wfbp.reconstruction.analytic.wfbp_apodization==.4,"flat WFBP roundtrip");
    c.geometry.kind=GeometryKind::FlatCbct;saveProjectionUiConfig(c,dir/"invalid-circular-wfbp.toml",WorkflowMode::ProjectAndReconstruct);
    bool wrongWfbp=false;try{loadConfig(dir/"invalid-circular-wfbp.toml");}catch(const std::exception&){wrongWfbp=true;}
    require(wrongWfbp,"circular WFBP accepted");
    c.reconstruction.analytic.pipeline="fdk";
    c.reconstruction.type="statistical";
    c.reconstruction.statistical.regularization.plugin="smoothed_tv";
    c.reconstruction.statistical.regularization.strength=.4;c.reconstruction.statistical.regularization.shape=.002;
    c.reconstruction.statistical.optimizer.iterations=7;c.reconstruction.statistical.optimizer.step_size=.0001;
    c.reconstruction.statistical.data_fidelity.normalize_by_ray_count=true;
    saveProjectionUiConfig(c,dir/"statistical.toml");
    const auto statistical=loadConfig(dir/"statistical.toml");
    // Old experimental settings must neither select a solver nor survive saving.
    {std::ofstream old(dir/"statistical.toml",std::ios::app);old<<"\n[reconstruction.statistical.composition]\nstrategy = \"invalid-experimental-strategy\"\nprox_iterations = -3\n";}
    const auto migrated=loadConfig(dir/"statistical.toml");
    saveProjectionUiConfig(migrated,dir/"statistical-clean.toml");
    {std::ifstream input(dir/"statistical-clean.toml");std::string saved((std::istreambuf_iterator<char>(input)),{});
     require(saved.find("composition")==std::string::npos&&saved.find("prox_iterations")==std::string::npos,"internal solver controls leaked to TOML");}

    require(statistical.reconstruction.type=="statistical"&&
        statistical.reconstruction.statistical.regularization.plugin=="smoothed_tv"&&
        statistical.reconstruction.statistical.regularization.strength==.4&&statistical.reconstruction.statistical.regularization.shape==.002&&
        statistical.reconstruction.statistical.optimizer.iterations==7&&statistical.reconstruction.statistical.optimizer.step_size==.0001&&
        statistical.reconstruction.statistical.data_fidelity.normalize_by_ray_count,"Statistical configuration roundtrip");
    c.reconstruction.type="iterative";
    c.reconstruction.input_projection_file=c.projection.output_file;c.reconstruction.output_volume_file=dir/"reconstruction.raw";
    c.geometry.parameters.reconstruction_volume_x=2;c.geometry.parameters.reconstruction_volume_y=2;c.geometry.parameters.reconstruction_volume_z=2;
    c.visualization.volume_opacity_scale=.02;
    saveProjectionUiConfig(c,dir/"chain.toml",WorkflowMode::ProjectAndReconstruct);
    const auto chain=loadConfig(dir/"chain.toml");
    require(chain.runsReconstruction() && chain.reconstruction.type=="iterative" && chain.reconstruction.iterative.algorithm=="sirt","reconstruction method roundtrip");
    require(chain.geometry.parameters.reconstruction_volume_z==2 && chain.visualization.volume_opacity_scale==.02,"reconstruction grid or opacity roundtrip");
    c.reconstruction.preprocessing.enabled=true;
    auto& correction=c.reconstruction.preprocessing.scatter;correction.enabled=true;
    correction.air_projection_file=dir/"air.raw";correction.model_file=dir/"correction-model.json";
    {std::ofstream f(correction.air_projection_file,std::ios::binary);float one=1;f.write(reinterpret_cast<const char*>(&one),sizeof(one));}
    {std::ofstream f(correction.model_file);f<<"{}";}
    saveProjectionUiConfig(c,dir/"scatter-correction.toml",WorkflowMode::Reconstruct);
    const auto corrected=loadConfig(dir/"scatter-correction.toml");
    require(corrected.reconstruction.preprocessing.scatter.air_projection_file==correction.air_projection_file && corrected.reconstruction.preprocessing.scatter.model_file==correction.model_file,"scatter correction paths were not saved");
    correction.air_projection_file=dir/"missing-air.raw";
    saveProjectionUiConfig(c,dir/"missing-correction-air.toml",WorkflowMode::Reconstruct);
    bool specific=false;try{loadConfig(dir/"missing-correction-air.toml");}catch(const std::exception& e){specific=std::string(e.what()).find("air projection file does not exist")!=std::string::npos;}
    require(specific,"missing correction air did not identify the offending file");
    correction.enabled=false;c.reconstruction.preprocessing.enabled=false;
    c.projection.scatter.enabled=true;
    c.projection.scatter.model_file=dir/"missing-scatter-model.json";
    saveProjectionUiConfig(c,dir/"missing-model.toml");
    bool rejected=false;try{loadConfig(dir/"missing-model.toml");}catch(const std::exception&){rejected=true;}
    require(rejected,"execution accepted missing scatter model");
    const auto setup=loadConfig(dir/"missing-model.toml",false);
    require(setup.projection.scatter.enabled && setup.projection.scatter.model_file==c.projection.scatter.model_file,"GUI discarded scatter settings");
    validateProjectionUiInputs(setup,labels);
    c.projection.scatter.enabled=false;
    c.projection.sampling.samples_per_pixel=7;c.projection.sampling.seed=77;c.projection.quantum_noise.seed=88;
    auto& glow=c.projection.detector_postprocess.afterglow;
    glow.enabled=true;glow.p=.15;glow.initialization="steady_state";glow.frame_time_ms=10;glow.exposure_time_ms=8;
    glow.weights={.25,.75};glow.time_constants_ms={2,20};
    auto& pp=c.reconstruction.preprocessing;pp.enabled=true;
    auto& bh=pp.beam_hardening;bh.enabled=true;bh.fit_from_water_plate=false;bh.order=3;bh.coefficients={0,1,.01,0};bh.clamp_min=.1;bh.clamp_max=8;
    auto& sc=pp.scatter;sc.enabled=false;sc.guide.enabled=true;sc.guide.sigma_u_pixels=1.5;sc.guide.sigma_v_pixels=2.5;
    sc.solver.type="adaptive_experimental";auto& a=sc.solver.adaptive;a.iterations=13;a.initial_damping=.6;a.min_damping=.2;a.damping_decay=.9;
    a.transmission_guard.low_quantile=.015;a.transmission_guard.risk_threshold=.025;a.transmission_guard.positivity_margin=.85;
    saveProjectionUiConfig(c,dir/"advanced-controls.toml",WorkflowMode::ProjectAndReconstruct);
    const auto advanced=loadConfig(dir/"advanced-controls.toml");
    require(advanced.projection.sampling.samples_per_pixel==7 && advanced.projection.sampling.seed==77 && advanced.projection.quantum_noise.seed==88,"sampling controls were overwritten");
    const auto& loaded_glow=advanced.projection.detector_postprocess.afterglow;
    require(loaded_glow.enabled && loaded_glow.initialization=="steady_state" && loaded_glow.weights==glow.weights && loaded_glow.time_constants_ms==glow.time_constants_ms && loaded_glow.frame_time_ms==10 && loaded_glow.exposure_time_ms==8,"afterglow controls were not preserved");
    const auto& loaded_bh=advanced.reconstruction.preprocessing.beam_hardening;
    require(loaded_bh.order==3 && loaded_bh.coefficients==bh.coefficients && loaded_bh.clamp_min==.1 && loaded_bh.clamp_max==8,"BHC controls were not preserved");
    const auto& loaded_sc=advanced.reconstruction.preprocessing.scatter;
    require(loaded_sc.guide.enabled && loaded_sc.guide.sigma_u_pixels==1.5 && loaded_sc.guide.sigma_v_pixels==2.5,"guidance controls were not preserved");
    require(loaded_sc.solver.type=="adaptive_experimental" && loaded_sc.solver.adaptive.iterations==13 && loaded_sc.solver.adaptive.initial_damping==.6 && loaded_sc.solver.adaptive.min_damping==.2 && loaded_sc.solver.adaptive.damping_decay==.9 && loaded_sc.solver.adaptive.transmission_guard.positivity_margin==.85,"adaptive controls were not preserved");
    sc.solver.type="fixed";saveProjectionUiConfig(c,dir/"back-to-fixed.toml",WorkflowMode::ProjectAndReconstruct);loadConfig(dir/"back-to-fixed.toml");
    labels[0]=9;bool missing=false;try{validateProjectionUiInputs(c,labels);}catch(const std::exception&){missing=true;}
    require(missing,"unmapped label was silently treated as air");labels[0]=0;
    labels.pop_back();bool wrongSize=false;try{validateProjectionUiInputs(c,labels);}catch(const std::exception&){wrongSize=true;}
    require(wrongSize,"raw file size validation");
    std::cout<<"Four scan types, geometry, labels and spectrum roundtrips passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
