#include "examples/DataExamples.hpp"
#include "examples/IcrpExamples.hpp"
#include "io/MatLabelVolume.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace yk::spectral::examples {
namespace {
std::filesystem::path dataRoot() {
    if(const char* value=std::getenv("YK_EXAMPLE_DATA_ROOT")) return std::filesystem::absolute(value);
    if(std::filesystem::exists("data/materials/icrp145/media.csv")) return std::filesystem::absolute("data");
    return MULTISPECTRUM_EXAMPLE_DATA_ROOT;
}
class DataExample final : public IExampleStage {
    const IcrpPhantom* phantom_;
    pipeline::PluginInfo info_;
    SimulationConfig config(const ExampleRequest& request, const IcrpMaterialMap& map={}) const {
        if(phantom_) {
            auto c=makeIcrpConfig(*phantom_,map,request.output_directory,request.output_directory);
            c.projection.label_volume.clear();return c;
        }
        // Use the same physical tissue assignment as the established BrainWeb example.
        SimulationConfig c;c.workflow.mode=WorkflowMode::ProjectAndReconstruct;
        auto& g=c.geometry.parameters;c.geometry.kind=GeometryKind::FlatCbct;
        g.sid_mm=709;g.sdd_mm=1143.699951;g.detector_u=1024;g.detector_v=256;
        g.pixel_u_mm=.5;g.pixel_v_mm=1.5;g.views=g.views_per_turn=360;g.scan_range_rad=6.283185307179586;
        g.volume_x=g.volume_z=181;g.volume_y=217;g.voxel_x_mm=g.voxel_y_mm=g.voxel_z_mm=1;
        g.reconstruction_volume_x=g.reconstruction_volume_z=91;g.reconstruction_volume_y=109;
        g.reconstruction_voxel_x_mm=g.reconstruction_voxel_z_mm=181./91;
        g.reconstruction_voxel_y_mm=217./109;
        const char* names[]={"background-air","csf","grey-matter","white-matter","fat","muscle-skin","skin","skull","glial-matter","connective"};
        const char* presets[]={"mixture.air","compound.water","tissue.icru_brain_adult","tissue.icru_brain_adult","tissue.icru_adipose_adult2","tissue.icru_muscle_adult","tissue.icru_skin_adult","tissue.ncat_skull","tissue.icru_brain_adult","tissue.icru_muscle_adult"};
        const double density[]={.001205,1.007,1.045,1.041,.95,1.05,1.09,1.61,1.04,1.12};
        for(int i=0;i<10;++i)c.projection.materials.push_back({std::uint8_t(i),names[i],"",presets[i],density[i]});
        const auto out=std::filesystem::absolute(request.output_directory);
        c.projection.output_file=out/"projection.raw";c.projection.energy_output_file=out/"energy.raw";c.projection.air_output_file=out/"air.raw";
        c.projection.spectrum.source="generated";c.projection.spectrum.kvp=120;c.projection.spectrum.filters={{"Al",2}};
        c.projection.spectrum.export_file=out/"spectrum.spec";c.projection.mAs_per_view=5;
        c.projection.sampling.samples_per_pixel=4;c.projection.quantum_noise.enabled=true;c.projection.geometry_flux.enabled=true;
        c.reconstruction.type="analytic";c.reconstruction.analytic.pipeline="fdk";c.reconstruction.analytic.filter="hann";
        c.reconstruction.output_volume_file=out/"reconstruction.raw";c.reconstruction.input_projection_file=c.projection.output_file;
        c.reconstruction.preprocessing.enabled=true;c.reconstruction.preprocessing.beam_hardening.enabled=true;
        c.reconstruction.preprocessing.beam_hardening.fit_from_water_plate=true;c.reconstruction.preprocessing.beam_hardening.reference_energy_keV=70;
        c.visualization.enabled=c.visualization.interactive=true;return c;
    }
public:
    explicit DataExample(const IcrpPhantom* phantom):phantom_(phantom) {
        info_.id=phantom?phantom->id():"brainweb-normal";
        info_.name=phantom?std::string(phantom->sex)+" / "+phantom->part:"BrainWeb / normal 1 mm";
        info_.description=phantom?"ICRP145 full material model, original 0.5 mm grid, helical WFBP.":"BrainWeb crisp labels, original 1 mm grid, circular FDK.";
        info_.kind=pipeline::PluginKind::ExampleStage;info_.version="1.0";
        info_.category=phantom?"ICRP145":"BrainWeb";info_.tags={"mat_v5","data_backed","polychromatic"};
    }
    const pipeline::PluginInfo& info()const noexcept override{return info_;}
    std::vector<Parameter> parameters()const override{return {};}
    ScanPreset scanPreset()const override {
        const auto c=config({"out/examples"});const auto& g=c.geometry.parameters;
        return {info_.name,c.geometry.kind,g.views,g.views_per_turn,g.detector_u,g.detector_v,g.sid_mm,g.sdd_mm,g.pixel_u_mm,g.pixel_v_mm,g.pitch_mm_per_turn,g.start_z_mm,c.reconstruction.analytic.pipeline};
    }
    ExampleResult generate(const ExampleRequest& request)const override {
        const auto root=dataRoot();ExampleResult result;
        if(phantom_) {
            const auto folder=root/"materials/icrp145";
            const auto mapping=loadIcrpMaterialMap(folder/(std::string("materials-")+phantom_->sex+".toml"));
            OrganLabelMap organs;std::ifstream in(folder/(std::string("organs-")+phantom_->sex+".csv"));
            if(!in)throw std::runtime_error("ICRP organ mapping is missing from example data bundle");
            std::string line;std::getline(in,line);
            while(std::getline(in,line)) {
                std::istringstream row(line);std::string label,id;std::getline(row,label,',');std::getline(row,id,',');
                const int index=std::stoi(label);if(index<0||index>255)throw std::runtime_error("Invalid ICRP source label");
                if(mapping.labels[index]>=0) organs.emplace(std::stoull(id),std::uint8_t(mapping.labels[index]));
            }
            result.config=config(request,mapping);
            const auto path=root/"phantoms/icrp145"/phantom_->sex/(std::string("data_")+phantom_->part+".mat");
            result.labels=readMatLabels(path,phantom_->dimensions,organs);
        } else {
            result.config=config(request);
            result.labels=readMatLabels(root/"phantoms/brainweb/normal-1mm.mat",{181,217,181});
            for(auto label:result.labels)if(label>9)throw std::runtime_error("BrainWeb contains an unknown tissue label");
        }
        if(request.views>0) {
            auto& g=result.config.geometry.parameters;
            if(result.config.geometry.kind==GeometryKind::FlatHelical)throw std::runtime_error("Use the scan controls to adjust helical views per turn");
            g.views=g.views_per_turn=request.views;
        }
        return result;
    }
};
}
ExampleRegistry clientExamples() {
    auto result=builtinExamples();
    for(const auto& phantom:icrpPhantoms()) {
        const auto* p=&phantom;DataExample stage(p);result.add(stage.info(),[p]{return std::make_unique<DataExample>(p);});
    }
    DataExample brain(nullptr);result.add(brain.info(),[]{return std::make_unique<DataExample>(nullptr);});
    return result;
}
}
