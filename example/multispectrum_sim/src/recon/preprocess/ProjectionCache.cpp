#include "recon/preprocess/ProjectionCache.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include <stdexcept>
namespace yk::spectral::preprocess {
namespace {
using Json=nlohmann::json;
Json stamp(const std::filesystem::path& file) {
    return {{"path",std::filesystem::weakly_canonical(file).generic_string()},
        {"bytes",std::filesystem::file_size(file)},
        {"modified",std::filesystem::last_write_time(file).time_since_epoch().count()}};
}
Json identity(const SimulationConfig& c,const std::filesystem::path& input) {
    const auto& g=c.geometry.parameters;const auto& pp=c.reconstruction.preprocessing;
    const auto& b=pp.beam_hardening;const auto& s=pp.scatter;
    Json key{{"revision",1},{"input",stamp(input)},
        {"detector",{g.detector_u,g.detector_v,g.views,g.pixel_u_mm,g.pixel_v_mm}},
        {"enabled",pp.enabled},
        {"bhc",{{"enabled",b.enabled},{"fit",b.fit_from_water_plate},{"order",b.order},
            {"coefficients",b.coefficients},{"clamp",{b.clamp_min,b.clamp_max}},
            {"plate",{b.water_plate_max_mm,b.water_plate_samples}}, {"reference_keV",b.reference_energy_keV}}},
        {"scatter",{{"enabled",s.enabled},{"iterations",s.correction.iterations},
            {"damping",s.correction.damping},{"scale",s.correction.scatter_scale},
            {"mask_threshold",s.correction.object_mask_threshold},{"kernel",{s.kernel.mode,s.kernel.segments}},
            {"guide",{s.guide.enabled,s.guide.sigma_u_pixels,s.guide.sigma_v_pixels}},
            {"solver",s.solver.type},{"adaptive",{s.solver.adaptive.iterations,
                s.solver.adaptive.initial_damping,s.solver.adaptive.min_damping,s.solver.adaptive.damping_decay,
                s.solver.adaptive.transmission_guard.low_quantile,s.solver.adaptive.transmission_guard.risk_threshold,
                s.solver.adaptive.transmission_guard.positivity_margin}}}}};
    if(s.enabled){key["air"]=stamp(s.air_projection_file);key["model"]=stamp(s.model_file);}
    if(b.enabled&&b.fit_from_water_plate){
        if(!b.spectrum_file.empty())key["spectrum_file"]=stamp(b.spectrum_file);
        else if(c.projection.spectrum.source=="file")key["spectrum_file"]=stamp(c.projection.spectrum_file);
        else {
            const auto& sp=c.projection.spectrum;
            Json filters=Json::array();for(const auto& f:sp.filters)filters.push_back(
                {f.material,f.thickness_mm,f.atomic_number,f.density_g_cm3});
            key["spectrum"]={{"source",sp.source},{"model",sp.model},{"kvp",sp.kvp},
                {"angle",sp.anode_angle_deg},{"bin",sp.bin_width_keV},{"filters",filters}};
        }
    }
    return key;
}
void writeJson(const std::filesystem::path& path,const Json& data){
    std::ofstream out(path);out.exceptions(std::ios::failbit|std::ios::badbit);out<<data.dump(2);out.close();
}
}
std::filesystem::path cachedProjection(const SimulationConfig& c,const std::filesystem::path& input,
    const ProjectionBuilder& build,std::string& diagnostic,TaskProgressCallback progress){
    if(!c.reconstruction.preprocessing.enabled)return input;
    if(progress)progress({"Checking corrected projection cache",0,0});
    const auto key=identity(c,input);
    const auto file=std::filesystem::path(input.string()+".corrected.raw");
    const auto manifest=std::filesystem::path(file.string()+".json");
    const auto& g=c.geometry.parameters;
    const auto bytes=uint64_t(g.detector_u)*g.detector_v*g.views*sizeof(float);
    try {
        std::ifstream in(manifest);Json saved;in>>saved;
        if(saved.at("identity")==key && saved.at("output")==stamp(file) &&
            std::filesystem::file_size(file)==bytes){
            diagnostic="corrected_projection_cache hit: "+file.string();
            if(progress)progress({"Reusing corrected projection cache",g.views,g.views});
            return file;
        }
    }catch(const std::exception&){ /* Missing/stale/corrupt cache: rebuild. */ }
    const auto staging=std::filesystem::path(file.string()+".partial-"+std::to_string(std::random_device{}()));
    const auto staged_manifest=std::filesystem::path(staging.string()+".json");
    struct Cleanup {std::filesystem::path a,b;~Cleanup(){std::error_code ec;
        std::filesystem::remove(a,ec);std::filesystem::remove(b,ec);}} cleanup{staging,staged_manifest};
    std::string details;
    if(!build(staging,details))throw std::runtime_error(details);
    if(std::filesystem::file_size(staging)!=bytes)throw std::runtime_error("corrected projection cache size mismatch");
    if(identity(c,input)!=key)throw std::runtime_error("preprocessing inputs changed during correction; retry reconstruction");
    // A manifest is published only after the complete projection has been closed.
    // Failed reconstruction does not remove this already successful correction.
    std::error_code ec;std::filesystem::remove(manifest,ec);
    std::filesystem::remove(file);std::filesystem::rename(staging,file);
    writeJson(staged_manifest,{{"identity",key},{"output",stamp(file)},
        {"format","float32"},{"domain","corrected_log_attenuation"},
        {"layout","view,v,u"},{"dimensions",{g.views,g.detector_v,g.detector_u}}, {"diagnostic",details}});
    std::filesystem::rename(staged_manifest,manifest);
    diagnostic=details+"\ncorrected_projection_cache saved: "+file.string();
    return file;
}
}
