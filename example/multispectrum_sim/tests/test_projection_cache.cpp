#include "recon/preprocess/ProjectionCache.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void write(const std::filesystem::path& file,const std::vector<float>& values){
    std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
}
int main(int argc,char** argv){try{
    require(argc==2,"test directory missing");std::filesystem::path dir=argv[1];std::filesystem::create_directories(dir);
    SimulationConfig c;auto& g=c.geometry.parameters;g.detector_u=4;g.detector_v=3;g.views=2;
    auto& pp=c.reconstruction.preprocessing;pp.enabled=true;pp.beam_hardening.enabled=true;
    const auto input=dir/"input.raw";write(input,std::vector<float>(24,.2f));
    const auto cached=std::filesystem::path(input.string()+".corrected.raw");
    std::filesystem::remove(cached);std::filesystem::remove(cached.string()+".json");
    int builds=0;std::string diagnostic;
    auto build=[&](const std::filesystem::path& path,std::string& details){++builds;write(path,std::vector<float>(24,float(builds)));details="corrected";return true;};
    auto run=[&]{return preprocess::cachedProjection(c,input,build,diagnostic);};
    require(run()==cached&&builds==1,"initial correction missing");
    require(run()==cached&&builds==1&&diagnostic.find("hit")!=std::string::npos,"cache not reused");
    c.reconstruction.iterative.algorithm="tigre_sart";c.reconstruction.iterative.iterations=99;g.reconstruction_volume_x=99;
    run();require(builds==1,"reconstruction settings invalidated cache");
    pp.beam_hardening.coefficients={0,2};run();require(builds==2,"BHC change reused stale cache");
    const auto originalTime=std::filesystem::last_write_time(input);
    write(input,std::vector<float>(24,.3f));std::filesystem::last_write_time(input,originalTime+std::chrono::seconds(1));
    run();require(builds==3,"input change reused stale cache");
    write(cached,{0});run();require(builds==4,"truncated cache accepted");
    const auto air=dir/"air.raw",model=dir/"scatter.json";write(air,std::vector<float>(12,1));
    {std::ofstream out(model);out<<"{}";}
    pp.scatter.enabled=true;pp.scatter.air_projection_file=air;pp.scatter.model_file=model;
    run();require(builds==5,"scatter enable reused stale cache");
    pp.scatter.guide.sigma_u_pixels=3;run();require(builds==6,"guide change reused stale cache");
    std::filesystem::last_write_time(model,std::filesystem::last_write_time(model)+std::chrono::seconds(1));
    run();require(builds==7,"model change reused stale cache");
    std::filesystem::last_write_time(air,std::filesystem::last_write_time(air)+std::chrono::seconds(1));
    run();require(builds==8,"air change reused stale cache");
    pp.beam_hardening.fit_from_water_plate=true;c.projection.spectrum.source="generated";
    run();require(builds==9,"water fit change reused stale cache");c.projection.spectrum.kvp=100;
    run();require(builds==10,"spectrum change reused stale cache");
    pp.beam_hardening.order=3;
    auto failed=[](const std::filesystem::path& path,std::string& details){write(path,{1});details="injected failure";return false;};
    bool rejected=false;try{preprocess::cachedProjection(c,input,failed,diagnostic);}catch(const std::exception&){rejected=true;}
    require(rejected,"failed correction published");run();require(builds==11,"failed correction treated as complete cache");
    run();require(builds==11,"successful retry not reused");
    for(const auto& file:std::filesystem::directory_iterator(dir))require(file.path().string().find(".partial-")==std::string::npos,"partial file leaked");
    pp.enabled=false;require(run()==input&&builds==11,"disabled preprocessing did not bypass cache");
    std::cout<<"Persistent projection cache invalidation and failure recovery passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
