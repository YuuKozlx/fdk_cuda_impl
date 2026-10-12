#include "LibraryGeometry.hpp"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral;
void ck(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void setting(const char* name,const char* value){
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
std::vector<float> run(const std::string& algorithm,int gpu,int cpu,bool reset,double offset,int block=0,float reduction=0,float voxel=0){
    setting("YK_TIGRE_WEIGHT_GPU_MIB",std::to_string(gpu).c_str());
    setting("YK_TIGRE_WEIGHT_CPU_MIB",std::to_string(cpu).c_str());
    SimulationConfig c;auto& g=c.geometry.parameters;
    g.views=g.views_per_turn=13;g.detector_u=32;g.detector_v=24;g.pixel_u_mm=.6;g.pixel_v_mm=.7;g.offset_u_mm=offset;
    g.volume_x=g.volume_y=g.volume_z=32;g.reconstruction_volume_x=g.reconstruction_volume_y=32;g.reconstruction_volume_z=20;
    c.reconstruction.type="iterative";auto& it=c.reconstruction.iterative;it.algorithm=algorithm;it.iterations=3;it.subsets=4;it.relaxation=.73;it.tv_iterations=2;
    auto system=makeLibrarySystem(c,algorithm=="sart"?YK::EPipeline::SART:YK::EPipeline::TigreGradient,
        YK::EProjectionModel::Joseph,YK::EFdkFilter::SheppLogan,GeometryUse::Reconstruction);
    if(block>0)system.reconstruction.tigre.block_size=block;
    system.reconstruction.tigre.lambda_reduction=reduction;
    if(voxel>0){auto& v=system.geometry.volume;v.voxel_x_mm=v.voxel_y_mm=v.voxel_z_mm=voxel;}
    std::vector<float> input(13*32*24),output(32*32*20);for(size_t i=0;i<input.size();++i)input[i]=.1f+.002f*float(i%31);
    float *dp=nullptr,*dv=nullptr;ck(cudaMalloc(&dp,input.size()*4)==cudaSuccess,"input allocation");ck(cudaMalloc(&dv,output.size()*4)==cudaSuccess,"volume allocation");
    cudaMemcpy(dp,input.data(),input.size()*4,cudaMemcpyHostToDevice);
    auto* session=YK::ReconstructionSessionFactory::create();ck(session&&session->initialize(system),"session initialize");
    YK::SExecutionRequest request;request.view_count=13;request.iteration_count=3;
    request.projection={dp,YK::EMemoryLocation::Device,input.size()};request.volume={dv,YK::EMemoryLocation::Device,output.size()};
    const auto execute=[&]{cudaMemset(dv,0,output.size()*4);ck(session->execute(request),"execute failed");cudaMemcpy(output.data(),dv,output.size()*4,cudaMemcpyDeviceToHost);};
    if(offset!=0){
        g.offset_u_mm=0;const auto prior=makeLibrarySystem(c,algorithm=="sart"?YK::EPipeline::SART:YK::EPipeline::TigreGradient,
            YK::EProjectionModel::Joseph,YK::EFdkFilter::SheppLogan,GeometryUse::Reconstruction);
        ck(session->initialize(prior),"prior geometry init");execute();
        ck(session->initialize(system),"geometry reinitialize");
    }
    execute();const auto first=output;
    if(reset){session->reset();execute();for(size_t i=0;i<output.size();++i)ck(std::abs(first[i]-output[i])<=1e-6f,"reset changed output");
        session->reset();execute();}
    YK::ReconstructionSessionFactory::destroy(session);cudaFree(dp);cudaFree(dv);return output;
}
int main(int argc,char** argv){try{
    if(argc==1){
        const auto eight=run("tigre_os_sart",0,0,false,0,8),seven=run("tigre_os_sart",0,0,false,0,7);
        double difference=0;for(size_t i=0;i<eight.size();++i)difference+=std::abs(eight[i]-seven[i]);
        ck(difference>1e-6,"SDK block_size ignored or rounded back to subset count");
        const auto automatic=run("tigre_sart_tv",0,0,false,0,0,0),decay=run("tigre_sart_tv",0,0,false,0,0,.99f),constant=run("tigre_sart_tv",0,0,false,0,0,1);
        difference=0;for(size_t i=0;i<automatic.size();++i){ck(std::abs(automatic[i]-decay[i])<1e-6,"POCS automatic decay differs from 0.99");difference+=std::abs(automatic[i]-constant[i]);}
        ck(difference>1e-6,"POCS explicit constant relaxation not honored");
        const auto art_auto=run("tigre_sart",0,0,false,0),art_one=run("tigre_sart",0,0,false,0,0,1);
        for(size_t i=0;i<art_auto.size();++i)ck(std::abs(art_auto[i]-art_one[i])<1e-6,"ART default relaxation changed");
        const auto short_legacy=run("sart",0,0,false,0,0,0,.02f),short_tigre=run("tigre_sart",0,0,false,0,0,0,.02f);
        double legacy_sum=0,tigre_sum=0;for(size_t i=0;i<short_tigre.size();++i){ck(std::isfinite(short_tigre[i]),"short ray instability");legacy_sum+=std::abs(short_legacy[i]);tigre_sum+=std::abs(short_tigre[i]);}
        ck(legacy_sum>0&&tigre_sum<legacy_sum*.8,"short-path inverse weight cap ineffective or changed legacy SART");
    }
    const char* algorithms[]={"sart","tigre_sart","tigre_os_sart","tigre_sirt","tigre_sart_tv","tigre_os_sart_tv"};
    for(const auto* algo:algorithms){const auto reference=run(algo,0,0,false,0);
        if(argc==3&&std::string(argv[1])=="--baseline"){
            std::filesystem::create_directories(argv[2]);std::ofstream out(std::filesystem::path(argv[2])/(std::string(algo)+".raw"),std::ios::binary);
            out.write(reinterpret_cast<const char*>(reference.data()),reference.size()*4);continue;
        }
        for(const auto limits:{std::pair{0,32},std::pair{1,0},std::pair{1,32},std::pair{256,4096}}){
            const auto output=run(algo,limits.first,limits.second,true,0);
            for(size_t i=0;i<output.size();++i)ck(std::isfinite(output[i])&&std::abs(reference[i]-output[i])<=1e-6f,"weight cache changed result");
        }
        const auto changed=run(algo,1,32,false,1.3),changed_reference=run(algo,0,0,false,1.3);
        for(size_t i=0;i<changed.size();++i)ck(std::abs(changed[i]-changed_reference[i])<=1e-6f,"geometry cache mismatch");
        std::cout<<algo<<" cache modes and reset passed\n";
    }
    if(argc!=3){
        SimulationConfig huge;auto& g=huge.geometry.parameters;g.views=g.views_per_turn=720;
        g.detector_u=g.detector_v=4096;g.volume_x=g.volume_y=g.volume_z=2048;
        huge.reconstruction.type="iterative";huge.reconstruction.iterative.algorithm="tigre_sirt";
        auto* session=YK::ReconstructionSessionFactory::create();ck(session!=nullptr,"budget session");
        ck(!session->initialize(makeLibrarySystem(huge,YK::EPipeline::TigreGradient)),"oversized mandatory volume was accepted");
        YK::ReconstructionSessionFactory::destroy(session);
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
