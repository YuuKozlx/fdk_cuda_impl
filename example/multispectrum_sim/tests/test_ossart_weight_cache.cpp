#include "LibraryGeometry.hpp"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
void budget(const char* value){
#ifdef _WIN32
    _putenv_s("YK_OSSART_WEIGHT_CACHE_MIB",value);
#else
    setenv("YK_OSSART_WEIGHT_CACHE_MIB",value,1);
#endif
}
std::vector<float> reconstruct(int cache_mib,double offset,bool reset_twice,bool reinitialize){
    budget(std::to_string(cache_mib).c_str());
    SimulationConfig c;auto& g=c.geometry.parameters;g.views=g.views_per_turn=12;g.detector_u=g.detector_v=64;
    g.volume_x=g.volume_y=g.volume_z=64;g.reconstruction_volume_x=g.reconstruction_volume_y=g.reconstruction_volume_z=64;
    g.reconstruction_voxel_x_mm=g.reconstruction_voxel_y_mm=g.reconstruction_voxel_z_mm=.3;
    c.reconstruction.type="iterative";c.reconstruction.iterative.algorithm="ossart";
    c.reconstruction.iterative.iterations=3;c.reconstruction.iterative.subsets=4;
    auto* session=YK::ReconstructionSessionFactory::create();check(session!=nullptr,"session create");
    g.offset_u_mm=offset;auto system=makeLibrarySystem(c,YK::EPipeline::OSSART);
    std::vector<float> input(12*64*64),output(64*64*64);for(size_t i=0;i<input.size();++i)input[i]=.1f+.003f*(i%17);
    float *dp=nullptr,*dv=nullptr;check(cudaMalloc(&dp,input.size()*4)==cudaSuccess,"dp alloc");check(cudaMalloc(&dv,output.size()*4)==cudaSuccess,"dv alloc");
    cudaMemcpy(dp,input.data(),input.size()*4,cudaMemcpyHostToDevice);
    YK::SExecutionRequest request;request.view_count=12;request.iteration_count=3;
    request.projection={dp,YK::EMemoryLocation::Device,input.size()};request.volume={dv,YK::EMemoryLocation::Device,output.size()};
    if(reinitialize){
        g.offset_u_mm=0;auto first=makeLibrarySystem(c,YK::EPipeline::OSSART);
        check(session->initialize(first),"first init");cudaMemset(dv,0,output.size()*4);cudaDeviceSynchronize();
        check(session->execute(request),"populate old geometry cache");cudaDeviceSynchronize();
    }
    check(session->initialize(system),"target geometry init");
    for(int repeat=0;repeat<(reset_twice?2:1);++repeat){
        if(repeat)session->reset();cudaMemset(dv,0,output.size()*4);cudaDeviceSynchronize();
        check(session->execute(request),"execute");check(cudaDeviceSynchronize()==cudaSuccess,"compute sync");
        std::vector<float> actual(output.size());cudaMemcpy(actual.data(),dv,actual.size()*4,cudaMemcpyDeviceToHost);
        if(repeat)for(size_t i=0;i<output.size();++i)check(std::abs(actual[i]-output[i])<1e-6f,"reset changed cached result");
        output=std::move(actual);
    }
    cudaFree(dp);cudaFree(dv);YK::ReconstructionSessionFactory::destroy(session);return output;
}
int main(){try{
    for(double offset:{0.0,2.25}){
        const auto baseline=reconstruct(0,offset,false,false);
        for(int mib:{1,2,256}){const auto cached=reconstruct(mib,offset,true,true);
            for(size_t i=0;i<cached.size();++i)check(std::isfinite(cached[i])&&std::abs(cached[i]-baseline[i])<1e-6f,"weight cache changed reconstruction");}
    }
    std::cout<<"OSSART uncached, row-only, partial-column, full cache, reset and geometry reinitialization passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
