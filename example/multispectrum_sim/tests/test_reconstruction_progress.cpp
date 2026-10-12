#include "LibraryGeometry.hpp"
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
struct Progress {int calls=0,last=0,total=0;};
int main(){try{
    SimulationConfig c;auto& g=c.geometry.parameters;
    g.views=g.views_per_turn=12;g.detector_u=32;g.detector_v=24;
    g.volume_x=g.volume_y=g.volume_z=16;g.reconstruction_volume_x=g.reconstruction_volume_y=g.reconstruction_volume_z=16;
    c.reconstruction.type="iterative";c.reconstruction.iterative.algorithm="sirt";c.reconstruction.iterative.iterations=4;
    auto system=makeLibrarySystem(c,YK::EPipeline::SIRT);
    auto* session=YK::ReconstructionSessionFactory::create();require(session&&session->initialize(system),"session init failed");
    std::vector<float> projection(12*32*24,.2f),baseline(16*16*16),observed(baseline.size());
    for(size_t i=0;i<projection.size();++i)projection[i]+=.003f*float(i%17);
    float* dp=nullptr;float* dv=nullptr;
    require(cudaMalloc(reinterpret_cast<void**>(&dp),projection.size()*sizeof(float))==cudaSuccess,"projection alloc failed");
    require(cudaMalloc(reinterpret_cast<void**>(&dv),baseline.size()*sizeof(float))==cudaSuccess,"volume alloc failed");
    require(cudaMemcpy(dp,projection.data(),projection.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess,"upload failed");
    YK::SExecutionRequest request;request.view_count=12;request.iteration_count=4;
    request.projection={dp,YK::EMemoryLocation::Device,projection.size()};request.volume={dv,YK::EMemoryLocation::Device,baseline.size()};
    Progress progress;
    auto callback=[](void* data,int completed,int total){auto& p=*static_cast<Progress*>(data);require(completed>=p.last&&completed<=total&&total>0,"invalid iteration progress");++p.calls;p.last=completed;p.total=total;};
    request.struct_size=offsetof(YK::SExecutionRequest,progress_callback);
    request.progress_callback=callback;request.progress_user_data=&progress;
    cudaMemset(dv,0,baseline.size()*sizeof(float));require(session->execute(request),"old ABI request failed");
    require(progress.calls==0,"old-size request accessed optional callback tail");
    cudaMemcpy(baseline.data(),dv,baseline.size()*sizeof(float),cudaMemcpyDeviceToHost);
    session->reset();
    cudaMemset(dv,0,baseline.size()*sizeof(float));request.struct_size=sizeof(request);
    require(session->execute(request),"progress request failed");
    require(progress.calls==4&&progress.last==4&&progress.total==4,"missing intermediate iteration progress");
    cudaMemcpy(observed.data(),dv,observed.size()*sizeof(float),cudaMemcpyDeviceToHost);
    for(size_t i=0;i<observed.size();++i)require(std::isfinite(observed[i])&&std::abs(observed[i]-baseline[i])<1e-6f,"progress changed reconstruction output");
    YK::ReconstructionSessionFactory::destroy(session);
    // SART has one update per view. Report inside its first sweep, while
    // preserving the no-callback numerical result and monotonic total units.
    c.reconstruction.iterative.algorithm="tigre_sart";
    system=makeLibrarySystem(c,YK::EPipeline::TigreGradient);
    session=YK::ReconstructionSessionFactory::create();require(session&&session->initialize(system),"TIGRE session init failed");
    request.progress_callback=nullptr;request.progress_user_data=nullptr;
    cudaMemset(dv,0,baseline.size()*sizeof(float));require(session->execute(request),"TIGRE baseline failed");
    cudaMemcpy(baseline.data(),dv,baseline.size()*sizeof(float),cudaMemcpyDeviceToHost);
    session->reset();progress={};request.progress_callback=callback;request.progress_user_data=&progress;
    cudaMemset(dv,0,baseline.size()*sizeof(float));require(session->execute(request),"TIGRE progress failed");
    require(progress.calls==48&&progress.last==48&&progress.total==48,"missing TIGRE within-sweep progress");
    cudaMemcpy(observed.data(),dv,observed.size()*sizeof(float),cudaMemcpyDeviceToHost);
    for(size_t i=0;i<observed.size();++i)require(std::isfinite(observed[i])&&std::abs(observed[i]-baseline[i])<1e-6f,"TIGRE progress changed output");
    YK::ReconstructionSessionFactory::destroy(session);
    // Ordinary SART keeps legacy coarse weights (no TIGRE reciprocal cap).
    // Check its own callback invariance rather than equality to TIGRE.
    system=makeLibrarySystem(c,YK::EPipeline::SART);
    session=YK::ReconstructionSessionFactory::create();require(session&&session->initialize(system),"internal SART init failed");
    request.progress_callback=nullptr;request.progress_user_data=nullptr;
    cudaMemset(dv,0,baseline.size()*sizeof(float));require(session->execute(request),"internal SART execution failed");
    cudaMemcpy(observed.data(),dv,observed.size()*sizeof(float),cudaMemcpyDeviceToHost);
    baseline=observed;session->reset();progress={};request.progress_callback=callback;request.progress_user_data=&progress;
    cudaMemset(dv,0,baseline.size()*sizeof(float));require(session->execute(request),"internal SART callback execution failed");
    require(progress.calls==4&&progress.last==4&&progress.total==4,"missing internal SART outer-iteration progress");
    cudaMemcpy(observed.data(),dv,observed.size()*sizeof(float),cudaMemcpyDeviceToHost);
    for(size_t i=0;i<observed.size();++i)require(std::isfinite(observed[i])&&std::abs(observed[i]-baseline[i])<=1e-6f,"internal SART callback changed output");
    cudaFree(dp);cudaFree(dv);YK::ReconstructionSessionFactory::destroy(session);
    std::cout<<"Iteration callback, old ABI and numerical invariance passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
