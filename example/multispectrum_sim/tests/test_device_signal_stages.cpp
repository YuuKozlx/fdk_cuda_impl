#include "MultiSpecFpStage.hpp"
#include "SpectralProjectionPipeline.hpp"
#include "../src/primary/ProjectionStages.hpp"
#include "../src/primary/CudaProjectionBackend.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
using namespace yk::spectral;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t status){require(status==cudaSuccess,cudaGetErrorString(status));}
struct Storage {
    float* pointer=nullptr;
    explicit Storage(size_t n){check(yk::spectral::cuda_resources::poolAllocate(&pointer,6*n*sizeof(float)));}
    ~Storage(){yk::spectral::cuda_resources::poolRelease(pointer);}
    DeviceProjectionFrame frame(size_t n){return {pointer,pointer+n,pointer+2*n,pointer+3*n,pointer+4*n,pointer+5*n,nullptr};}
};
}
void testDeviceSignalStages(){
    SimulationConfig c;auto& g=c.geometry.parameters;g.detector_u=31;g.detector_v=17;
    c.projection.quantum_noise.enabled=false;auto& d=c.projection.detector_postprocess;
    d.optical_crosstalk={true,{1,2,1},{1,3,1}};d.electronic_crosstalk={true,{1,1,1},{1}};
    d.afterglow.enabled=true;d.afterglow.p=.3;d.afterglow.weights={.4,.6};d.afterglow.time_constants_ms={2,10};
    d.afterglow.frame_time_ms=2;d.afterglow.exposure_time_ms=1;
    d.das.enabled=true;d.das.gain_electrons_per_keV=10;d.das.offset_electrons=5;d.das.saturation_electrons=1800;d.das.adc_lsb_electrons=.5;
    MultiSpecFpStage source(std::make_unique<primary_detail::CudaProjectionBackend>());const size_t n=size_t(g.detector_u)*g.detector_v;
    Storage storage(n);ProjectionStageContext context{c,source};
    for(auto initialization:{"zero","steady_state"}){
        d.afterglow.initialization=initialization;
        auto cpu=primary_detail::makeProjectionStages(c),gpu=primary_detail::makeCudaProjectionStages(c);
        for(auto& stage:cpu)stage->prepare(context);for(auto& stage:gpu)stage->prepare(context);
        for(int view=0;view<5;++view){ProjectionFrame host,device;host.resize(g.detector_u,g.detector_v);device.resize(g.detector_u,g.detector_v);
            host.view=device.view=view;device.location=ProjectionMemoryLocation::Device;device.device=storage.frame(n);
            for(size_t p=0;p<n;++p){host.signal_energy[p]=float((p*7)%37+view*11);host.reference_energy[p]=float(150+(p%13)+view);}
            check(cudaMemcpy(device.device.signal,host.signal_energy.data(),n*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemcpy(device.device.reference,host.reference_energy.data(),n*sizeof(float),cudaMemcpyHostToDevice));
            for(size_t i=0;i<cpu.size();++i){cpu[i]->process(host,context);cpu[i]->processAir(host,context);gpu[i]->process(device,context);gpu[i]->processAir(device,context);}
            std::vector<float> signal(n),air(n);check(cudaMemcpy(signal.data(),device.device.signal,n*sizeof(float),cudaMemcpyDeviceToHost));check(cudaMemcpy(air.data(),device.device.reference,n*sizeof(float),cudaMemcpyDeviceToHost));
            for(size_t p=0;p<n;++p){require(std::abs(signal[p]-host.signal_energy[p])<=3e-6f*std::max(1.f,std::abs(host.signal_energy[p])),"CUDA detector processing disagrees with CPU reference");
                require(std::abs(air[p]-host.reference_energy[p])<=3e-6f*std::max(1.f,std::abs(host.reference_energy[p])),"CUDA air history disagrees with CPU reference");}
        }
        for(auto& stage:gpu)stage->reset();
    }
    // CUDA electronic noise: deterministic for a view and statistically Gaussian.
    c.projection.detector_postprocess={};d.das.enabled=true;d.das.gain_electrons_per_keV=10;d.das.electronic_noise_std_electrons=20;
    g.detector_u=65536;g.detector_v=1;const size_t count=65536;Storage noise(count);
    auto stages=primary_detail::makeCudaProjectionStages(c);for(auto& stage:stages)stage->prepare(context);
    ProjectionFrame f;f.width=int(count);f.height=1;f.location=ProjectionMemoryLocation::Device;f.device=noise.frame(count);
    std::vector<float> input(count,100),first(count),second(count);
    for(int repeat=0;repeat<2;++repeat){check(cudaMemcpy(f.device.signal,input.data(),count*sizeof(float),cudaMemcpyHostToDevice));stages[0]->process(f,context);
        check(cudaMemcpy((repeat?second:first).data(),f.device.signal,count*sizeof(float),cudaMemcpyDeviceToHost));}
    require(first==second,"CUDA DAS fixed-seed reproducibility failed");double mean=0,variance=0;
    for(float value:first)mean+=value;mean/=count;for(float value:first)variance+=(value-mean)*(value-mean);variance/=count;
    require(std::abs(mean-100)<.06&&std::abs(variance-4)<.2,"CUDA DAS Gaussian noise moments changed");
    for(auto& stage:stages)stage->reset();
}
