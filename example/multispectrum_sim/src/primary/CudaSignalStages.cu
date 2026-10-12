#include "ProjectionStages.hpp"
#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"
#include "QuantumSampling.cuh"
#include "recon/preprocess/DeviceMemoryPool.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace yk::spectral::primary_detail {
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
__global__ void convolve(const float* in,float* out,int w,int h,const double* kernel,int count,bool vertical){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=w*h)return;int x=p%w,y=p/w;double sum=0,weight=0;
    for(int k=-count/2;k<=count/2;++k){int sx=vertical?x:x+k,sy=vertical?y+k:y;if(sx<0||sy<0||sx>=w||sy>=h)continue;
        double value=kernel[k+count/2];sum+=value*in[sy*w+sx];weight+=value;}
    out[p]=float(sum/fmax(weight,1e-30));
}
struct LagCoefficient {float weight,tau,exposure,de,dr,steady;};
__global__ void lag(float* signal,float* states,size_t pixels,const LagCoefficient* coeff,int components,float p,bool initialize){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=pixels)return;float input=signal[i],output=(1-p)*input;
    for(int c=0;c<components;++c){auto a=coeff[c];float& state=states[size_t(c)*pixels+i];if(initialize)state=a.steady*input;
        float fraction=1-a.de,trapped=p*a.weight*input;
        output+=state*fraction+trapped*(1-a.tau*fraction/a.exposure);
        float after=state*a.de+trapped*a.tau*fraction/a.exposure;state=after*a.dr;
    }signal[i]=output;
}
__global__ void readout(float* signal,size_t n,DetectorPostprocessConfig::DasConfig c,int view){
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=n)return;
    unsigned seed=c.noise_seed^(unsigned(view)*unsigned(n)+unsigned(p))*0x9e3779b9U;
    double electrons=double(signal[p])*c.gain_electrons_per_keV+c.offset_electrons+
        c.electronic_noise_std_electrons*double(normal(seed));
    if(c.saturation_electrons>0)electrons=fmin(electrons,c.saturation_electrons);
    if(c.adc_lsb_electrons>0)electrons=round(electrons/c.adc_lsb_electrons)*c.adc_lsb_electrons;
    signal[p]=float(electrons/c.gain_electrons_per_keV);
}
__global__ void addScatter(float* signal,const float* scatter,size_t n){size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p<n)signal[p]+=scatter[p];}
class DeviceStage:public IProjectionStage {
public:ProjectionSignalDomain domain()const override{return ProjectionSignalDomain::EnergyKeV;}
    ProjectionMemoryLocation location()const override{return ProjectionMemoryLocation::Device;}
};
class Scatter final:public DeviceStage {
    ScatterConfig config_;std::unique_ptr<FasksRuntimeModel> model_;std::unique_ptr<CudaFasksScatter> scatter_;float* prediction_=nullptr;size_t pixels_=0;
public:
    explicit Scatter(ScatterConfig c):config_(std::move(c)){}
    ~Scatter(){reset();}
    const char* name()const override{return "scatter";}
    ProjectionStagePhase phase()const override{return ProjectionStagePhase::Scatter;}
    void prepare(const SimulationConfig& c)override{const auto& g=c.geometry.parameters;pixels_=size_t(g.detector_u)*g.detector_v;
        model_=std::make_unique<FasksRuntimeModel>(config_.model_file);
        scatter_=std::make_unique<CudaFasksScatter>(g.detector_u,g.detector_v,g.pixel_u_mm,g.pixel_v_mm,*model_,config_.kernel.mode=="segmented"?config_.kernel.segments:0);
        scatter_->prepareDeviceMask();check(preprocess::poolAllocate(&prediction_,pixels_*sizeof(float)));
    }
    void process(ProjectionFrame& f,const ProjectionStageContext&)override{
        scatter_->predictDeviceMask(f.device.expected,f.device.air,f.device.mask,prediction_);
        addScatter<<<int((pixels_+255)/256),256>>>(f.device.signal,prediction_,pixels_);check(cudaGetLastError());
    }
    void reset()noexcept override{preprocess::poolRelease(prediction_);prediction_=nullptr;scatter_.reset();model_.reset();}
};
class Quantum final:public DeviceStage {
public:const char* name()const override{return "scatter_noise";}
    ProjectionStagePhase phase()const override{return ProjectionStagePhase::QuantumNoise;}
    void process(ProjectionFrame& f,const ProjectionStageContext& c)override{
        if(c.config.projection.scatter.enabled)c.backend.sampleScatterNoise(f);
        else check(cudaMemcpy(f.device.signal,f.device.sampled,size_t(f.width)*f.height*sizeof(float),cudaMemcpyDeviceToDevice));
    }
};
class Crosstalk final:public DeviceStage {
    CrosstalkConfig config_;ProjectionStagePhase phase_;const char* name_;double *u_=nullptr,*v_=nullptr;float* scratch_=nullptr;
public:Crosstalk(CrosstalkConfig c,ProjectionStagePhase phase,const char* name):config_(std::move(c)),phase_(phase),name_(name){}
    ~Crosstalk(){reset();}
    const char* name()const override{return name_;}ProjectionStagePhase phase()const override{return phase_;}
    void prepare(const SimulationConfig& c)override{
        check(preprocess::poolAllocate(&u_,config_.kernel_u.size()*sizeof(double)));check(preprocess::poolAllocate(&v_,config_.kernel_v.size()*sizeof(double)));
        check(preprocess::poolAllocate(&scratch_,size_t(c.geometry.parameters.detector_u)*c.geometry.parameters.detector_v*sizeof(float)));
        check(cudaMemcpy(u_,config_.kernel_u.data(),config_.kernel_u.size()*sizeof(double),cudaMemcpyHostToDevice));
        check(cudaMemcpy(v_,config_.kernel_v.data(),config_.kernel_v.size()*sizeof(double),cudaMemcpyHostToDevice));
    }
    void apply(float* p,int w,int h){int blocks=(w*h+255)/256;
        convolve<<<blocks,256>>>(p,scratch_,w,h,u_,int(config_.kernel_u.size()),false);
        convolve<<<blocks,256>>>(scratch_,p,w,h,v_,int(config_.kernel_v.size()),true);check(cudaGetLastError());}
    void process(ProjectionFrame& f,const ProjectionStageContext&)override{apply(f.device.signal,f.width,f.height);}
    void processAir(ProjectionFrame& f,const ProjectionStageContext&)override{apply(f.device.reference,f.width,f.height);}
    void reset()noexcept override{preprocess::poolRelease(u_);preprocess::poolRelease(v_);preprocess::poolRelease(scratch_);u_=v_=nullptr;scratch_=nullptr;}
};
class Afterglow final:public DeviceStage {
    DetectorPostprocessConfig::AfterglowConfig config_;LagCoefficient* coefficients_=nullptr;float* states_=nullptr;size_t pixels_=0;bool object_first_=true,air_first_=true;
public:explicit Afterglow(DetectorPostprocessConfig::AfterglowConfig c):config_(std::move(c)){}
    ~Afterglow(){reset();}
    const char* name()const override{return "afterglow";}ProjectionStagePhase phase()const override{return ProjectionStagePhase::Afterglow;}
    void prepare(const SimulationConfig& c)override{
        if(config_.weights.empty()||config_.weights.size()!=config_.time_constants_ms.size())throw std::runtime_error("invalid CUDA afterglow components");
        pixels_=size_t(c.geometry.parameters.detector_u)*c.geometry.parameters.detector_v;std::vector<LagCoefficient> coeff;
        float exposure=float(config_.exposure_time_ms),readout=float(config_.frame_time_ms-config_.exposure_time_ms);
        for(size_t i=0;i<config_.weights.size();++i){float tau=float(config_.time_constants_ms[i]),de=std::exp(-exposure/tau),dr=std::exp(-readout/tau);
            float input=float(config_.p)*float(config_.weights[i])*tau*(1-de)/exposure;
            coeff.push_back({float(config_.weights[i]),tau,exposure,de,dr,dr*input/std::max(1-de*dr,1e-6f)});}
        check(preprocess::poolAllocate(&coefficients_,coeff.size()*sizeof(LagCoefficient)));check(cudaMemcpy(coefficients_,coeff.data(),coeff.size()*sizeof(LagCoefficient),cudaMemcpyHostToDevice));
        check(preprocess::poolAllocate(&states_,2*pixels_*coeff.size()*sizeof(float)));check(cudaMemset(states_,0,2*pixels_*coeff.size()*sizeof(float)));
    }
    void apply(float* p,bool air){bool& first=air?air_first_:object_first_;bool initialize=first&&config_.initialization=="steady_state";
        lag<<<int((pixels_+255)/256),256>>>(p,states_+(air?pixels_*config_.weights.size():0),pixels_,coefficients_,int(config_.weights.size()),float(config_.p),initialize);
        check(cudaGetLastError());first=false;}
    void process(ProjectionFrame& f,const ProjectionStageContext&)override{apply(f.device.signal,false);}
    void processAir(ProjectionFrame& f,const ProjectionStageContext&)override{apply(f.device.reference,true);}
    void reset()noexcept override{preprocess::poolRelease(states_);preprocess::poolRelease(coefficients_);states_=nullptr;coefficients_=nullptr;object_first_=air_first_=true;}
};
class Das final:public DeviceStage {
    DetectorPostprocessConfig::DasConfig config_;
public:explicit Das(DetectorPostprocessConfig::DasConfig c):config_(c){}
    const char* name()const override{return "das";}ProjectionStagePhase phase()const override{return ProjectionStagePhase::Readout;}
    void process(ProjectionFrame& f,const ProjectionStageContext&)override{readout<<<(f.width*f.height+255)/256,256>>>(f.device.signal,size_t(f.width)*f.height,config_,f.view);check(cudaGetLastError());}
};
}
std::vector<std::unique_ptr<IProjectionStage>> makeCudaProjectionStages(const SimulationConfig& c){
    std::vector<std::unique_ptr<IProjectionStage>> stages;const auto& p=c.projection;const auto& d=p.detector_postprocess;
    if(p.scatter.enabled)stages.push_back(std::make_unique<Scatter>(p.scatter));
    if(p.quantum_noise.enabled)stages.push_back(std::make_unique<Quantum>());
    if(d.optical_crosstalk.enabled)stages.push_back(std::make_unique<Crosstalk>(d.optical_crosstalk,ProjectionStagePhase::OpticalCrosstalk,"optical_crosstalk"));
    if(d.afterglow.enabled)stages.push_back(std::make_unique<Afterglow>(d.afterglow));
    if(d.electronic_crosstalk.enabled)stages.push_back(std::make_unique<Crosstalk>(d.electronic_crosstalk,ProjectionStagePhase::ElectronicCrosstalk,"electronic_crosstalk"));
    if(d.das.enabled)stages.push_back(std::make_unique<Das>(d.das));return stages;
}
}
