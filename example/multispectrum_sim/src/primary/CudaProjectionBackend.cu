#include "common/YkTimingWatchdog.hpp"
#include "CudaProjectionBackend.hpp"
#include "DisplayGeometry.hpp"
#include "LibraryGeometry.hpp"
#include "WarpSpectralTransport.cuh"
#include "DetectorResponse.cuh"
#include "DetectorSignal.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <limits>
#include "recon/preprocess/DeviceMemoryPool.hpp"

namespace yk::spectral::primary_detail {
namespace {
void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
}
__global__ void initializeSignals(float* expected,float* air,float* signal,float* reference,float* logarithm,size_t n){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){signal[i]=expected[i];reference[i]=air[i];logarithm[i]=0;}}
__global__ void logarithmKernel(const float* signal,const float* reference,float* output,size_t n){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)output[i]=-logf(fmaxf(signal[i]/fmaxf(reference[i],1e-30f),1e-30f));}
__global__ void scatterNoiseDevice(float* signal,const float* primary,const float* sampled,size_t n,float energy,unsigned seed,int view){
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=n)return;
    float scatter=fmaxf(signal[p]-primary[p],0);if(scatter>0&&energy>0){unsigned noise_seed=seed^(unsigned(view)*unsigned(n)+unsigned(p))*0x9e3779b9U;scatter=poissonSample(scatter/energy,noise_seed)*energy;}
    signal[p]=sampled[p]+scatter;
}
struct CudaProjectionBackend::Impl {
    SimulationConfig config;
    DeviceGeometry dg{};
    std::vector<DeviceConeView> host_cone_views;
    int pixels = 0, batch_capacity = 1;
    void* batch_device=nullptr;
    void* batch_host=nullptr;
    size_t batch_downloads=0,batch_uploads=0;
    cudaEvent_t kernel_start=nullptr,kernel_done=nullptr,transfer_done=nullptr,transfer_start=nullptr;
    bool kernel_timing_recorded=false;
    double kernel_ms=0,download_ms=0;
    unsigned long long* primary_profile=nullptr;
    bool warp_primary=true;
    float incident = 0.f, mean_photon_energy = 0.f;
        unsigned char* dl=nullptr; unsigned char* dmask=nullptr; int* dm=nullptr; float *dmu=nullptr,*ds=nullptr,*de=nullptr,*dresponse=nullptr,*do1=nullptr,*do2=nullptr,*do3=nullptr;
        DeviceConeView* d_cone_views=nullptr;
        cudaArray_t label_array=nullptr; cudaTextureObject_t label_texture=0;
        float *dprotective_thickness=nullptr,*dimpurity_thickness=nullptr,*dscintillator_thickness=nullptr;
        float *dprotective_mu=nullptr,*dimpurity_mu=nullptr,*dscintillator_mu=nullptr;

    std::vector<void*> allocations;
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl() {
        if (label_texture) cudaDestroyTextureObject(label_texture);
        if (label_array) cudaFreeArray(label_array);
        if(kernel_start)cudaEventDestroy(kernel_start);
        if(kernel_done)cudaEventDestroy(kernel_done);
        if(transfer_done)cudaEventDestroy(transfer_done);
        if(transfer_start)cudaEventDestroy(transfer_start);
        if(batch_host)cudaFreeHost(batch_host);
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) preprocess::poolRelease(*it);
    }
    template<class T> cudaError_t allocate(T** pointer, size_t bytes) {
        void* allocated = nullptr;
        const auto error = preprocess::poolAllocate(&allocated, bytes);
        if (error != cudaSuccess) return error;
        try { allocations.push_back(allocated); }
        catch (...) { preprocess::poolRelease(allocated); throw; }
        *pointer = static_cast<T*>(allocated);
        return cudaSuccess;
    }
    void prepare(const SimulationConfig& c, const std::vector<std::uint8_t>& labels,
        const SpectralTransmissionModel& model) {
        config = c;
        const auto& g = c.geometry.parameters;
        const auto count=size_t(g.detector_u)*g.detector_v;
        if(g.detector_u<=0||g.detector_v<=0||count>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("invalid CUDA detector size");
        pixels=int(count);
        const int nm = static_cast<int>(c.projection.materials.size());
        const int ne = static_cast<int>(model.spectrum().size());
        const char* kernel_choice=std::getenv("YK_PRIMARY_KERNEL");
        const std::string selected_kernel=kernel_choice?kernel_choice:"";
        if(!selected_kernel.empty()&&selected_kernel!="warp"&&selected_kernel!="scalar")
            throw std::runtime_error("YK_PRIMARY_KERNEL must be warp or scalar");
        warp_primary=selected_kernel!="scalar";
        std::cout<<"primary_kernel="<<(warp_primary?"warp":"scalar")<<'\n';
        if (pixels <= 0 || g.views <= 0 || nm <= 0 || nm > 32 || ne <= 0 || ne > 256)
            throw std::runtime_error("invalid CUDA projection dimensions, materials or spectrum");
        if(g.volume_x<=0||g.volume_y<=0||g.volume_z<=0||labels.size()!=(size_t)g.volume_x*g.volume_y*g.volume_z) throw std::runtime_error("CUDA前投标签体尺寸不匹配");
        if(nm > 32) throw std::runtime_error("CUDA primary projector supports at most 32 materials");
        dg = makeDeviceGeometry(c, nm, ne);
        const auto cone_geometry = buildNonStandardProjectionViews(c);
        std::vector<DeviceConeView> cone_views;
        cone_views.reserve(cone_geometry.size());
        for (const auto& view : cone_geometry)
            cone_views.push_back({make_float3(view.src.x, view.src.y, view.src.z),
                make_float3(view.detS.x, view.detS.y, view.detS.z),
                make_float3(view.detU.x, view.detU.y, view.detU.z),
                make_float3(view.detV.x, view.detV.y, view.detV.z)});
        if (!cone_views.empty() && static_cast<int>(cone_views.size()) != g.views)
            throw std::runtime_error("non-standard projection geometry view count mismatch");
        host_cone_views = cone_views;
        std::vector<int> map(256,-1); for(int m=0;m<nm;++m)map[c.projection.materials[m].label]=m;
        std::vector<float> mu(nm*ne), sw(ne), en(ne); incident=0;
        const double pixel_area_cm2 = g.pixel_u_mm * g.pixel_v_mm / 100.0;
        for(int e=0;e<ne;++e){const auto& point=model.spectrum()[e];sw[e]=(float)(point.fluence_per_keV_cm2_mAs_at_1m*point.bin_width_keV*c.projection.mAs_per_view*pixel_area_cm2);en[e]=(float)point.energy_keV;incident+=sw[e]*en[e];for(int m=0;m<nm;++m)mu[m*ne+e]=(float)(c.projection.materials[m].density_g_cm3*model.massAttenuation(c.projection.materials[m].label)[e]);}
        const bool detector_response = c.projection.detector_response.enabled;
        float total_incident_photons = 0.f;
        for (float value : sw) total_incident_photons += value;
        mean_photon_energy = total_incident_photons > 0.f
            ? incident / total_incident_photons : 0.f;
        std::vector<float> protective_mu, impurity_mu, scintillator_mu;
        std::vector<float> protective_thickness, impurity_thickness, scintillator_thickness;
        if (detector_response) {
            const auto& protective = c.projection.detector_response.protective_layer;
            const auto& impurity = c.projection.detector_response.impurity_layer;
            const auto& scintillator = c.projection.detector_response.scintillator_layer;
            const auto protective_mu_double = model.detectorMassAttenuation(protective);
            const auto scintillator_mu_double = model.detectorMassAttenuation(scintillator);
            const auto impurity_mu_double = impurity.enabled
                ? model.detectorMassAttenuation(impurity) : std::vector<double>(static_cast<size_t>(ne), 0.0);
            protective_mu.assign(protective_mu_double.begin(), protective_mu_double.end());
            impurity_mu.assign(impurity_mu_double.begin(), impurity_mu_double.end());
            scintillator_mu.assign(scintillator_mu_double.begin(), scintillator_mu_double.end());
            protective_thickness = loadLayerThickness(protective, pixels);
            scintillator_thickness = loadLayerThickness(scintillator, pixels);
            impurity_thickness = impurity.enabled
                ? loadLayerThickness(impurity, pixels) : std::vector<float>(static_cast<size_t>(pixels), 0.f);
        }
        auto ck=[](cudaError_t x,const char* w){ checkCuda(x, w); return true; };
        if(!ck(allocate(&dl,labels.size()),"cudaMalloc labels")||!ck(allocate(&dm,256*sizeof(int)),"cudaMalloc map")||!ck(allocate(&dmu,mu.size()*sizeof(float)),"cudaMalloc mu")||!ck(allocate(&ds,sw.size()*sizeof(float)),"cudaMalloc spectrum")||!ck(allocate(&de,en.size()*sizeof(float)),"cudaMalloc energies")) throw std::runtime_error("CUDA preparation failed");
        checkCuda(cudaMemcpy(dl,labels.data(),labels.size(),cudaMemcpyHostToDevice), "Memcpy");checkCuda(cudaMemcpy(dm,map.data(),256*sizeof(int),cudaMemcpyHostToDevice), "Memcpy");checkCuda(cudaMemcpy(dmu,mu.data(),mu.size()*sizeof(float),cudaMemcpyHostToDevice), "Memcpy");checkCuda(cudaMemcpy(ds,sw.data(),sw.size()*sizeof(float),cudaMemcpyHostToDevice), "Memcpy");checkCuda(cudaMemcpy(de,en.data(),en.size()*sizeof(float),cudaMemcpyHostToDevice), "Memcpy");
        if (!cone_views.empty()) {
            if (!ck(allocate(&d_cone_views, cone_views.size() * sizeof(DeviceConeView)),
                    "cudaMalloc non-standard views") ||
                !ck(cudaMemcpy(d_cone_views, cone_views.data(),
                    cone_views.size() * sizeof(DeviceConeView), cudaMemcpyHostToDevice),
                    "upload non-standard views"))
                throw std::runtime_error("CUDA preparation failed");
            dg.cone_views = d_cone_views;
            dg.use_cone_views = 1;
        }
        const bool use_constant_tables = ne <= 256 && nm * ne <= 15000;
        if (use_constant_tables) {
            checkCuda(cudaMemcpyToSymbol(c_mu,mu.data(),mu.size()*sizeof(float)), "MemcpyToSymbol");
            checkCuda(cudaMemcpyToSymbol(c_spectrum,sw.data(),sw.size()*sizeof(float)), "MemcpyToSymbol");
            checkCuda(cudaMemcpyToSymbol(c_energies,en.data(),en.size()*sizeof(float)), "MemcpyToSymbol");
            dg.use_constant_tables=1;
        }
        {
            const cudaChannelFormatDesc desc=cudaCreateChannelDesc<unsigned char>();
            const cudaExtent extent=make_cudaExtent(g.volume_x,g.volume_y,g.volume_z);
            checkCuda(cudaMalloc3DArray(&label_array,&desc,extent), "Malloc3DArray");
            cudaMemcpy3DParms copy={}; copy.srcPtr=make_cudaPitchedPtr(const_cast<unsigned char*>(labels.data()),g.volume_x,g.volume_x,g.volume_y); copy.dstArray=label_array; copy.extent=extent; copy.kind=cudaMemcpyHostToDevice; checkCuda(cudaMemcpy3D(&copy), "Memcpy3D");
            cudaResourceDesc resource={}; resource.resType=cudaResourceTypeArray; resource.res.array.array=label_array;
            cudaTextureDesc texture={}; texture.readMode=cudaReadModeElementType; texture.filterMode=cudaFilterModePoint; texture.addressMode[0]=cudaAddressModeClamp; texture.addressMode[1]=cudaAddressModeClamp; texture.addressMode[2]=cudaAddressModeClamp; texture.normalizedCoords=0;
            checkCuda(cudaCreateTextureObject(&label_texture,&resource,&texture,nullptr), "CreateTextureObject");
            dg.label_texture=label_texture; dg.use_label_texture=1;
        }
        if (detector_response) {
            const size_t pixel_bytes = static_cast<size_t>(pixels) * sizeof(float);
            const size_t energy_bytes = static_cast<size_t>(ne) * sizeof(float);
            const size_t response_bytes = static_cast<size_t>(pixels) * ne * sizeof(float);
            if(!ck(allocate(&dresponse,response_bytes),"cudaMalloc detector response")||
               !ck(allocate(&dprotective_thickness,pixel_bytes),"cudaMalloc protective thickness")||
               !ck(allocate(&dimpurity_thickness,pixel_bytes),"cudaMalloc impurity thickness")||
               !ck(allocate(&dscintillator_thickness,pixel_bytes),"cudaMalloc scintillator thickness")||
               !ck(allocate(&dprotective_mu,energy_bytes),"cudaMalloc protective mu")||
               !ck(allocate(&dimpurity_mu,energy_bytes),"cudaMalloc impurity mu")||
               !ck(allocate(&dscintillator_mu,energy_bytes),"cudaMalloc scintillator mu")) throw std::runtime_error("CUDA preparation failed");
            checkCuda(cudaMemcpy(dprotective_thickness,protective_thickness.data(),pixel_bytes,cudaMemcpyHostToDevice), "Memcpy");
            checkCuda(cudaMemcpy(dimpurity_thickness,impurity_thickness.data(),pixel_bytes,cudaMemcpyHostToDevice), "Memcpy");
            checkCuda(cudaMemcpy(dscintillator_thickness,scintillator_thickness.data(),pixel_bytes,cudaMemcpyHostToDevice), "Memcpy");
            checkCuda(cudaMemcpy(dprotective_mu,protective_mu.data(),energy_bytes,cudaMemcpyHostToDevice), "Memcpy");
            checkCuda(cudaMemcpy(dimpurity_mu,impurity_mu.data(),energy_bytes,cudaMemcpyHostToDevice), "Memcpy");
            checkCuda(cudaMemcpy(dscintillator_mu,scintillator_mu.data(),energy_bytes,cudaMemcpyHostToDevice), "Memcpy");
            const auto& response=c.projection.detector_response;
            const int total=pixels*ne;
            detectorEfficiencyKernel<<<(total+255)/256,256>>>(dg,
                dprotective_thickness,dimpurity_thickness,dscintillator_thickness,
                dprotective_mu,dimpurity_mu,dscintillator_mu,
                static_cast<float>(response.protective_layer.density_g_cm3),
                static_cast<float>(response.impurity_layer.density_g_cm3),
                static_cast<float>(response.scintillator_layer.density_g_cm3),
                response.impurity_layer.enabled?1:0,
                response.oblique_path_correction?1:0,dresponse);
            if(!ck(cudaGetLastError(),"detector efficiency kernel")||
               !ck(cudaDeviceSynchronize(),"detector efficiency synchronize")) throw std::runtime_error("CUDA preparation failed");
            dg.detector_response=1; dg.detector_efficiency=dresponse;
        }

        prepareBatch();
    }
    void prepareBatch() {
        YK::diagnostics::TimingSpan watch("projection.batch.allocate");
        if(config.projection.batching.views<1||config.projection.batching.views>64||config.projection.batching.max_buffer_mib<1||config.projection.batching.max_buffer_mib>256)throw std::runtime_error("invalid projection batch limits");
        size_t free_bytes=0,total=0;checkCuda(cudaMemGetInfo(&free_bytes,&total),"inspect batch memory");
        const size_t reserve=std::min(total/2,std::max(size_t(512ull*1024*1024),total/5));
        const size_t frame_bytes=size_t(pixels)*(6*sizeof(float)+sizeof(unsigned char));
        const size_t available=free_bytes>reserve?free_bytes-reserve:0;
        const size_t limit=size_t(config.projection.batching.max_buffer_mib)*1024*1024;
        if(frame_bytes>limit)throw std::runtime_error("one projection frame exceeds the batch memory budget");
        batch_capacity=int(std::min({size_t(config.projection.batching.views),size_t(dg.views),available/frame_bytes,limit/frame_bytes}));
        if(batch_capacity<1)throw std::runtime_error("insufficient GPU memory for a projection frame with VTK reservation");
        // Retry a smaller batch if other active workspaces consume the pool budget.
        while(true){
            auto status=allocate(&batch_device,size_t(batch_capacity)*frame_bytes);
            if(status==cudaSuccess)break;
            if(status!=cudaErrorMemoryAllocation||batch_capacity==1)checkCuda(status,"allocate projection batch with VTK reservation");
            batch_capacity=std::max(1,batch_capacity/2);
        }
        checkCuda(cudaHostAlloc(&batch_host,size_t(batch_capacity)*frame_bytes,cudaHostAllocDefault),"allocate pinned projection batch");
        do1=static_cast<float*>(batch_device);
        if(const char* flag=std::getenv("YK_PRIMARY_PROFILE")){if(std::string(flag)!="0")checkCuda(allocate(&primary_profile,8*sizeof(unsigned long long)),"allocate primary profile counters");}
        checkCuda(cudaEventCreate(&kernel_start),"create batch timing");
        checkCuda(cudaEventCreate(&kernel_done),"create batch timing");
        checkCuda(cudaEventCreate(&transfer_done),"create batch timing");
        checkCuda(cudaEventCreate(&transfer_start),"create batch timing");
        std::cout<<"projection_batch capacity="<<batch_capacity<<" buffer_bytes="<<size_t(batch_capacity)*frame_bytes<<" VTK_reserve_bytes="<<reserve<<'\n';
    }
    void projectBatch(int view,int count,std::vector<ProjectionFrame>& frames) {
        const size_t elements=size_t(pixels)*count;
        do1=static_cast<float*>(batch_device);do3=do1+elements;do2=do3+elements;
        dmask=reinterpret_cast<unsigned char*>(do1+6*elements);
        const int quantum_noise=config.projection.quantum_noise.enabled?1:0;
        if(primary_profile)checkCuda(cudaMemset(primary_profile,0,8*sizeof(unsigned long long)),"clear primary profile");
        kernel_timing_recorded=false;
        checkCuda(cudaEventRecord(kernel_start),"record batch kernel start");
        if(warp_primary){
            switch(dg.nm){
            case 1: launchWarpPrimaryKernel<1>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 2: launchWarpPrimaryKernel<2>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 3: launchWarpPrimaryKernel<3>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 4: launchWarpPrimaryKernel<4>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 5: launchWarpPrimaryKernel<5>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 6: launchWarpPrimaryKernel<6>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 7: launchWarpPrimaryKernel<7>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 8: launchWarpPrimaryKernel<8>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 9: launchWarpPrimaryKernel<9>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            case 10: launchWarpPrimaryKernel<10>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            default: launchWarpPrimaryKernel<32>(dg,dl,dm,dmu,ds,de,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile!=nullptr); break;
            }
        }else{
            switch (dg.nm) {
            case 1: launchPrimaryKernel<1>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 2: launchPrimaryKernel<2>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 3: launchPrimaryKernel<3>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 4: launchPrimaryKernel<4>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 5: launchPrimaryKernel<5>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 6: launchPrimaryKernel<6>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 7: launchPrimaryKernel<7>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 8: launchPrimaryKernel<8>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 9: launchPrimaryKernel<9>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            case 10: launchPrimaryKernel<10>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            default: launchPrimaryKernel<32>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask,count,primary_profile); break;
            }
        }
        checkCuda(cudaGetLastError(),"batched primary kernel");
        checkCuda(cudaEventRecord(kernel_done),"record batch kernel end");
        if(primary_profile&&!warp_primary){unsigned long long c[8]{};checkCuda(cudaMemcpy(c,primary_profile,sizeof(c),cudaMemcpyDeviceToHost),"read primary profile counters");
            std::cout<<"primary_sample first="<<view<<" count="<<count<<" geometry_cycles="<<c[0]<<" path_cycles="<<c[1]<<" spectrum_cycles="<<c[2]<<" noise_cycles="<<c[3]
                <<" pixels="<<c[4]<<" rays="<<c[5]<<" hits="<<c[6]<<" voxels="<<c[7]<<std::endl;
        }
        initializeSignals<<<int((elements+255)/256),256>>>(do1,do2,do1+3*elements,do1+4*elements,do1+5*elements,elements);
        checkCuda(cudaGetLastError(),"initialize device signal frames");
        frames.resize(count);
        for(int slot=0;slot<count;++slot){auto& f=frames[slot];f.resize(dg.nu,dg.nv);f.view=view+slot;
            f.domain=ProjectionSignalDomain::EnergyKeV;f.location=ProjectionMemoryLocation::Device;f.logarithm_valid=false;
            const size_t offset=size_t(slot)*pixels;
            f.device={do1+offset,do3+offset,do2+offset,do1+3*elements+offset,do1+4*elements+offset,do1+5*elements+offset,dmask+offset};
        }
    }
    void ensureLocation(std::vector<ProjectionFrame>& frames,ProjectionMemoryLocation location){
        if(frames.empty())return;const auto current=frames.front().location;
        for(auto& f:frames)if(f.location!=current)throw std::runtime_error("inconsistent batch memory location");
        if(current==location)return;
        YK::diagnostics::TimingSpan transfer_watch(location==ProjectionMemoryLocation::Host?"projection.batch.download_unpack":"projection.batch.pack_upload","count="+std::to_string(frames.size()));
        const size_t elements=size_t(pixels)*frames.size(),bytes=elements*(6*sizeof(float)+1);
        auto* host=static_cast<float*>(batch_host);auto* masks=reinterpret_cast<unsigned char*>(host+6*elements);
        if(location==ProjectionMemoryLocation::Host){
            checkCuda(cudaEventRecord(transfer_start),"record download start");
            {YK::diagnostics::TimingSpan watch("projection.batch.download","bytes="+std::to_string(bytes));checkCuda(cudaMemcpy(batch_host,batch_device,bytes,cudaMemcpyDeviceToHost),"download final projection batch");}++batch_downloads;
            checkCuda(cudaEventRecord(transfer_done),"record batch completion");checkCuda(cudaEventSynchronize(transfer_done),"finish batch timing");
            float elapsed=0;
            if(!kernel_timing_recorded){checkCuda(cudaEventElapsedTime(&elapsed,kernel_start,kernel_done),"time primary kernel");kernel_ms+=elapsed;kernel_timing_recorded=true;}
            checkCuda(cudaEventElapsedTime(&elapsed,transfer_start,transfer_done),"time final download");download_ms+=elapsed;
        }
        for(size_t slot=0;slot<frames.size();++slot){auto& f=frames[slot];const size_t offset=slot*pixels;
            std::vector<float>* arrays[]={&f.expected_primary_energy,&f.sampled_primary_energy,&f.air_energy,&f.signal_energy,&f.reference_energy,&f.logarithm};
            for(size_t plane=0;plane<6;++plane){arrays[plane]->resize(pixels);
                if(location==ProjectionMemoryLocation::Host)std::memcpy(arrays[plane]->data(),host+plane*elements+offset,size_t(pixels)*sizeof(float));
                else std::memcpy(host+plane*elements+offset,arrays[plane]->data(),size_t(pixels)*sizeof(float));}
            if(location==ProjectionMemoryLocation::Host)std::memcpy(f.object_mask.data(),masks+offset,pixels);
            else std::memcpy(masks+offset,f.object_mask.data(),pixels);
            f.location=location;
        }
        if(location==ProjectionMemoryLocation::Device){checkCuda(cudaMemcpy(batch_device,batch_host,bytes,cudaMemcpyHostToDevice),"upload CPU plugin batch");++batch_uploads;}
    }
    void sampleScatterNoise(ProjectionFrame& frame) {
        if(frame.location==ProjectionMemoryLocation::Device){
            scatterNoiseDevice<<<(pixels+255)/256,256>>>(frame.device.signal,frame.device.expected,frame.device.sampled,pixels,mean_photon_energy,config.projection.quantum_noise.seed^0xa511e9b3U,frame.view);
            checkCuda(cudaGetLastError(),"resident scatter noise");return;
        }
        std::vector<float> scatter(static_cast<size_t>(pixels));
        for (int p = 0; p < pixels; ++p)
            scatter[p] = std::max(frame.signal_energy[p] - frame.expected_primary_energy[p], 0.f);
        checkCuda(cudaMemcpy(do1, scatter.data(), pixels * sizeof(float),
            cudaMemcpyHostToDevice), "upload scatter quantum expectation");
        applyQuantumNoiseAfterScatter<<<(pixels + 255) / 256, 256>>>(
            do1, pixels, mean_photon_energy,
            config.projection.quantum_noise.seed ^ 0xa511e9b3U, frame.view);
        checkCuda(cudaGetLastError(), "scatter quantum noise");
        checkCuda(cudaMemcpy(frame.signal_energy.data(), do1, pixels * sizeof(float),
            cudaMemcpyDeviceToHost), "download scatter quantum noise");
        for (int p = 0; p < pixels; ++p)
            frame.signal_energy[p] += frame.sampled_primary_energy[p];
    }
};
CudaProjectionBackend::CudaProjectionBackend() = default;
CudaProjectionBackend::~CudaProjectionBackend() = default;
void CudaProjectionBackend::prepare(const SimulationConfig& c,
    const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model) {
    reset();
    auto prepared = std::make_unique<Impl>();
    prepared->prepare(c, labels, model);
    impl_ = std::move(prepared);
}
void CudaProjectionBackend::project(int view, ProjectionFrame& frame) {
    if (!impl_ || view < 0 || view >= impl_->dg.views ||
        frame.width != impl_->dg.nu || frame.height != impl_->dg.nv)
        throw std::runtime_error("invalid CUDA backend projection request");
    std::vector<ProjectionFrame> frames;
    impl_->projectBatch(view,1,frames);impl_->ensureLocation(frames,ProjectionMemoryLocation::Host);frame=std::move(frames.front());
}
void CudaProjectionBackend::ensureLocation(std::vector<ProjectionFrame>& frames,ProjectionMemoryLocation location){
    if(!impl_)throw std::runtime_error("CUDA backend is not prepared");impl_->ensureLocation(frames,location);}
void CudaProjectionBackend::finalizeBatch(std::vector<ProjectionFrame>& frames){
    if(frames.empty())return;ensureLocation(frames,ProjectionMemoryLocation::Device);
    for(auto& f:frames){logarithmKernel<<<(impl_->pixels+255)/256,256>>>(f.device.signal,f.device.reference,f.device.logarithm,impl_->pixels);f.logarithm_valid=true;}
    checkCuda(cudaGetLastError(),"device log normalization");
}
void CudaProjectionBackend::synchronize(){checkCuda(cudaStreamSynchronize(nullptr),"synchronize projection stage");}
int CudaProjectionBackend::batchCapacity()const{return impl_?impl_->batch_capacity:1;}
void CudaProjectionBackend::projectBatch(int first,int count,std::vector<ProjectionFrame>& frames){
    if(!impl_||first<0||count<=0||count>impl_->batch_capacity||first>impl_->dg.views-count)
        throw std::runtime_error("invalid CUDA projection batch request");
    impl_->projectBatch(first,count,frames);
}
void CudaProjectionBackend::sampleScatterNoise(ProjectionFrame& frame) {
    if (!impl_) throw std::runtime_error("CUDA backend is not prepared");
    impl_->sampleScatterNoise(frame);
}
ProjectionViewGeometry CudaProjectionBackend::geometryForDisplay(int view) const {
    if(!impl_||view<0||view>=impl_->dg.views)throw std::runtime_error("invalid display geometry request");
    auto g=impl_->dg;
    // dg points at device storage; display uses the identical resolved host views.
    if(g.use_cone_views)g.cone_views=impl_->host_cone_views.data();
    return resolveDisplayGeometry(g,view);
}
void CudaProjectionBackend::reset() noexcept {
    if(impl_&&impl_->batch_downloads)std::cout<<"projection_batch downloads="<<impl_->batch_downloads<<" capacity="<<impl_->batch_capacity<<" CPU_plugin_uploads="<<impl_->batch_uploads<<" kernel_ms="<<impl_->kernel_ms<<" download_ms="<<impl_->download_ms<<std::endl;
    impl_.reset();
}
} // namespace yk::spectral::primary_detail
