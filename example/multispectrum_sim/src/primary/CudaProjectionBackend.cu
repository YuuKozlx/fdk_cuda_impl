#include "CudaProjectionBackend.hpp"
#include "LibraryGeometry.hpp"
#include "SpectralTransport.cuh"
#include "DetectorResponse.cuh"
#include "DetectorSignal.hpp"
#include <algorithm>
#include <stdexcept>

namespace yk::spectral::primary_detail {
namespace {
void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
}
struct CudaProjectionBackend::Impl {
    SimulationConfig config;
    DeviceGeometry dg{};
    std::vector<DeviceConeView> host_cone_views;
    int pixels = 0;
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
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) cudaFree(*it);
    }
    template<class T> cudaError_t allocate(T** pointer, size_t bytes) {
        void* allocated = nullptr;
        const auto error = cudaMalloc(&allocated, bytes);
        if (error != cudaSuccess) return error;
        try { allocations.push_back(allocated); }
        catch (...) { cudaFree(allocated); throw; }
        *pointer = static_cast<T*>(allocated);
        return cudaSuccess;
    }
    void prepare(const SimulationConfig& c, const std::vector<std::uint8_t>& labels,
        const SpectralTransmissionModel& model) {
        config = c;
        const auto& g = c.geometry.parameters;
        pixels = g.detector_u * g.detector_v;
        const int nm = static_cast<int>(c.projection.materials.size());
        const int ne = static_cast<int>(model.spectrum().size());
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
        if(!ck(allocate(&dl,labels.size()),"cudaMalloc labels")||!ck(allocate(&dmask,pixels),"cudaMalloc object mask")||!ck(allocate(&dm,256*sizeof(int)),"cudaMalloc map")||!ck(allocate(&dmu,mu.size()*sizeof(float)),"cudaMalloc mu")||!ck(allocate(&ds,sw.size()*sizeof(float)),"cudaMalloc spectrum")||!ck(allocate(&de,en.size()*sizeof(float)),"cudaMalloc energies")||!ck(allocate(&do1,pixels*sizeof(float)),"cudaMalloc expected primary")||!ck(allocate(&do2,pixels*sizeof(float)),"cudaMalloc air energy")||!ck(allocate(&do3,pixels*sizeof(float)),"cudaMalloc sampled primary")) throw std::runtime_error("CUDA preparation failed");
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

    }
    void project(int view, ProjectionFrame& frame) {
        const int quantum_noise = config.projection.quantum_noise.enabled ? 1 : 0;
        auto ck=[](cudaError_t error, const char* operation) {
            checkCuda(error, operation); return true;
        };
            switch (dg.nm) {
            case 1: launchPrimaryKernel<1>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 2: launchPrimaryKernel<2>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 3: launchPrimaryKernel<3>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 4: launchPrimaryKernel<4>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 5: launchPrimaryKernel<5>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 6: launchPrimaryKernel<6>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 7: launchPrimaryKernel<7>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 8: launchPrimaryKernel<8>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 9: launchPrimaryKernel<9>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            case 10: launchPrimaryKernel<10>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            default: launchPrimaryKernel<32>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,view,config.projection.sampling.seed,config.projection.quantum_noise.seed,do1,do3,do2,dmask); break;
            }
            if(!ck(cudaGetLastError(),"primary kernel")||!ck(cudaDeviceSynchronize(),"primary synchronize"))throw std::runtime_error("CUDA primary launch failed");
            checkCuda(cudaMemcpy(frame.expected_primary_energy.data(),do1,pixels*sizeof(float),cudaMemcpyDeviceToHost), "download projection frame");
            checkCuda(cudaMemcpy(frame.air_energy.data(),do2,pixels*sizeof(float),cudaMemcpyDeviceToHost), "download projection frame");
            checkCuda(cudaMemcpy(frame.sampled_primary_energy.data(),do3,pixels*sizeof(float),cudaMemcpyDeviceToHost), "download projection frame");
            checkCuda(cudaMemcpy(frame.object_mask.data(),dmask,pixels,cudaMemcpyDeviceToHost), "download projection frame");

    }
    void sampleScatterNoise(ProjectionFrame& frame) {
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
    impl_->project(view, frame);
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
    auto point=[](float3 p)->yk::visualization::Point{return {p.x,p.y,p.z};};
    auto unit=[](float3 p){return scale3(p,1.f/sqrtf(p.x*p.x+p.y*p.y+p.z*p.z));};
    ProjectionViewGeometry out;out.view=view;
    const float cu=.5f*(g.nu-1),cv=.5f*(g.nv-1);
    auto center=scannerRay(g,view,cu,cv);out.source=point(center.source);out.detector_center=point(center.detector);
    float3 u,v,n;
    if(g.use_cone_views){u=unit(g.cone_views[view].detector_u);v=unit(g.cone_views[view].detector_v);}
    else {
        const float pi=3.14159265358979323846f;
        const float angle=g.start+g.rotation_direction*2.f*pi*view/g.views_per_turn-.5f*pi;
        detectorFrame(g,angle,u,v,n);
        if(g.cylindrical){const float gamma=g.offset_u/g.sdd;u=add3(scale3(u,cosf(gamma)),scale3(n,sinf(gamma)));}
        u=unit(u);v=unit(v);
    }
    out.detector_axes={point(u),point(v),point(unit(cross3(u,v)))};
    const int segments=g.cylindrical?128:1;
    for(int row=0;row<2;++row)for(int col=0;col<=segments;++col)
        out.detector.push_back(point(scannerRay(g,view,-.5f+g.nu*float(col)/segments,row?g.nv-.5f:-.5f).detector));
    const auto origin=phantomForward(g,make_float3(0,0,0),false);
    const float3 basis[]={make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
    const auto t=point(origin);for(int r=0;r<3;++r)out.object_pose[r*4+3]=t[r];
    for(int c=0;c<3;++c){const auto b=point(phantomForward(g,basis[c],true));for(int r=0;r<3;++r)out.object_pose[r*4+c]=b[r];}
    return out;
}
void CudaProjectionBackend::reset() noexcept { impl_.reset(); }
} // namespace yk::spectral::primary_detail
