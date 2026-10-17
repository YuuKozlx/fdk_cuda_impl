#include "primary/ScalarProjector.hpp"
#include "config/SimConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include "primary/ProjectionGeometry.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fp = yk::spectral::scalar_fp;
using namespace yk::spectral;
static void checked(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n){checked(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
struct Events {
    cudaEvent_t a{},b{};
    Events(){checked(cudaEventCreate(&a));checked(cudaEventCreate(&b));}
    ~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}
    template<class F> float time(F f){checked(cudaEventRecord(a));f();checked(cudaEventRecord(b));checked(cudaEventSynchronize(b));float ms=0;checked(cudaEventElapsedTime(&ms,a,b));return ms;}
};
struct Texture {
    cudaArray_t array=nullptr;cudaTextureObject_t id=0,point=0;
    Texture(const float* volume,const fp::Grid& g){
        auto channel=cudaCreateChannelDesc<float>();
        checked(cudaMalloc3DArray(&array,&channel,make_cudaExtent(g.nx,g.ny,g.nz)));
        cudaMemcpy3DParms copy{};
        copy.srcPtr=make_cudaPitchedPtr(const_cast<float*>(volume),size_t(g.nx)*sizeof(float),g.nx,g.ny);
        copy.dstArray=array;copy.extent=make_cudaExtent(g.nx,g.ny,g.nz);copy.kind=cudaMemcpyDeviceToDevice;
        checked(cudaMemcpy3D(&copy));
        cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
        cudaTextureDesc desc{};desc.filterMode=cudaFilterModeLinear;desc.readMode=cudaReadModeElementType;
        for(auto& mode:desc.addressMode)mode=cudaAddressModeBorder;
        checked(cudaCreateTextureObject(&id,&resource,&desc,nullptr));
        desc.filterMode=cudaFilterModePoint;
        checked(cudaCreateTextureObject(&point,&resource,&desc,nullptr));
    }
    ~Texture(){cudaDestroyTextureObject(point);cudaDestroyTextureObject(id);cudaFreeArray(array);}
};
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Usage: multispectrum_benchmark_scalar_fp <projection.toml>; RAW labels, flat circular/helical only");
    const auto c=loadConfig(argv[1]);const auto& g=c.geometry.parameters;
    if(c.geometry.kind!=GeometryKind::FlatCbct && c.geometry.kind!=GeometryKind::FlatHelical)
        throw std::runtime_error("Benchmark requires flat circular/helical geometry");
    const size_t voxels=size_t(g.volume_x)*g.volume_y*g.volume_z,pixels=size_t(g.detector_u)*g.detector_v;
    std::vector<unsigned char> labels(voxels);
    if(std::filesystem::file_size(c.projection.label_volume)!=voxels)throw std::runtime_error("Expected uint8 RAW labels matching grid");
    std::ifstream in(c.projection.label_volume,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(labels.data()),labels.size()))throw std::runtime_error("Cannot read labels");
    std::array<float,256> attenuation{};std::array<bool,256> known{};
    XcomAttenuationProvider provider({});std::string error;
    for(const auto& m:c.projection.materials){std::vector<double> values;if(!provider.query(m,{70.},values,error))throw std::runtime_error(error);
        attenuation[m.label]=float(values.at(0)*m.density_g_cm3*.1);known[m.label]=true;}
    std::vector<float> volume(voxels);
    for(size_t i=0;i<voxels;++i){if(!known[labels[i]])throw std::runtime_error("Unknown label");volume[i]=attenuation[labels[i]];}
    fp::Grid grid{g.volume_x,g.volume_y,g.volume_z,{float(g.voxel_x_mm),float(g.voxel_y_mm),float(g.voxel_z_mm)},{}};
    auto dg=primary_detail::makeDeviceGeometry(c,int(c.projection.materials.size()),1);
    std::vector<fp::View> views;
    for(int i=0;i<g.views;++i){
        const auto ray=primary_detail::scannerRay(dg,i,0,0);
        constexpr float pi=3.14159265358979323846f;
        const float angle=dg.start+float(dg.rotation_direction)*2.f*pi*i/dg.views_per_turn;
        float3 u,v,n;primary_detail::detectorFrame(dg,angle-.5f*pi,u,v,n);
        auto local=[&](float3 p,bool direction){return primary_detail::phantomInverse(dg,p,direction);};
        views.push_back({local(ray.source,false),local(ray.detector,false),
            local(primary_detail::scale3(u,dg.pu),true),local(primary_detail::scale3(v,dg.pv),true)});
    }
    // At most 64 MiB of projection workspace, regardless of total scan length.
    const int batch=std::min(g.views,std::max(1,int((64ULL*1024*1024)/(pixels*sizeof(float)))));
    Device<float> dv(voxels),out(pixels*batch);Device<fp::DeviceView> geometry(g.views);Events timer;
    const float upload=timer.time([&]{checked(cudaMemcpy(dv.p,volume.data(),volume.size()*sizeof(float),cudaMemcpyHostToDevice));});
    const auto before=std::chrono::steady_clock::now();Texture texture(dv.p,grid);
    const double textureMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    std::cout<<"volume="<<g.volume_x<<"x"<<g.volume_y<<"x"<<g.volume_z<<" detector="<<g.detector_u<<"x"<<g.detector_v<<" views="<<g.views<<" batch="<<batch<<" scalar_energy_keV=70 pixel_rays=1 jitter=0 upload_ms="<<upload<<" texture_setup_ms="<<textureMs<<std::endl;
    for(int variant=0;variant<3;++variant){
        const auto method=variant==2?fp::Method::Joseph:fp::Method::Siddon;
        const auto memory=variant==1?fp::SiddonMemory::Texture:fp::SiddonMemory::Linear;
        const auto prepared=fp::prepareViews(views,grid,g.detector_u,g.detector_v,method,fp::DetectorAnchor::Pixel00);
        checked(cudaMemcpy(geometry.p,prepared.data(),prepared.size()*sizeof(fp::DeviceView),cudaMemcpyHostToDevice));
        auto launch=[&](int first,int count){fp::launch(method,dv.p,variant==1?texture.point:texture.id,geometry.p+first,out.p,grid,g.detector_u,g.detector_v,count,false,nullptr,1,memory);};
        auto scan=[&]{for(int first=0;first<g.views;first+=batch)launch(first,std::min(batch,g.views-first));};
        scan();checked(cudaDeviceSynchronize());
        std::vector<float> times;for(int repeat=0;repeat<3;++repeat)times.push_back(timer.time(scan));
        std::sort(times.begin(),times.end());
        std::vector<float> host(pixels*batch),reference(variant==1?pixels*batch:0);double download=0,compute=0,sum=0;size_t count=0;float minimum=INFINITY,maximum=-INFINITY;
        for(int first=0;first<g.views;first+=batch){const int n=std::min(batch,g.views-first);
            compute+=timer.time([&]{launch(first,n);});
            download+=timer.time([&]{checked(cudaMemcpy(host.data(),out.p,size_t(n)*pixels*sizeof(float),cudaMemcpyDeviceToHost));});
            if(variant==1){
                fp::launch(method,dv.p,texture.point,geometry.p+first,out.p,grid,g.detector_u,g.detector_v,n);
                checked(cudaMemcpy(reference.data(),out.p,size_t(n)*pixels*sizeof(float),cudaMemcpyDeviceToHost));
                if(!std::equal(host.begin(),host.begin()+size_t(n)*pixels,reference.begin()))throw std::runtime_error("Full-volume Siddon texture/linear mismatch");
            }
            for(size_t i=0;i<size_t(n)*pixels;++i){float v=host[i];if(!std::isfinite(v)||v<0)throw std::runtime_error("Invalid scalar output");sum+=v;minimum=std::min(minimum,v);maximum=std::max(maximum,v);++count;}
        }
        std::cout<<"method="<<(variant==0?"siddon_linear":variant==1?"siddon_texture":"joseph")<<" kernel_median_ms="<<times[1]<<" kernel_runs_ms="<<times[0]<<","<<times[1]<<","<<times[2]<<" readback_scan_kernel_ms="<<compute<<" download_ms="<<download<<" min="<<minimum<<" max="<<maximum<<" mean="<<sum/count<<std::endl;
        if(variant==1)std::cout<<"siddon_texture_linear_exact_match=true elements="<<count<<std::endl;
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
