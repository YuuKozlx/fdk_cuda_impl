#include "LibraryGeometry.hpp"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace yk::spectral;
void ck(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Reader {std::vector<float> input;int views=13,calls=0,max_count=0;bool interleaved=false,fail=false,throws=false;
    static bool read(void* p,const int* ids,int count,float* output,size_t elements){auto& r=*static_cast<Reader*>(p);
        if(r.throws)throw std::runtime_error("injected read exception");if(r.fail)return false;
        ck(elements==size_t(count)*32*24,"reader shape");++r.calls;r.max_count=std::max(r.max_count,count);
        for(int i=0;i<count;++i){ck(ids[i]>=0&&ids[i]<r.views,"reader index");if(i&&ids[i]!=ids[i-1]+1)r.interleaved=true;
            std::memcpy(output+size_t(i)*32*24,r.input.data()+size_t(ids[i])*32*24,32*24*sizeof(float));}return true;
    }
};
std::vector<float> run(const std::string& algo,int source,int batch,int views=13,bool tv=false,bool helical=false){
    std::cerr<<"case "<<algo<<" source="<<source<<" batch="<<batch<<std::endl;
    SimulationConfig c;auto& g=c.geometry.parameters;g.views=g.views_per_turn=views;g.detector_u=32;g.detector_v=24;
    g.volume_x=g.volume_y=g.volume_z=16;g.reconstruction_volume_x=g.reconstruction_volume_y=g.reconstruction_volume_z=16;
    if(helical){c.geometry.kind=GeometryKind::FlatHelical;g.views_per_turn=6;g.pitch_mm_per_turn=8;g.start_z_mm=-8;}
    c.reconstruction.type="iterative";c.reconstruction.iterative.algorithm=algo;c.reconstruction.iterative.iterations=3;c.reconstruction.iterative.subsets=4;
    // This test verifies batching/source equivalence, not automatic projector
    // selection. Keep both sides on the historical operator pair.
    c.reconstruction.iterative.forward_projector="joseph";
    c.reconstruction.iterative.back_projector="joseph_v3";
    if(tv){auto& r=c.reconstruction.iterative.regularization;r.plugin="huber_tv_prox";r.strength=.002;r.huber_delta=.001;r.inner_iterations=7;}
    const auto pipeline=algo=="sart"?YK::EPipeline::SART:algo=="ossart"?YK::EPipeline::OSSART:algo=="sirt"?YK::EPipeline::SIRT:YK::EPipeline::TigreGradient;
    auto* session=YK::ReconstructionSessionFactory::create();ck(session&&session->initialize(makeLibrarySystem(c,pipeline,YK::EFdkFilter::RamLak,GeometryUse::Reconstruction)),"initialize");
    Reader reader;reader.views=views;reader.input.resize(size_t(views)*32*24);for(size_t i=0;i<reader.input.size();++i)reader.input[i]=.1f+.001f*float(i%29);
    std::vector<float> output(16*16*16),repeat(output.size());float *dp=nullptr,*dv=nullptr;
    ck(cudaMalloc(&dv,output.size()*4)==cudaSuccess,"volume allocation");
    YK::SExecutionRequest request;request.view_count=views;request.iteration_count=3;request.projection_batch_views=batch;request.projection_batch_mib=1;
    request.volume={dv,YK::EMemoryLocation::Device,output.size()};
    if(source==0){ck(cudaMalloc(&dp,reader.input.size()*4)==cudaSuccess,"resident allocation");cudaMemcpy(dp,reader.input.data(),reader.input.size()*4,cudaMemcpyHostToDevice);request.projection={dp,YK::EMemoryLocation::Device,reader.input.size()};}
    else if(source==1)request.projection={reader.input.data(),YK::EMemoryLocation::Host,reader.input.size()};
    else {request.projection_read=&Reader::read;request.projection_read_user_data=&reader;}
    int progress_calls=0;
    if(tv){request.progress_user_data=&progress_calls;
        request.progress_callback=[](void* p,int completed,int total){ck(total==3&&completed==++*static_cast<int*>(p),"TV sweep progress");};}
    cudaMemset(dv,0,output.size()*4);ck(session->execute(request),"execute");
    if(tv)ck(progress_calls==3,"TV missing sweep completion");
    request.progress_callback=nullptr;cudaMemcpy(output.data(),dv,output.size()*4,cudaMemcpyDeviceToHost);
    session->reset();cudaMemset(dv,0,output.size()*4);ck(session->execute(request),"repeat execute");cudaMemcpy(repeat.data(),dv,output.size()*4,cudaMemcpyDeviceToHost);
    for(size_t i=0;i<output.size();++i)ck(std::isfinite(output[i])&&std::abs(output[i]-repeat[i])<1e-6f,"reset mismatch");
    if(source==2){ck(reader.calls>0&&reader.max_count<views,"reader loaded complete sequence");if(algo=="ossart")ck(reader.interleaved,"OS-SART subset membership changed");
        reader.fail=true;session->reset();ck(!session->execute(request)&&session->lastError()==YK::EApiErrorCode::InvalidBuffer,"reader failure ignored");
        reader.fail=false;reader.throws=true;session->reset();ck(!session->execute(request),"reader exception ignored");reader.throws=false;
        session->reset();cudaMemset(dv,0,output.size()*4);ck(session->execute(request),"reader retry failed");
        request.struct_size=offsetof(YK::SExecutionRequest,projection_read);session->reset();ck(!session->execute(request),"old ABI read new callback tail");
        request.struct_size=offsetof(YK::SExecutionRequest,projection_read_user_data);ck(!session->execute(request)&&session->lastError()==YK::EApiErrorCode::InvalidConfig,"truncated source tail accepted");
    }
    if(dp)cudaFree(dp);cudaFree(dv);YK::ReconstructionSessionFactory::destroy(session);return output;
}

void helicalSphere(bool helical=true){
    SimulationConfig c;c.geometry.kind=helical?GeometryKind::FlatHelical:GeometryKind::FlatCbct;auto& g=c.geometry.parameters;
    g.views=96;g.views_per_turn=24;g.detector_u=64;g.detector_v=16;g.pixel_u_mm=g.pixel_v_mm=1;
    g.sid_mm=500;g.sdd_mm=1000;g.pitch_mm_per_turn=helical?8:0;g.start_z_mm=helical?16:0;if(!helical)g.views_per_turn=g.views;
    g.volume_x=g.volume_y=g.volume_z=32;g.reconstruction_volume_x=g.reconstruction_volume_y=g.reconstruction_volume_z=32;
    g.voxel_x_mm=g.voxel_y_mm=g.voxel_z_mm=g.reconstruction_voxel_x_mm=g.reconstruction_voxel_y_mm=g.reconstruction_voxel_z_mm=1;
    c.reconstruction.type="iterative";c.reconstruction.iterative.algorithm="sart";c.reconstruction.iterative.iterations=3;
    c.reconstruction.iterative.forward_projector="joseph";
    c.reconstruction.iterative.back_projector="joseph_v3";
    constexpr float mu=.02f;std::vector<float> truth(32*32*32),output(truth.size());
    for(int z=0;z<32;++z)for(int y=0;y<32;++y)for(int x=0;x<32;++x){float xx=x-15.5f,yy=y-15.5f,zz=z-15.5f;truth[(z*32+y)*32+x]=xx*xx+yy*yy+zz*zz<25?mu:0;}
    float *projection=nullptr,*volume=nullptr;ck(cudaMalloc(&projection,size_t(96*64*16)*4)==cudaSuccess&&cudaMalloc(&volume,truth.size()*4)==cudaSuccess,"sphere buffers");
    auto* session=YK::ReconstructionSessionFactory::create();
    auto system=makeLibrarySystem(c,YK::EPipeline::ForwardProjection,YK::EFdkFilter::RamLak,GeometryUse::Reconstruction);
    system.reconstruction.forward_projector=YK::EForwardProjector::Joseph;
    ck(session&&session->initialize(system),"sphere FP initialize");cudaMemcpy(volume,truth.data(),truth.size()*4,cudaMemcpyHostToDevice);
    YK::SExecutionRequest request;request.view_count=96;request.projection={projection,YK::EMemoryLocation::Device,size_t(96*64*16)};request.volume={volume,YK::EMemoryLocation::Device,truth.size()};
    ck(session->execute(request),"sphere FP");
    system=makeLibrarySystem(c,YK::EPipeline::SART,YK::EFdkFilter::RamLak,GeometryUse::Reconstruction);
    ck(session->initialize(system),"sphere SART initialize");cudaMemset(volume,0,truth.size()*4);ck(session->execute(request),"sphere SART");cudaMemcpy(output.data(),volume,output.size()*4,cudaMemcpyDeviceToHost);
    double sum=0,norm=0,error=0,peak=0;int count=0;
    for(int z=0;z<32;++z)for(int y=0;y<32;++y)for(int x=0;x<32;++x){size_t i=(z*32+y)*32+x;ck(std::isfinite(output[i]),"sphere nonfinite");double xx=x-15.5,yy=y-15.5,zz=z-15.5;if(xx*xx+yy*yy+zz*zz<4){sum+=output[i];++count;}double d=output[i]-truth[i];error+=d*d;norm+=truth[i]*truth[i];peak=std::max(peak,double(output[i]));}
    std::cerr<<(helical?"helical":"narrow circular")<<" sphere mean="<<sum/count<<" expected="<<mu<<" relative_error="<<std::sqrt(error/norm)<<" peak="<<peak<<'\n';
    ck(sum/count>mu*.65&&sum/count<mu*1.35&&std::sqrt(error/norm)<.8&&peak<mu*2.5,"helical SART normalization distorts known sphere");
    cudaFree(projection);cudaFree(volume);YK::ReconstructionSessionFactory::destroy(session);
}
int main(int argc,char** argv){try{if(argc==2&&std::string(argv[1])=="--helical-sphere"){helicalSphere();return 0;}if(argc==2&&std::string(argv[1])=="--circular-sphere"){helicalSphere(false);return 0;}for(const char* algo:{"sart","ossart","sirt","tigre_sart","tigre_os_sart","tigre_sirt","tigre_sart_tv","tigre_os_sart_tv"}){
    const auto reference=run(algo,0,5);
    if(argc==3&&std::string(argv[1])=="--baseline"){
        std::filesystem::create_directories(argv[2]);std::ofstream out(std::filesystem::path(argv[2])/(std::string(algo)+".raw"),std::ios::binary);
        out.write(reinterpret_cast<const char*>(reference.data()),reference.size()*sizeof(float));continue;
    }
    for(int source:{1,2})for(int batch:{1,5,8}){const auto output=run(algo,source,batch);
        for(size_t i=0;i<output.size();++i)ck(std::abs(reference[i]-output[i])<1e-6f,"batched changed reconstruction");}
    std::cout<<algo<<" resident/Host/reader batch equivalence passed\n";
}
    if(argc!=3)for(const char* algo:{"sirt","tigre_sirt"}){
        const auto reference=run(algo,0,64,67);
        for(int batch:{5,64}){const auto output=run(algo,2,batch,67);
            for(size_t i=0;i<output.size();++i)ck(std::abs(reference[i]-output[i])<1e-6f,"SIRT tail chunk mismatch");}
    }
    if(argc!=3)for(const char* algo:{"sart","sirt","ossart"}){
        const auto plain=run(algo,0,5),reference=run(algo,0,5,13,true);
        float difference=0;for(size_t i=0;i<plain.size();++i)difference=std::max(difference,std::abs(plain[i]-reference[i]));
        ck(difference>1e-6f,"TV plugin did not affect volume");
        for(int source:{1,2}){const auto output=run(algo,source,5,13,true);
            for(size_t i=0;i<output.size();++i)ck(std::abs(reference[i]-output[i])<1e-6f,"batched TV mismatch");}
    }
    if(argc!=3)for(const char* algo:{"sart","sirt","ossart","tigre_sart","tigre_os_sart","tigre_sirt","tigre_sart_tv","tigre_os_sart_tv"}){
        auto reference=run(algo,0,5,13,false,true),observed=run(algo,2,5,13,false,true);
        for(size_t i=0;i<reference.size();++i)ck(std::isfinite(reference[i])&&std::abs(reference[i]-observed[i])<1e-6,"helical batch mismatch/nonfinite");
    }
    if(argc!=3){helicalSphere();helicalSphere(false);}
    return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
