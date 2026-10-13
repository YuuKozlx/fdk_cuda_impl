#include "LibraryGeometry.hpp"
#include "global/YkMem3d.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void cudaCheck(cudaError_t result) {
    require(result == cudaSuccess, cudaGetErrorString(result));
}
}

void checkBuffers(yk::spectral::GeometryKind kind) {
    try {
        yk::spectral::SimulationConfig config;
        config.geometry.kind = kind;
        auto& g = config.geometry.parameters;
        g.views = 360; g.views_per_turn = 120;
        g.detector_u = 64; g.detector_v = 16;
        g.pixel_u_mm = g.pixel_v_mm = 1;
        g.sid_mm = 500; g.sdd_mm = 1000;
        g.pitch_mm_per_turn = 4; g.start_z_mm = -6;
        g.volume_x = g.volume_y = 16; g.volume_z = 8;
        auto system = yk::spectral::makeLibrarySystem(config, YK::EPipeline::WFBP);
        using Session = std::unique_ptr<YK::IReconstructionSession,
            decltype(&YK::ReconstructionSessionFactory::destroy)>;
        Session session(YK::ReconstructionSessionFactory::create(),
            &YK::ReconstructionSessionFactory::destroy);
        require(session && session->initialize(system), "wFBP initialize failed");
        const size_t np = size_t(g.views)*g.detector_u*g.detector_v;
        const size_t nv = size_t(g.volume_x)*g.volume_y*g.volume_z;
        std::vector<float> input(np), output(nv), reference;
        // 非常量平滑投影，避免全零输出也能通过内存组合对比。
        for (size_t i = 0; i < np; ++i) {
            const double u = (int(i % g.detector_u)-31.5)/14.;
            input[i] = float(std::exp(-u*u)*(1+0.1*std::sin(double(i/(g.detector_u*g.detector_v))*0.05)));
        }
        // 只复用内部内存工具；被测算法入口仍为公共 DLL Session。
        YK::Mem::MemoryController memory;
        auto dp = memory.allocateDevice3D<float>(g.detector_u,g.detector_v,g.views,0,false);
        auto dv = memory.allocateDevice3D<float>(g.volume_x,g.volume_y,g.volume_z,0,false);
        cudaCheck(cudaMemcpy(dp.data(),input.data(),np*sizeof(float),cudaMemcpyHostToDevice));
        for (int round = 0; round < 3; ++round) {
            if (round == 1) {
#ifdef _WIN32
                _putenv_s("YK_WFBP_BATCHED", "1");
#else
                setenv("YK_WFBP_BATCHED", "1", 1);
#endif
                const bool initialized = session->initialize(system);
#ifdef _WIN32
                _putenv_s("YK_WFBP_BATCHED", "");
#else
                unsetenv("YK_WFBP_BATCHED");
#endif
                require(initialized, "phase-batched wFBP initialize failed");
            }
            if (round == 2) session->reset();
            for (int combination = 0; combination < 4; ++combination) {
                const bool device_input = combination & 1, device_output = combination & 2;
                std::fill(output.begin(),output.end(),-99.f);
                cudaCheck(cudaMemset(dv.data(),0xff,nv*sizeof(float)));
                YK::SExecutionRequest request;
                request.projection = {device_input ? dp.data() : input.data(),
                    device_input ? YK::EMemoryLocation::Device : YK::EMemoryLocation::Host,np};
                request.volume = {device_output ? dv.data() : output.data(),
                    device_output ? YK::EMemoryLocation::Device : YK::EMemoryLocation::Host,nv};
                request.view_count = g.views;
                require(session->execute(request),session->lastErrorMessage());
                if (device_output) cudaCheck(cudaMemcpy(output.data(),dv.data(),nv*sizeof(float),cudaMemcpyDeviceToHost));
                require(std::all_of(output.begin(),output.end(),[](float x){return std::isfinite(x);}),"nonfinite output");
                if (reference.empty()) reference = output;
                double max_diff = 0, peak = 0;
                for (size_t i=0;i<nv;++i) {
                    max_diff=std::max(max_diff,std::abs(double(output[i])-reference[i]));
                    peak=std::max(peak,std::abs(double(reference[i])));
                }
                require(peak>1e-8 && max_diff<=1e-6+peak*1e-5,"buffer/repeat/reset mismatch");
                std::cout << "round="<<round<<" combination="<<combination<<" max_diff="<<max_diff<<'\n';
            }
        }
        // Public enum selection must affect WFBP, and common gain must have
        // the same linear meaning as in FDK. No TOML coefficient aliases.
        std::vector<float> hann;
        for (auto filter : { YK::EFdkFilter::RamLak, YK::EFdkFilter::SheppLogan,
                YK::EFdkFilter::Cosine, YK::EFdkFilter::Hann, YK::EFdkFilter::Hamming,
                YK::EFdkFilter::Blackman, YK::EFdkFilter::Butterworth,
                YK::EFdkFilter::Kaiser, YK::EFdkFilter::Tukey }) {
            auto selected=system;
            selected.reconstruction.wfbp.filter.type=filter;
            require(session->initialize(selected),"enum WFBP filter initialize failed");
            YK::SExecutionRequest r;
            r.projection={input.data(),YK::EMemoryLocation::Host,np};
            r.volume={output.data(),YK::EMemoryLocation::Host,nv};r.view_count=g.views;
            require(session->execute(r),session->lastErrorMessage());
            require(std::all_of(output.begin(),output.end(),[](float x){return std::isfinite(x);}),"nonfinite enum filter output");
            if(filter==YK::EFdkFilter::Hann) hann=output;
        }
        require(!hann.empty(),"Hann result missing");
        double change=0;
        for(size_t i=0;i<nv;++i)change=std::max(change,std::abs(double(hann[i])-reference[i]));
        require(change>1e-7,"window enum was ignored");
        auto doubled=system; doubled.reconstruction.wfbp.filter.type=YK::EFdkFilter::Hann;
        doubled.reconstruction.wfbp.filter.gain=2;
        require(session->initialize(doubled),"shared filter gain initialize failed");
        YK::SExecutionRequest gain_request;
        gain_request.projection={input.data(),YK::EMemoryLocation::Host,np};
        gain_request.volume={output.data(),YK::EMemoryLocation::Host,nv}; gain_request.view_count=g.views;
        require(session->execute(gain_request),session->lastErrorMessage());
        double gain_error=0,peak=0;
        for(size_t i=0;i<nv;++i){gain_error=std::max(gain_error,std::abs(double(output[i])-2*hann[i]));peak=std::max(peak,std::abs(double(hann[i])));}
        require(gain_error<=1e-6+peak*1e-5,"common gain semantics mismatch");
        YK::SExecutionRequest partial;
        partial.projection={input.data(),YK::EMemoryLocation::Host,np};
        partial.volume={output.data(),YK::EMemoryLocation::Host,nv};
        partial.view_count=g.views/2;
        require(!session->execute(partial) && session->lastError()==YK::EApiErrorCode::StreamingStateError,
            "partial wFBP input must be rejected");
        if(kind==yk::spectral::GeometryKind::FlatHelical){
            for(int axis=0;axis<3;++axis){auto invalid=system;float* offset=&invalid.geometry.flat_detector.pose.offset_unv_mm.x;offset[axis]=1;require(!session->initialize(invalid)&&session->lastError()==YK::EApiErrorCode::UnsupportedCombination,"flat WFBP offset must be rejected");}
            auto tilted=system;tilted.geometry.flat_detector.pose.tilt_u_rad=.01f;require(!session->initialize(tilted)&&session->lastError()==YK::EApiErrorCode::UnsupportedCombination,"flat WFBP tilt must be rejected");
        }
        return;
    } catch(const std::exception& e) {
        throw;
    }
}

void checkWaterCylinder(){
    yk::spectral::SimulationConfig config;config.geometry.kind=yk::spectral::GeometryKind::FlatHelical;
    auto& g=config.geometry.parameters;g.views=360;g.views_per_turn=120;g.detector_u=64;g.detector_v=16;
    g.pixel_u_mm=g.pixel_v_mm=1;g.sid_mm=500;g.sdd_mm=1000;g.pitch_mm_per_turn=4;g.start_z_mm=-6;
    g.volume_x=g.volume_y=16;g.volume_z=8;g.voxel_x_mm=g.voxel_y_mm=g.voxel_z_mm=.6;
    auto system=yk::spectral::makeLibrarySystem(config,YK::EPipeline::WFBP);
    auto* session=YK::ReconstructionSessionFactory::create();require(session&&session->initialize(system),"flat water init");
    const double radius=4,mu=.02;std::vector<float> input(size_t(g.views)*g.detector_u*g.detector_v),output(16*16*8);
    for(int view=0;view<g.views;++view)for(int row=0;row<g.detector_v;++row)for(int channel=0;channel<g.detector_u;++channel){
        double u=(channel-.5*(g.detector_u-1))*g.pixel_u_mm,v=(row-.5*(g.detector_v-1))*g.pixel_v_mm;
        double ray_xy=std::hypot(g.sdd_mm,u),impact=g.sid_mm*u/ray_xy;
        input[(size_t(view)*g.detector_v+row)*g.detector_u+channel]=float(std::abs(impact)<radius?mu*2*std::sqrt(radius*radius-impact*impact)*std::sqrt(ray_xy*ray_xy+v*v)/ray_xy:0);
    }
    YK::SExecutionRequest request;request.projection={input.data(),YK::EMemoryLocation::Host,input.size()};request.volume={output.data(),YK::EMemoryLocation::Host,output.size()};request.view_count=g.views;
    require(session->execute(request),session->lastErrorMessage());
    double sum=0;int count=0;for(int z=2;z<6;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
        double xx=(x-7.5)*.6,yy=(y-7.5)*.6;if(xx*xx+yy*yy<1.2*1.2){sum+=output[(z*16+y)*16+x];++count;}
    }
    std::cout<<"flat water mu="<<sum/count<<" expected="<<mu<<'\n';
    require(std::isfinite(sum)&&std::abs(sum/count-mu)<mu*.15,"flat helical water attenuation scale mismatch");
    YK::ReconstructionSessionFactory::destroy(session);
}
int main(){try{checkBuffers(yk::spectral::GeometryKind::CylHelical);checkBuffers(yk::spectral::GeometryKind::FlatHelical);checkWaterCylinder();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
