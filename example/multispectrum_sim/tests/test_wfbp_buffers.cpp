#include "LibraryGeometry.hpp"
#include "global/YkMem3d.hpp"
#include <algorithm>
#include <cmath>
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

int main() {
    try {
        yk::spectral::SimulationConfig config;
        config.geometry = yk::spectral::GeometryKind::CylHelical;
        auto& g = config.geometry_config;
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
        YK::SExecutionRequest partial;
        partial.projection={input.data(),YK::EMemoryLocation::Host,np};
        partial.volume={output.data(),YK::EMemoryLocation::Host,nv};
        partial.view_count=g.views/2;
        require(!session->execute(partial) && session->lastError()==YK::EApiErrorCode::StreamingStateError,
            "partial wFBP input must be rejected");
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n'; return 1;
    }
}
