// DLL smoke test for the Session API.  It uses a tiny synthetic acquisition,
// so it runs without external raw-data files.
#include <cstdio>
#include <cuda_runtime_api.h>
#include "YKCBCT/interface/IYkSession.hpp"

namespace {
constexpr int kNu = 16, kNv = 12, kNx = 16, kNy = 16, kNz = 12, kAngles = 2;

YK::SessionDesc makeDesc(YK::EPipeline pipeline) {
    YK::SessionDesc d{};
    d.scan.Nu = kNu; d.scan.Nv = kNv; d.scan.NAng = kAngles;
    d.scan.du_mm = d.scan.dv_mm = 1.f; d.scan.SOD_mm = 500.f; d.scan.SDD_mm = 1000.f;
    d.scan.scanRangeRad = 6.283185307f;
    d.volume.Nx = kNx; d.volume.Ny = kNy; d.volume.Nz = kNz;
    d.volume.voxX_mm = d.volume.voxY_mm = d.volume.voxZ_mm = 1.f;
    d.algorithm.pipeline = pipeline;
    d.algorithm.forward_projector = YK::ETask::FP_Joseph;
    d.algorithm.back_projector = YK::ETask::BP_Joseph_v2;
    d.angles = { 0.f, 3.1415926535f };
    return d;
}

bool cudaOk(cudaError_t e, const char* where) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
    return false;
}

bool testForwardProjection() {
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    const auto desc = makeDesc(YK::EPipeline::ForwardProjection);
    if (!session->initialize(desc)) { YK::SessionFactory::destroy(session); return false; }
    const size_t volN = size_t(kNx) * kNy * kNz;
    const size_t sinoN = size_t(kAngles) * kNu * kNv;
    float *dVol = nullptr, *dSino = nullptr;
    bool ok = cudaOk(cudaMalloc(reinterpret_cast<void**>(&dVol), volN * sizeof(float)), "cudaMalloc(volume)") &&
              cudaOk(cudaMalloc(reinterpret_cast<void**>(&dSino), sinoN * sizeof(float)), "cudaMalloc(sinogram)") &&
              cudaOk(cudaMemset(dVol, 0, volN * sizeof(float)), "cudaMemset(volume)");
    if (ok) {
        YK::ExecuteRequest r{};
        r.angles = desc.angles.data(); r.K = kAngles;
        r.volume = { dVol, YK::EMemoryLocation::Device };
        r.projection = { dSino, YK::EMemoryLocation::Device };
        ok = session->execute(r) && cudaOk(cudaDeviceSynchronize(), "FP synchronize");
    }
    cudaFree(dSino); cudaFree(dVol);
    session->release(); YK::SessionFactory::destroy(session);
    return ok;
}

}

int main() {
    const bool fpOk = testForwardProjection();
    std::printf("Session DLL smoke test: FP=%s\n", fpOk ? "OK" : "FAILED");
    return fpOk ? 0 : 1;
}
