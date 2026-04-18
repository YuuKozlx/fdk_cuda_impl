#include "test_common.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <fstream>

#include "FDK/YkFdkReconstructor.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"

using namespace YK;
using namespace Mem;

int main_fdk()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 4.0f / 3.0f;
    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f; params.vox_y_mm = 0.1f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    const int    Ang = params.iPAng;
    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float("proj_1024x1024x360.raw", h_proj)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    auto dump = [](void* p) {
        auto* payload = static_cast<DumpPayload*>(p);
        if (payload->viewIdx != 0) return;
        float* d_buf = static_cast<float*>(payload->buf);
        const size_t n = payload->n;

        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.f, maxv = -1e30f, minv = 1e30f;
        for (auto v : h) { sum += v; maxv = std::max(maxv, v); minv = std::min(minv, v); }
        YK_LOGI("[dump][a={}][{}] n={} min={:.4f} max={:.4f} mean={:.6f}",
            payload->viewIdx, payload->stage, n, minv, maxv, sum / (float)n);

        auto path = fmt::format("dump_a{}_{}.raw", payload->viewIdx, payload->stage);
        std::ofstream f(path, std::ios::binary);
        if (f) f.write(reinterpret_cast<const char*>(h.data()), n * sizeof(float));
        else   YK_LOGE("[dump] cannot save {}", path);
        };

    MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ---- 离线重建 ----
    {
        YK::Util::CudaTimer timer("offline", s);
        YK::fdk_recon(h_proj.data(), d_vol_buf.data(), params,
            /*Kchunk=*/32, s, /*clear_vol=*/true, dump, nullptr);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float("fdk_vec_vol_offline.raw", h_vol.cdata(), vol_elems);
        YK_LOGI("saved: fdk_vec_vol_offline.raw");
    }

    // ---- 在线重建 ----
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    FdkReconstructor recon;
    recon.init(params, /*Kchunk=*/32, s);
    {
        YK::Util::CudaTimer timer("online", s);
        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams bp = params;
            bp.iPAng = count;
            bp.angle_list = std::vector<float>(
                angle_list.begin() + base, angle_list.begin() + base + count);

            recon.feed(h_proj.data() + base * view_elems, bp, s,
                d_vol_buf.data(), (i == 0));
        }
    }
    recon.reset();
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float("fdk_vec_vol_online.raw", h_vol.cdata(), vol_elems);
        YK_LOGI("saved: fdk_vec_vol_online.raw");
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: fdk offline + online (%d x %d x %d)\n", Nx, Ny, Nz);
    return 0;
}
