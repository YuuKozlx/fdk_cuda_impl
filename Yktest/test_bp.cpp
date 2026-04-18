#include "test_common.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>

#include "BP/YkBPRunner.hpp"
#include "FDK/YkFdkReconstructor.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"

using namespace YK;
using namespace Mem;

const std::string test_data_dir = R"(G:\Code\fanproj\fdk-test\TestData\)";

// ----------------------------------------------------------------
// main_bp_runner：离线 + 在线反投影
// ----------------------------------------------------------------
int main_bp_runner()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = false;
    params.scan_range_rad = 2.0f * (float)CUDA_PI;
    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f; params.vox_y_mm = 0.1f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)CUDA_PI / params.iPAng;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    const int    Ang = params.iPAng;
    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> h_flt(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_flt)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    MemoryController ctrl;

    auto d_flt_buf = ctrl.allocateDevice3D<float>(
        (size_t)params.iPU * params.iPV, Ang, 1, 0);
    {
        auto h_view = ctrl.allocateCpu3D<float>(
            (size_t)params.iPU * params.iPV, Ang, 1, false);
        std::memcpy(h_view.data(), h_flt.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_flt_buf, h_view);
    }

    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ---- 离线反投影 ----
    {
        BpReconstructor recon;
        recon.init(params, /*Kchunk=*/32, s);
        YK::Util::CudaTimer timer("bp_offline", s);
        recon.feed(d_flt_buf.data(), params, s, d_vol_buf.data(), true);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "bp_vol_offline.raw").c_str(), h_vol.cdata(), vol_elems);
        YK_LOGI("saved: bp_vol_offline.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- 在线分包反投影 ----
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    BpReconstructor recon_online;
    recon_online.init(params, /*Kchunk=*/32, s);
    {
        YK::Util::CudaTimer timer("bp_online", s);
        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams bp = params;
            bp.iPAng = count;
            bp.angle_list = std::vector<float>(
                angle_list.begin() + base, angle_list.begin() + base + count);

            recon_online.feed(d_flt_buf.data() + base * view_elems,
                bp, s, d_vol_buf.data(), (i == 0));
        }
    }
    recon_online.reset();
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "bp_vol_online.raw").c_str(), h_vol.cdata(), vol_elems);
        YK_LOGI("saved: bp_vol_online.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

// ----------------------------------------------------------------
// main_bp_verify：验证 BpReconstructor 与 FDK 反投影阶段一致性
// ----------------------------------------------------------------
int main_bp_verify()
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
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_proj)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }
    YK_LOGI("proj loaded");

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_vol_fdk = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_vol_bp = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    std::vector<float> h_flt_all(proj_elems, 0.f);

    // ---- 路径一：FDK + dump flt ----
    {
        FdkReconstructor recon;
        recon.init(params, 32, s);

        struct DumpCtx { cudaStream_t stream; float* buf; size_t ve; };
        DumpCtx ctx{ s, h_flt_all.data(), view_elems };

        auto onDump = [](void* p) {
            auto* payload = static_cast<DumpPayload*>(p);
            if (std::string(payload->stage) != "flt") return;
            auto* ctx = static_cast<DumpCtx*>(payload->userdata);
            cudaMemcpyAsync(
                ctx->buf + (size_t)payload->viewIdx * ctx->ve,
                payload->buf, payload->n * sizeof(float),
                cudaMemcpyDeviceToHost, ctx->stream);
            };

        recon.feed(h_proj.data(), params, s,
            d_vol_fdk.data(), true, onDump, &ctx);
    }

    cudaStreamSynchronize(s);
    {
        auto h_vol_fdk = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_fdk, d_vol_fdk);
        write_raw_float((test_data_dir + "verify_vol_fdk.raw").c_str(), h_vol_fdk.cdata(), vol_elems);
        YK_LOGI("saved: verify_vol_fdk.raw");

        float maxv = *std::max_element(h_flt_all.begin(), h_flt_all.end());
        float minv = *std::min_element(h_flt_all.begin(), h_flt_all.end());
        YK_LOGI("h_flt_all stats: min={:.6f} max={:.6f}", minv, maxv);
    }

    // ---- 路径二：BpReconstructor ----
    float* d_flt_raw = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice));
    {
        BpReconstructor recon;
        recon.init(params, 32, s);
        recon.feed(d_flt_raw, params, s, d_vol_bp.data(), true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_vol_bp = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_bp, d_vol_bp);
        write_raw_float((test_data_dir + "verify_vol_bp.raw").c_str(), h_vol_bp.cdata(), vol_elems);
        YK_LOGI("saved: verify_vol_bp.raw");
    }
    cudaFree(d_flt_raw);

    // ---- 对比 ----
    {
        auto h_vol_fdk = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_vol_bp = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_fdk, d_vol_fdk);
        ctrl.download3D(h_vol_bp, d_vol_bp);

        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = std::abs(
                (double)h_vol_fdk.cdata()[i] - (double)h_vol_bp.cdata()[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_elems;

        YK_LOGI("maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
        if (maxDiff < 1e-5)
            YK_LOGI("PASS: BpReconstructor matches FdkReconstructor");
        else
            YK_LOGE("FAIL: maxDiff={:.8f} exceeds threshold", maxDiff);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
