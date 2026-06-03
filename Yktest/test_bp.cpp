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
#include <BP/BpZSlabRunner.hpp>
#include <BP/YkSiddonBPRunner.hpp>
#include <BP/YkSiddonBpZSlabRunner.hpp>

using namespace YK;
using namespace Mem;

const std::string test_data_dir = R"(H:\Code\fanproj\fdk-test\TestData\)";

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


int main_fdkbp_vs_onlybp_verify()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = 4.0f / 3.f * (float)CUDA_PI;
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

    // ---- 读投影 ----
    std::vector<float> h_proj(proj_elems);
    std::vector<float> h_flt(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_proj)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    // 上传投影到 GPU
    auto d_flt_buf = ctrl.allocateDevice3D<float>(
        (size_t)params.iPU * params.iPV, Ang, 1, 0);

    {

        auto d_vol_fdk = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
        FdkReconstructor recon;
        recon.init(params, 32, s);

        struct DumpCtx { cudaStream_t stream; float* buf; size_t ve; };
        DumpCtx ctx{ s, d_flt_buf.data(), view_elems };

        auto onDump = [](void* p) {
            auto* payload = static_cast<DumpPayload*>(p);
            if (std::string(payload->stage) != "flt") return;
            auto* ctx = static_cast<DumpCtx*>(payload->userdata);
            cudaMemcpyAsync(
                ctx->buf + (size_t)payload->viewIdx * ctx->ve,
                payload->buf, payload->n * sizeof(float),
                cudaMemcpyDeviceToDevice, ctx->stream);
            };

        recon.feed(h_proj.data(), params, s,
            d_vol_fdk.data(), true, onDump, &ctx);
    }


    auto d_vol_bp = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_vol_joseph = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ---- 路径一：FDK BpReconstructor（基准）----
    {
        BpReconstructor recon;
        recon.init(params, 32, s);
        YK::Util::CudaTimer timer("bp_fdk", s);
        recon.feed(d_flt_buf.data(), params, s, d_vol_bp.data(), true);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_bp);
        write_raw_float((test_data_dir + "bp_vol_fdk.raw").c_str(),
            h_vol.cdata(), vol_elems);
        float vmin = *std::min_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        float vmax = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        YK_LOGI("saved: bp_vol_fdk.raw  min={:.4e} max={:.4e}", vmin, vmax);
    }

    // ---- 路径二：ConeBackprojector Joseph v2 ----
    auto h_joseph = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false); // ← 提到外面

    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_FDK);
        YK::Util::CudaTimer timer("BP_FDK", s);
        bp.run(d_flt_buf.data(), params, d_vol_joseph.data(), s, true);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));
    }
    {
        ctrl.download3D(h_joseph, d_vol_joseph);

        // dtheta * fScaleDTheta
        const float dtheta = 2.0f * (float)CUDA_PI / 720.0f;
        const float fScaleDTheta = 2.0f * (float)CUDA_PI / params.scan_range_rad;
        float compensate = dtheta * fScaleDTheta;
        compensate = 1;

        std::for_each(h_joseph.data(), h_joseph.data() + vol_elems,
            [compensate](float& v) { v *= compensate; });

        write_raw_float((test_data_dir + "bp_vol_joseph_v2.raw").c_str(),
            h_joseph.cdata(), vol_elems);
        float vmin = *std::min_element(h_joseph.cdata(), h_joseph.cdata() + vol_elems);
        float vmax = *std::max_element(h_joseph.cdata(), h_joseph.cdata() + vol_elems);
        YK_LOGI("saved: bp_vol_joseph_v2.raw  min={:.4e} max={:.4e}", vmin, vmax);
    }

    // ---- 对比（用补偿后的 h_joseph）----
    {
        auto h_fdk = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_fdk, d_vol_bp);
        // h_joseph 已经是补偿后的值，直接用，不再重新 download

        double mse = 0.0, maxdiff = 0.0;
        double sum_fdk = 0.0, sum_joseph = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            const double a = h_fdk.cdata()[i];
            const double b = h_joseph.cdata()[i];   // ← 补偿后
            const double d = std::abs(a - b);
            mse += d * d;
            maxdiff = std::max(maxdiff, d);
            sum_fdk += a;
            sum_joseph += b;
        }
        mse /= (double)vol_elems;

        YK_LOGI("[compare] FDK sum={:.4e}  Joseph sum={:.4e}  ratio={:.6f}",
            sum_fdk, sum_joseph, sum_fdk / (sum_joseph + 1e-30));
        YK_LOGI("[compare] maxDiff={:.6e}  MSE={:.6e}", maxdiff, mse);

        if (maxdiff < 1e-4)
            YK_LOGI("PASS: Joseph v2 matches FDK BP");
        else
            YK_LOGW("DIFF: maxDiff={:.6e} — check denom_c Z-component difference", maxdiff);
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
    auto h_flt_all = ctrl.allocatePinnedCpu3D<float>(proj_elems, 1, 1);

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

        float maxv = *std::max_element(h_flt_all.cdata(), h_flt_all.cdata() + proj_elems);
        float minv = *std::min_element(h_flt_all.cdata(), h_flt_all.cdata() + proj_elems);
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


int main_bp_zslab_verify()
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
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;

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

    // ---- 路径一：BpReconstructor（参考） ----
    // 先跑一次 FDK dump 出滤波投影，再用 BpReconstructor 反投影
    auto d_vol_bp_ref = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    std::vector<float> h_flt_all(proj_elems, 0.f);

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
            ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false).data(),
            true, onDump, &ctx);
    }
    cudaStreamSynchronize(s);

    // 上传滤波投影到设备端
    float* d_flt_raw = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice));

    {
        YK::Util::CudaTimer timer("bp_ref", s);
        BpReconstructor recon;
        recon.init(params, 128, s);
        recon.feed(d_flt_raw, params, s, d_vol_bp_ref.data(), true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_bp_ref);
        write_raw_float((test_data_dir + "verify_bp_ref.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: verify_bp_ref.raw");
    }

    // ---- 路径二：BpZSlabReconstructor（设备端输入）----
    const int z_block_size = 100;  // 400 / 100 = 4 slabs
    std::vector<float> h_vol_zslab(vol_elems, 0.f);

    {
        YK::Util::CudaTimer timer("bp_zslab_device", s);
        BpZSlabReconstructor recon;
        if (!recon.init(params, 32, s, z_block_size)) {
            YK_LOGE("BpZSlabReconstructor init failed");
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        if (!recon.feed(d_flt_raw, params, s, h_vol_zslab.data())) {
            YK_LOGE("BpZSlabReconstructor feed failed");
            recon.release();
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        recon.release();
    }
    write_raw_float((test_data_dir + "verify_bp_zslab_device.raw").c_str(),
        h_vol_zslab.data(), vol_elems);
    YK_LOGI("saved: verify_bp_zslab_device.raw");

    // ---- 路径三：BpZSlabReconstructor（主机端输入）----
    std::vector<float> h_vol_zslab_host(vol_elems, 0.f);

    {
        YK::Util::CudaTimer timer("bp_zslab_host", s);
        BpZSlabReconstructor recon;
        if (!recon.init(params, 32, s, z_block_size)) {
            YK_LOGE("BpZSlabReconstructor init failed");
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        if (!recon.feedFromHost(h_flt_all.data(), params, s, h_vol_zslab_host.data())) {
            YK_LOGE("BpZSlabReconstructor feedFromHost failed");
            recon.release();
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        recon.release();
    }
    write_raw_float((test_data_dir + "verify_bp_zslab_host.raw").c_str(),
        h_vol_zslab_host.data(), vol_elems);
    YK_LOGI("saved: verify_bp_zslab_host.raw");

    cudaFree(d_flt_raw);

    // ---- 对比：zslab_device vs bp_ref ----
    {
        auto h_vol_ref = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_ref, d_vol_bp_ref);

        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = std::abs(
                (double)h_vol_ref.cdata()[i] - (double)h_vol_zslab[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_elems;
        YK_LOGI("[zslab_device vs bp_ref] maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
        if (maxDiff < 1e-4)
            YK_LOGI("PASS: BpZSlabReconstructor(device) matches BpReconstructor");
        else
            YK_LOGE("FAIL: maxDiff={:.8f} exceeds threshold", maxDiff);
    }

    // ---- 对比：zslab_host vs zslab_device ----
    {
        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = std::abs(
                (double)h_vol_zslab[i] - (double)h_vol_zslab_host[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_elems;
        YK_LOGI("[zslab_host vs zslab_device] maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
        if (maxDiff < 1e-5)
            YK_LOGI("PASS: feedFromHost matches feed(device)");
        else
            YK_LOGE("FAIL: maxDiff={:.8f} exceeds threshold", maxDiff);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: bp_zslab verify (z_block=%d)\n", z_block_size);
    return 0;
}


// ----------------------------------------------------------------
// main_siddon_bp_runner：离线 + 在线 Siddon 反投影
// 数据未经滤波，直接用 Siddon BP 反投影，结果会比较模糊，但可以验证 Siddon BP 的正确性和性能
// ----------------------------------------------------------------
int main_siddon_bp_runner()
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

    // ---- 读投影数据 ----
    std::vector<float> h_flt(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_flt)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    MemoryController ctrl;
    auto d_sino_buf = ctrl.allocateDevice3D<float>(
        (size_t)params.iPU * params.iPV, Ang, 1, 0);
    {
        auto h_view = ctrl.allocateCpu3D<float>(
            (size_t)params.iPU * params.iPV, Ang, 1, false);
        std::memcpy(h_view.data(), h_flt.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_sino_buf, h_view);
    }

    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ---- 离线反投影 ----
    {
        ConeBackprojector recon;
        recon.init(params, ETask::BP_Siddon_RayDriven);
        YK::Util::CudaTimer timer("siddon_bp_offline", s);
        recon.run(d_sino_buf.data(), params, d_vol_buf.data(), s, /*clear_vol=*/true);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "siddon_bp_vol_offline.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: siddon_bp_vol_offline.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- 在线分包反投影 ----
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    ConeBackprojector recon_online;
    recon_online.init(params);
    {
        YK::Util::CudaTimer timer("siddon_bp_online", s);
        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);
            SCBCTParams bp_params = params;
            bp_params.iPAng = count;
            bp_params.angle_list = std::vector<float>(
                angle_list.begin() + base,
                angle_list.begin() + base + count);
            recon_online.run(
                d_sino_buf.data() + base * view_elems,
                bp_params,
                d_vol_buf.data(), s,
                /*clear_vol=*/(i == 0));
        }
    }
    recon_online.reset();
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "siddon_bp_vol_online.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: siddon_bp_vol_online.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}


// ----------------------------------------------------------------
// main_siddon_adjoint_verify：验证 Siddon FP/BP 伴随一致性
// <Ax, y> ≈ <x, A^T y>
// ----------------------------------------------------------------
int main_siddon_ray_adjoint_verify()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 24; params.iPAngTotal = 24;
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
    const size_t sino_n = view_elems * Ang;
    const size_t vol_n = (size_t)Nx * Ny * Nz;

    // ---- 随机初始化 x（体积）和 y（正弦图）────────────────────────
    std::vector<float> h_x(vol_n), h_y(sino_n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_y) v = dist(rng);

    YK_LOGI("x  stats: min={:.4f} max={:.4f}",
        *std::min_element(h_x.begin(), h_x.end()),
        *std::max_element(h_x.begin(), h_x.end()));
    YK_LOGI("y  stats: min={:.4f} max={:.4f}",
        *std::min_element(h_y.begin(), h_y.end()),
        *std::max_element(h_y.begin(), h_y.end()));

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    // 上传 x 和 y
    auto d_x = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_y = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_Ax = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_ATy = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        auto h_x_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_y_buf = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        std::memcpy(h_x_buf.data(), h_x.data(), vol_n * sizeof(float));
        std::memcpy(h_y_buf.data(), h_y.data(), sino_n * sizeof(float));
        ctrl.upload3D(d_x, h_x_buf);
        ctrl.upload3D(d_y, h_y_buf);
    }
    YK_CUDA_CHECK(cudaMemsetAsync(d_Ax.data(), 0, sino_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_ATy.data(), 0, vol_n * sizeof(float), s));

    // ---- Ax：正投影 x → sino ────────────────────────────────────
    {
        ConeProjector fp;
        fp.init(params, ETask::FP_Siddon);
        YK::Util::CudaTimer timer("siddon_fp", s);
        fp.run(d_x.data(), params, d_Ax.data(), s);
    }
    cudaStreamSynchronize(s);
    {
        auto h_Ax = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        ctrl.download3D(h_Ax, d_Ax);
        YK_LOGI("Ax stats: min={:.4e} max={:.4e}",
            *std::min_element(h_Ax.cdata(), h_Ax.cdata() + sino_n),
            *std::max_element(h_Ax.cdata(), h_Ax.cdata() + sino_n));
    }

    // ---- A^T y：反投影 y → vol ──────────────────────────────────
    // ---- A^T y：射线驱动 BP ────────────────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Siddon_RayDriven);
        YK::Util::CudaTimer timer("siddon_bp_ray", s);
        bp.run(d_y.data(), params, d_ATy.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_ATy = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_ATy, d_ATy);
        YK_LOGI("ATy stats: min={:.4e} max={:.4e}",
            *std::min_element(h_ATy.cdata(), h_ATy.cdata() + vol_n),
            *std::max_element(h_ATy.cdata(), h_ATy.cdata() + vol_n));
    }

    // ---- 下载结果，计算内积 ─────────────────────────────────────
    auto h_Ax_final = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
    auto h_ATy_final = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    ctrl.download3D(h_Ax_final, d_Ax);
    ctrl.download3D(h_ATy_final, d_ATy);
    cudaStreamSynchronize(s);

    // <Ax, y>
    double dot_Ax_y = 0.0;
    for (size_t i = 0; i < sino_n; ++i)
        dot_Ax_y += (double)h_Ax_final.cdata()[i] * (double)h_y[i];

    // <x, A^T y>
    double dot_x_ATy = 0.0;
    for (size_t i = 0; i < vol_n; ++i)
        dot_x_ATy += (double)h_x[i] * (double)h_ATy_final.cdata()[i];

    const double ratio = dot_Ax_y / dot_x_ATy;

    YK_LOGI("dot(Ax, y)   = {:.8e}", dot_Ax_y);
    YK_LOGI("dot(x, ATy)  = {:.8e}", dot_x_ATy);
    YK_LOGI("ratio        = {:.8f}", ratio);

    if (std::abs(ratio - 1.0) < 1e-3)
        YK_LOGI("PASS: Siddon FP/BP are adjoint (ratio close to 1)");
    else
        YK_LOGE("FAIL: ratio={:.8f} deviates from 1, FP/BP not adjoint", ratio);

    // ---- 保存中间结果供目视检查 ─────────────────────────────────
    {
        write_raw_float((test_data_dir + "adjoint_Ax.raw").c_str(),
            h_Ax_final.cdata(), sino_n);
        write_raw_float((test_data_dir + "adjoint_ATy.raw").c_str(),
            h_ATy_final.cdata(), vol_n);
        YK_LOGI("saved: adjoint_Ax.raw, adjoint_ATy.raw");
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

int main_siddon_voxel_adjoint_verify()
{
    // 参数和 main_siddon_adjoint_verify 完全一致
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 240; params.iPAngTotal = 240;
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
    const size_t sino_n = view_elems * Ang;
    const size_t vol_n = (size_t)Nx * Ny * Nz;

    std::vector<float> h_x(vol_n), h_y(sino_n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_y) v = dist(rng);

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_x = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_y = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_Ax = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_ATy = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        auto h_x_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_y_buf = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        std::memcpy(h_x_buf.data(), h_x.data(), vol_n * sizeof(float));
        std::memcpy(h_y_buf.data(), h_y.data(), sino_n * sizeof(float));
        ctrl.upload3D(d_x, h_x_buf);
        ctrl.upload3D(d_y, h_y_buf);
    }
    YK_CUDA_CHECK(cudaMemsetAsync(d_Ax.data(), 0, sino_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_ATy.data(), 0, vol_n * sizeof(float), s));

    // ---- Ax：Siddon FP（不变）────────────────────────────────────
    {
        ConeProjector fp;
        fp.init(params, ETask::FP_Siddon);
        YK::Util::CudaTimer timer("siddon_fp", s);
        fp.run(d_x.data(), params, d_Ax.data(), s);
    }
    cudaStreamSynchronize(s);
    {
        auto h_Ax = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        ctrl.download3D(h_Ax, d_Ax);
        YK_LOGI("Ax stats: min={:.4e} max={:.4e}",
            *std::min_element(h_Ax.cdata(), h_Ax.cdata() + sino_n),
            *std::max_element(h_Ax.cdata(), h_Ax.cdata() + sino_n));
    }

    // ---- A^T y：体素驱动 Siddon BP（被测）──────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_FDK);
        YK::Util::CudaTimer timer("siddon_bp_voxel", s);
        bp.run(d_y.data(), params, d_ATy.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_ATy = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_ATy, d_ATy);
        YK_LOGI("ATy stats: min={:.4e} max={:.4e}",
            *std::min_element(h_ATy.cdata(), h_ATy.cdata() + vol_n),
            *std::max_element(h_ATy.cdata(), h_ATy.cdata() + vol_n));
    }

    // ---- 下载，计算内积 ──────────────────────────────────────────
    auto h_Ax_final = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
    auto h_ATy_final = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    ctrl.download3D(h_Ax_final, d_Ax);
    ctrl.download3D(h_ATy_final, d_ATy);
    cudaStreamSynchronize(s);

    double dot_Ax_y = 0.0;
    for (size_t i = 0; i < sino_n; ++i)
        dot_Ax_y += (double)h_Ax_final.cdata()[i] * (double)h_y[i];

    double dot_x_ATy = 0.0;
    for (size_t i = 0; i < vol_n; ++i)
        dot_x_ATy += (double)h_x[i] * (double)h_ATy_final.cdata()[i];

    const double ratio = dot_Ax_y / dot_x_ATy;

    YK_LOGI("dot(Ax, y)   = {:.8e}", dot_Ax_y);
    YK_LOGI("dot(x, ATy)  = {:.8e}", dot_x_ATy);
    YK_LOGI("ratio        = {:.8f}", ratio);

    if (std::abs(ratio - 1.0) < 1e-3)
        YK_LOGI("PASS: voxel-driven Siddon BP is adjoint to Siddon FP");
    else
        YK_LOGE("FAIL: ratio={:.8f}, voxel BP not adjoint to Siddon FP", ratio);

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

int main_joseph_adjoint_verify()
{
    SCBCTParams params;
    params.iPU = params.iPV = 1024;
    params.iPAng = 240; params.iPAngTotal = 240;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = false;
    params.scan_range_rad = 2.0f * (float)CUDA_PI;
    params.SID = 400.0f; params.SDD = 1200.0f;
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
    const size_t sino_n = view_elems * Ang;
    const size_t vol_n = (size_t)Nx * Ny * Nz;

    std::vector<float> h_x(vol_n), h_y(sino_n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_y) v = dist(rng);

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_x = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_y = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_Ax = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_ATy_bilinear = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_ATy_tex = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        auto h_x_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_y_buf = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        std::memcpy(h_x_buf.data(), h_x.data(), vol_n * sizeof(float));
        std::memcpy(h_y_buf.data(), h_y.data(), sino_n * sizeof(float));
        ctrl.upload3D(d_x, h_x_buf);
        ctrl.upload3D(d_y, h_y_buf);
    }
    YK_CUDA_CHECK(cudaMemsetAsync(d_Ax.data(), 0, sino_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_ATy_bilinear.data(), 0, vol_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_ATy_tex.data(), 0, vol_n * sizeof(float), s));

    // ---- Ax：Joseph FP ──────────────────────────────────────────
    {
        ConeProjector fp;
        fp.init(params, ETask::FP_Joseph);
        YK::Util::CudaTimer timer("FP_Joseph", s);
        fp.run(d_x.data(), params, d_Ax.data(), s);
    }
    cudaStreamSynchronize(s);
    {
        auto h_Ax = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        ctrl.download3D(h_Ax, d_Ax);
        YK_LOGI("Ax stats: min={:.4e} max={:.4e}",
            *std::min_element(h_Ax.cdata(), h_Ax.cdata() + sino_n),
            *std::max_element(h_Ax.cdata(), h_Ax.cdata() + sino_n));
    }

    // ---- A^T y：Joseph BP 手动双线性 ───────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Joseph);
        YK::Util::CudaTimer timer("BP_Joseph", s);
        bp.run(d_y.data(), params, d_ATy_bilinear.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_ATy = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_ATy, d_ATy_bilinear);
        YK_LOGI("ATy(raw pointer) stats: min={:.4e} max={:.4e}",
            *std::min_element(h_ATy.cdata(), h_ATy.cdata() + vol_n),
            *std::max_element(h_ATy.cdata(), h_ATy.cdata() + vol_n));
    }

    // ---- A^T y：Joseph BP 纹理双线性 ───────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_FDK);
        YK::Util::CudaTimer timer("BP_FDK", s);
        bp.run(d_y.data(), params, d_ATy_tex.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_ATy = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_ATy, d_ATy_tex);
        YK_LOGI("ATy(tex) stats: min={:.4e} max={:.4e}",
            *std::min_element(h_ATy.cdata(), h_ATy.cdata() + vol_n),
            *std::max_element(h_ATy.cdata(), h_ATy.cdata() + vol_n));
    }

    // ---- 下载，计算内积 ─────────────────────────────────────────
    auto h_Ax_final = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
    auto h_ATy_bilinear_final = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    auto h_ATy_tex_final = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    ctrl.download3D(h_Ax_final, d_Ax);
    ctrl.download3D(h_ATy_bilinear_final, d_ATy_bilinear);
    ctrl.download3D(h_ATy_tex_final, d_ATy_tex);
    cudaStreamSynchronize(s);

    // <Ax, y>（两个 BP 共用同一个 Ax）
    double dot_Ax_y = 0.0;
    for (size_t i = 0; i < sino_n; ++i)
        dot_Ax_y += (double)h_Ax_final.cdata()[i] * (double)h_y[i];

    // <x, A^T y>  bilinear
    double dot_x_ATy_bilinear = 0.0;
    for (size_t i = 0; i < vol_n; ++i)
        dot_x_ATy_bilinear += (double)h_x[i] * (double)h_ATy_bilinear_final.cdata()[i];

    // <x, A^T y>  tex
    double dot_x_ATy_tex = 0.0;
    for (size_t i = 0; i < vol_n; ++i)
        dot_x_ATy_tex += (double)h_x[i] * (double)h_ATy_tex_final.cdata()[i];

    const double ratio_bilinear = dot_Ax_y / dot_x_ATy_bilinear;
    const double ratio_tex = dot_Ax_y / dot_x_ATy_tex;

    YK_LOGI("dot(Ax, y)            = {:.8e}", dot_Ax_y);
    YK_LOGI("dot(x, ATy_Joseph)  = {:.8e}  ratio={:.8f}", dot_x_ATy_bilinear, ratio_bilinear);
    YK_LOGI("dot(x, ATy_FDK)       = {:.8e}  ratio={:.8f}", dot_x_ATy_tex, ratio_tex);

    const double kThresh = 1e-2;
    if (std::abs(ratio_bilinear - 1.0) < kThresh)
        YK_LOGI("PASS: Joseph BP is adjoint to Joseph FP");
    else
        YK_LOGE("FAIL: Joseph BP ratio={:.8f}", ratio_bilinear);

    if (std::abs(ratio_tex - 1.0) < kThresh)
        YK_LOGI("PASS: FDK BP is adjoint to FDK FP");
    else
        YK_LOGE("FAIL: FDK BP ratio={:.8f}", ratio_tex);

    // ---- bilinear vs tex 一致性 ─────────────────────────────────
    {
        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_n; ++i) {
            double diff = std::abs(
                (double)h_ATy_bilinear_final.cdata()[i]
                - (double)h_ATy_tex_final.cdata()[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_n;
        YK_LOGI("[joseph_bp vs fdk_bp] maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

int main_joseph_v2_v3_adjoint_verify()
{
    SCBCTParams params;
    params.iPU = params.iPV = 1024;
    params.iPAng = 240; params.iPAngTotal = 240;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = false;
    params.scan_range_rad = 2.0f * (float)CUDA_PI;
    params.SID = 400.0f; params.SDD = 1200.0f;
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
    const size_t sino_n = view_elems * Ang;
    const size_t vol_n = (size_t)Nx * Ny * Nz;

    std::vector<float> h_x(vol_n), h_y(sino_n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_y) v = dist(rng);

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_x = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_y = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_Ax = ctrl.allocateDevice3D<float>(view_elems, Ang, 1, 0);
    auto d_ATy_v2 = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_ATy_v3 = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        auto h_x_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_y_buf = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        std::memcpy(h_x_buf.data(), h_x.data(), vol_n * sizeof(float));
        std::memcpy(h_y_buf.data(), h_y.data(), sino_n * sizeof(float));
        ctrl.upload3D(d_x, h_x_buf);
        ctrl.upload3D(d_y, h_y_buf);
    }
    YK_CUDA_CHECK(cudaMemsetAsync(d_Ax.data(), 0, sino_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_ATy_v2.data(), 0, vol_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_ATy_v3.data(), 0, vol_n * sizeof(float), s));

    // ---- Ax：Joseph FP ──────────────────────────────────────────
    {
        ConeProjector fp;
        fp.init(params, ETask::FP_Joseph);
        YK::Util::CudaTimer timer("FP_Joseph", s);
        fp.run(d_x.data(), params, d_Ax.data(), s);
    }
    cudaStreamSynchronize(s);
    {
        auto h_Ax = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
        ctrl.download3D(h_Ax, d_Ax);
        YK_LOGI("Ax stats: min={:.4e} max={:.4e}",
            *std::min_element(h_Ax.cdata(), h_Ax.cdata() + sino_n),
            *std::max_element(h_Ax.cdata(), h_Ax.cdata() + sino_n));
    }

    // ---- A^T y：Joseph BP 手动双线性 ───────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Joseph_v2);
        YK::Util::CudaTimer timer("BP_Joseph_v2", s);
        bp.run(d_y.data(), params, d_ATy_v2.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_ATy = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_ATy, d_ATy_v2);
        YK_LOGI("ATy(raw pointer) stats: min={:.4e} max={:.4e}",
            *std::min_element(h_ATy.cdata(), h_ATy.cdata() + vol_n),
            *std::max_element(h_ATy.cdata(), h_ATy.cdata() + vol_n));
    }

    // ---- A^T y：Joseph BP 纹理双线性 ───────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Joseph_v3);
        YK::Util::CudaTimer timer("BP_Joseph_v3", s);
        bp.run(d_y.data(), params, d_ATy_v3.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_ATy = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_ATy, d_ATy_v3);
        YK_LOGI("ATy(tex) stats: min={:.4e} max={:.4e}",
            *std::min_element(h_ATy.cdata(), h_ATy.cdata() + vol_n),
            *std::max_element(h_ATy.cdata(), h_ATy.cdata() + vol_n));
    }

    // ---- 下载，计算内积 ─────────────────────────────────────────
    auto h_Ax_final = ctrl.allocateCpu3D<float>(view_elems, Ang, 1, false);
    auto h_ATy_v2 = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    auto h_ATy_v3 = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    ctrl.download3D(h_Ax_final, d_Ax);
    ctrl.download3D(h_ATy_v2, d_ATy_v2);
    ctrl.download3D(h_ATy_v3, d_ATy_v3);
    cudaStreamSynchronize(s);

    // <Ax, y>（两个 BP 共用同一个 Ax）
    double dot_Ax_y = 0.0;
    for (size_t i = 0; i < sino_n; ++i)
        dot_Ax_y += (double)h_Ax_final.cdata()[i] * (double)h_y[i];

    // <x, A^T y>  bilinear
    double dot_x_ATy_v2 = 0.0;
    for (size_t i = 0; i < vol_n; ++i)
        dot_x_ATy_v2 += (double)h_x[i] * (double)h_ATy_v2.cdata()[i];

    // <x, A^T y>  tex
    double dot_x_ATy_v3 = 0.0;
    for (size_t i = 0; i < vol_n; ++i)
        dot_x_ATy_v3 += (double)h_x[i] * (double)h_ATy_v3.cdata()[i];

    const double ratio_v2 = dot_Ax_y / dot_x_ATy_v2;
    const double ratio_v3 = dot_Ax_y / dot_x_ATy_v3;

    YK_LOGI("dot(Ax, y)            = {:.8e}", dot_Ax_y);
    YK_LOGI("dot(x, ATy_v2)        = {:.8e}  ratio={:.8f}", dot_x_ATy_v2, ratio_v2);
    YK_LOGI("dot(x, ATy_v3)        = {:.8e}  ratio={:.8f}", dot_x_ATy_v3, ratio_v3);

    const double kThresh = 1e-2;
    if (std::abs(ratio_v2 - 1.0) < kThresh)
        YK_LOGI("PASS: Joseph BP is adjoint to Joseph FP");
    else
        YK_LOGE("FAIL: v2 BP ratio={:.8f}", ratio_v2);

    if (std::abs(ratio_v3 - 1.0) < kThresh)
        YK_LOGI("PASS: v3 BP is adjoint to v3 FP");
    else
        YK_LOGE("FAIL: v3 BP ratio={:.8f}", ratio_v3);

    // ---- bilinear vs tex 一致性 ─────────────────────────────────
    {
        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_n; ++i) {
            double diff = std::abs(
                (double)h_ATy_v2.cdata()[i]
                - (double)h_ATy_v3.cdata()[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_n;
        YK_LOGI("[v2 BP vs v3 BP] maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}


// ----------------------------------------------------------------
// main_siddon_bp_verify：对比 SiddonBpReconstructor 与 BpReconstructor
// 输入：相同的滤波投影，对比两者反投影结果
// ----------------------------------------------------------------
int main_siddon_bp_verify()
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

    auto d_vol_voxel = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_vol_ray = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_vol_vox = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    std::vector<float> h_flt_all(proj_elems, 0.f);

    // ---- 路径一：FDK dump 滤波投影 + 体素驱动 FDK BP ───────────
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
            d_vol_voxel.data(), true, onDump, &ctx);
    }
    cudaStreamSynchronize(s);
    {
        auto h_vol_voxel = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_voxel, d_vol_voxel);
        write_raw_float((test_data_dir + "siddon_verify_vol_voxel.raw").c_str(),
            h_vol_voxel.cdata(), vol_elems);
        YK_LOGI("saved: siddon_verify_vol_voxel.raw");

        float maxv = *std::max_element(h_flt_all.begin(), h_flt_all.end());
        float minv = *std::min_element(h_flt_all.begin(), h_flt_all.end());
        YK_LOGI("h_flt_all stats: min={:.6f} max={:.6f}", minv, maxv);
    }

    float* d_flt_raw = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 路径二：射线驱动 Siddon BP ────────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Siddon_RayDriven);
        YK::Util::CudaTimer timer("siddon_bp_ray", s);
        bp.run(d_flt_raw, params, d_vol_ray.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_vol_ray = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_ray, d_vol_ray);
        write_raw_float((test_data_dir + "siddon_verify_vol_ray.raw").c_str(),
            h_vol_ray.cdata(), vol_elems);
        YK_LOGI("saved: siddon_verify_vol_ray.raw");
    }

    // ---- 路径三：体素驱动 Siddon BP ────────────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Siddon_VoxDriven);
        YK::Util::CudaTimer timer("siddon_bp_vox", s);
        bp.run(d_flt_raw, params, d_vol_vox.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_vol_vox = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_vox, d_vol_vox);
        write_raw_float((test_data_dir + "siddon_verify_vol_vox.raw").c_str(),
            h_vol_vox.cdata(), vol_elems);
        YK_LOGI("saved: siddon_verify_vol_vox.raw");
    }

    cudaFree(d_flt_raw);

    // ---- 三路对比 ───────────────────────────────────────────────
    {
        auto h_vol_voxel = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_vol_ray = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_vol_vox = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_voxel, d_vol_voxel);
        ctrl.download3D(h_vol_ray, d_vol_ray);
        ctrl.download3D(h_vol_vox, d_vol_vox);

        auto compare = [&](const char* nameA, const float* A,
            const char* nameB, const float* B) {
                double maxDiff = 0.0, mse = 0.0, sumA = 0.0, sumB = 0.0;
                for (size_t i = 0; i < vol_elems; ++i) {
                    double diff = std::abs((double)A[i] - (double)B[i]);
                    maxDiff = std::max(maxDiff, diff);
                    mse += diff * diff;
                    sumA += A[i];
                    sumB += B[i];
                }
                mse /= (double)vol_elems;
                YK_LOGI("{} sum={:.6e}  {} sum={:.6e}  maxDiff={:.6f}  MSE={:.6e}",
                    nameA, sumA, nameB, sumB, maxDiff, mse);
            };

        compare("FDK_voxel", h_vol_voxel.cdata(),
            "Siddon_ray", h_vol_ray.cdata());
        compare("FDK_voxel", h_vol_voxel.cdata(),
            "Siddon_vox", h_vol_vox.cdata());
        compare("Siddon_ray", h_vol_ray.cdata(),
            "Siddon_vox", h_vol_vox.cdata());

        // Siddon ray 和 vox 之间的一致性检查
        double sum_ray = 0.0, sum_vox = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            sum_ray += h_vol_ray.cdata()[i];
            sum_vox += h_vol_vox.cdata()[i];
        }
        if (sum_ray > 0.0 && sum_vox > 0.0)
            YK_LOGI("PASS: both Siddon BP variants produce non-zero results");
        else
            YK_LOGE("FAIL: one or both Siddon BP results are zero");
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}


int main_siddon_zslab_verify()
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
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;

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

    // ---- 路径一：Siddon BP 全量参考（不分 slab）────────────────
    auto d_vol_ref = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    std::vector<float> h_flt_all(proj_elems, 0.f);

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
            ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false).data(),
            true, onDump, &ctx);
    }
    cudaStreamSynchronize(s);

    float* d_flt_raw = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice));

    {
        YK::Util::CudaTimer timer("siddon_bp_ref", s);
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Siddon_VoxDriven);
        bp.run(d_flt_raw, params, d_vol_ref.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_ref);
        write_raw_float((test_data_dir + "siddon_zslab_ref.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: siddon_zslab_ref.raw");
    }

    const int z_block_size = 100;  // 400 / 100 = 4 slabs

    // ---- 路径二：SiddonZSlabReconstructor（设备端输入）─────────
    std::vector<float> h_vol_zslab(vol_elems, 0.f);
    {
        YK::Util::CudaTimer timer("siddon_zslab_device", s);
        SiddonZSlabReconstructor recon;
        if (!recon.init(params, ETask::BP_Siddon_VoxDriven, s, z_block_size)) {
            YK_LOGE("SiddonZSlabReconstructor init failed");
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        if (!recon.feed(d_flt_raw, params, s, h_vol_zslab.data())) {
            YK_LOGE("SiddonZSlabReconstructor feed failed");
            recon.release();
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        recon.release();
    }
    write_raw_float((test_data_dir + "siddon_zslab_device.raw").c_str(),
        h_vol_zslab.data(), vol_elems);
    YK_LOGI("saved: siddon_zslab_device.raw");

    // ---- 路径三：SiddonZSlabReconstructor（主机端输入）─────────
    std::vector<float> h_vol_zslab_host(vol_elems, 0.f);
    {
        YK::Util::CudaTimer timer("siddon_zslab_host", s);
        SiddonZSlabReconstructor recon;
        if (!recon.init(params, ETask::BP_Siddon_VoxDriven, s, z_block_size)) {
            YK_LOGE("SiddonZSlabReconstructor init failed");
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        if (!recon.feedFromHost(h_flt_all.data(), params, s, h_vol_zslab_host.data())) {
            YK_LOGE("SiddonZSlabReconstructor feedFromHost failed");
            recon.release();
            cudaFree(d_flt_raw);
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        recon.release();
    }
    write_raw_float((test_data_dir + "siddon_zslab_host.raw").c_str(),
        h_vol_zslab_host.data(), vol_elems);
    YK_LOGI("saved: siddon_zslab_host.raw");

    cudaFree(d_flt_raw);

    // ---- 对比：zslab_device vs ref ──────────────────────────────
    {
        auto h_vol_ref = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_ref, d_vol_ref);

        double maxDiff = 0.0, mse = 0.0;
        double sum_ref = 0.0, sum_zslab = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = std::abs(
                (double)h_vol_ref.cdata()[i] - (double)h_vol_zslab[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
            sum_ref += h_vol_ref.cdata()[i];
            sum_zslab += h_vol_zslab[i];
        }
        mse /= (double)vol_elems;
        YK_LOGI("ref sum={:.6e}  zslab_device sum={:.6e}", sum_ref, sum_zslab);
        YK_LOGI("[zslab_device vs ref] maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
        if (maxDiff < 5e-3)
            YK_LOGI("PASS: SiddonZSlab(device) matches full BP");
        else
            YK_LOGE("FAIL: maxDiff={:.8f} exceeds threshold", maxDiff);
    }

    // ---- 对比：zslab_host vs zslab_device ───────────────────────
    {
        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = std::abs(
                (double)h_vol_zslab[i] - (double)h_vol_zslab_host[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_elems;
        YK_LOGI("[zslab_host vs zslab_device] maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);
        if (maxDiff < 1e-5)
            YK_LOGI("PASS: feedFromHost matches feed(device)");
        else
            YK_LOGE("FAIL: maxDiff={:.8f} exceeds threshold", maxDiff);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: siddon_zslab verify (z_block=%d)\n", z_block_size);
    return 0;
}



int main_fp_bp_geometry_verify()
{
    SCBCTParams params;
    params.iPU = params.iPV = 1024;
    params.iPAng = 240; params.iPAngTotal = 240;
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

    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t sino_n = view_elems * params.iPAng;
    const size_t vol_n = (size_t)Nx * Ny * Nz;

    // ── 构造球形 phantom ──────────────────────────────────────────
    std::vector<float> h_phantom(vol_n, 0.f);
    {
        const int cx = Nx / 2, cy = Ny / 2, cz = Nz / 2;
        const int r = std::min({ Nx, Ny, Nz }) / 8;
        for (int iz = 0; iz < Nz; ++iz)
            for (int iy = 0; iy < Ny; ++iy)
                for (int ix = 0; ix < Nx; ++ix) {
                    const int dx = ix - cx, dy = iy - cy, dz = iz - cz;
                    if (dx * dx + dy * dy + dz * dz <= r * r)
                        h_phantom[(size_t)iz * Ny * Nx + iy * Nx + ix] = 1.f;
                }
    }
    write_raw_float((test_data_dir + "geom_phantom.raw").c_str(), h_phantom);
    YK_LOGI("[geom] saved geom_phantom.raw  {} {} {}", Nx, Ny, Nz);

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_phantom = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    auto d_bp = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    {
        auto h_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        std::memcpy(h_buf.data(), h_phantom.data(), vol_n * sizeof(float));
        ctrl.upload3D(d_phantom, h_buf);
    }
    YK_CUDA_CHECK(cudaMemsetAsync(d_sino.data(), 0, sino_n * sizeof(float), s));
    YK_CUDA_CHECK(cudaMemsetAsync(d_bp.data(), 0, vol_n * sizeof(float), s));

    // ── FP ───────────────────────────────────────────────────────
    {
        ConeProjector fp;
        fp.init(params, ETask::FP_Siddon);
        YK::Util::CudaTimer timer("FP_Siddon", s);
        fp.run(d_phantom.data(), params, d_sino.data(), s);
    }
    cudaStreamSynchronize(s);
    {
        auto h_sino = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        ctrl.download3D(h_sino, d_sino);
        YK_LOGI("[geom] sino: min={:.4e} max={:.4e}",
            *std::min_element(h_sino.cdata(), h_sino.cdata() + sino_n),
            *std::max_element(h_sino.cdata(), h_sino.cdata() + sino_n));
        write_raw_float((test_data_dir + "geom_sino.raw").c_str(), h_sino.cdata(), sino_n);
        YK_LOGI("[geom] saved geom_sino.raw  {} {} {}",
            params.iPU, params.iPV, params.iPAng);
    }

    // ── BP：改成 OSSART 里实际用的 bp_task ───────────────────────
    {
        ConeBackprojector bp;
        bp.init(params, ETask::BP_Joseph);
        YK::Util::CudaTimer timer("BP_Joseph", s);
        bp.run(d_sino.data(), params, d_bp.data(), s, /*clear_vol=*/true);
    }
    cudaStreamSynchronize(s);
    {
        auto h_bp = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_bp, d_bp);
        YK_LOGI("[geom] bp: min={:.4e} max={:.4e}",
            *std::min_element(h_bp.cdata(), h_bp.cdata() + vol_n),
            *std::max_element(h_bp.cdata(), h_bp.cdata() + vol_n));
        write_raw_float((test_data_dir + "geom_bp.raw").c_str(), h_bp.cdata(), vol_n);
        YK_LOGI("[geom] saved geom_bp.raw  {} {} {}", Nx, Ny, Nz);

        // 重心检查
        double sx = 0, sy = 0, sz = 0, sw = 0;
        for (int iz = 0; iz < Nz; ++iz)
            for (int iy = 0; iy < Ny; ++iy)
                for (int ix = 0; ix < Nx; ++ix) {
                    const float w = h_bp.cdata()[(size_t)iz * Ny * Nx + iy * Nx + ix];
                    if (w > 0.f) {
                        sx += w * ix; sy += w * iy; sz += w * iz;
                        sw += w;
                    }
                }
        if (sw > 0.f) {
            const double cx = sx / sw, cy = sy / sw, cz = sz / sw;
            YK_LOGI("[geom] bp centroid  : ({:.2f}, {:.2f}, {:.2f})", cx, cy, cz);
            YK_LOGI("[geom] phantom center: ({}, {}, {})", Nx / 2, Ny / 2, Nz / 2);
            const double ddx = cx - Nx / 2;
            const double ddy = cy - Ny / 2;
            const double ddz = cz - Nz / 2;
            YK_LOGI("[geom] centroid offset: ({:.2f}, {:.2f}, {:.2f}) voxels",
                ddx, ddy, ddz);
            const double dist = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            if (dist < 1.0)
                YK_LOGI("[geom] PASS: centroid within 1 voxel of phantom center");
            else
                YK_LOGE("[geom] FAIL: centroid offset={:.2f} voxels", dist);
        }
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}