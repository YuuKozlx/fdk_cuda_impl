#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include <global/YkMacro.hpp>
#include "FDK/YkFdkReconstructor.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"
#include "test_common.hpp"
#include <BP/YkBPRunner.hpp>

using namespace YK;
using namespace Mem;

const std::string test_data_dir = "/workspace/fdk-test/TestData/";

const std::string PROJ_FILE = test_data_dir + "proj_1024x1024x360.raw";

// ----------------------------------------------------------------
// 公共参数构造
// ----------------------------------------------------------------
static SCBCTParams makeFdkParams()
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

    params.angle_list.resize(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        params.angle_list[i] = i * 2.0f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = params.angle_list[0];
    return params;
}

// ----------------------------------------------------------------
// FdkTest：离线重建（目视类，验证能跑通且结果非零）
// ----------------------------------------------------------------
TEST(FdkTest, OfflineRecon_NonZeroOutput)
{
    auto params = makeFdkParams();
    const size_t proj_elems = (size_t)params.iPU * params.iPV * params.iPAng;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float(PROJ_FILE.c_str(), h_proj)) {
        GTEST_SKIP() << "proj_1024x1024x360.raw not found, skipping";
    }

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    MemoryController ctrl;
    auto d_vol = ctrl.allocateDevice3D<float>(params.iVX, params.iVY, params.iVZ, 0, false);

    bool ok = YK::fdk_recon(h_proj.data(), d_vol.data(), params,
        /*Kchunk=*/32, s, /*clear_vol=*/true);
    EXPECT_TRUE(ok);

    auto h_vol = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    ctrl.download3D(h_vol, d_vol);

    float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
    EXPECT_GT(maxv, 0.f) << "Reconstructed volume is all zeros";

    write_raw_float((test_data_dir + "fdk_vec_vol_offline.raw").c_str(), h_vol.cdata(), vol_elems);

    cudaStreamDestroy(s);
}

// ----------------------------------------------------------------
// FdkTest：在线分包重建，结果与离线一致
// ----------------------------------------------------------------
TEST(FdkTest, OnlineRecon_MatchesOffline)
{
    auto params = makeFdkParams();
    const int Ang = params.iPAng;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float(PROJ_FILE.c_str(), h_proj)) {
        GTEST_SKIP() << "proj_1024x1024x360.raw not found, skipping";
    }

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    MemoryController ctrl;
    auto d_vol_offline = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);
    auto d_vol_online = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);

    // 离线
    ASSERT_TRUE(YK::fdk_recon(h_proj.data(), d_vol_offline.data(), params,
        32, s, true));

    // 在线分包
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;
    auto angle_list = params.angle_list;

    FdkReconstructor recon;
    ASSERT_TRUE(recon.init(params, 32, s));

    for (int i = 0; i < batch_num; ++i) {
        const int base = i * batch_size;
        const int count = std::min(batch_size, Ang - base);

        SCBCTParams bp = params;
        bp.iPAng = count;
        bp.angle_list = std::vector<float>(
            angle_list.begin() + base,
            angle_list.begin() + base + count);

        ASSERT_TRUE(recon.feed(h_proj.data() + base * view_elems,
            bp, s, d_vol_online.data(), (i == 0)));
    }

    cudaStreamSynchronize(s);

    auto h_offline = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    auto h_online = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    ctrl.download3D(h_offline, d_vol_offline);
    ctrl.download3D(h_online, d_vol_online);

    double maxDiff = 0.0;
    for (size_t i = 0; i < vol_elems; ++i) {
        double diff = std::abs((double)h_offline.cdata()[i] - (double)h_online.cdata()[i]);
        maxDiff = std::max(maxDiff, diff);
    }

    EXPECT_LT(maxDiff, 1e-4) << "Online recon differs from offline, maxDiff=" << maxDiff;

    write_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_online.cdata(), vol_elems);

    cudaStreamDestroy(s);
}

// ----------------------------------------------------------------
// FdkTest：BpReconstructor 与 FDK 反投影阶段一致性验证
// 数值验证类
// ----------------------------------------------------------------
TEST(FdkTest, BpReconstructor_MatchesFdk)
{
    auto params = makeFdkParams();
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float(PROJ_FILE.c_str(), h_proj)) {
        GTEST_SKIP() << "proj_1024x1024x360.raw not found, skipping";
    }

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    MemoryController ctrl;
    auto d_vol_fdk = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);
    auto d_vol_bp = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);

    std::vector<float> h_flt_all(proj_elems, 0.f);

    // FDK + dump flt
    {
        FdkReconstructor recon;
        ASSERT_TRUE(recon.init(params, 32, s));

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

        ASSERT_TRUE(recon.feed(h_proj.data(), params, s,
            d_vol_fdk.data(), true, onDump, &ctx));
    }

    cudaStreamSynchronize(s);

    // BpReconstructor
    float* d_flt_raw = nullptr;
    ASSERT_EQ(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    {
        YK::BpReconstructor recon;
        ASSERT_TRUE(recon.init(params, 32, s));
        ASSERT_TRUE(recon.feed(d_flt_raw, params, s, d_vol_bp.data(), true));
    }

    cudaStreamSynchronize(s);
    cudaFree(d_flt_raw);

    auto h_vol_fdk = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    auto h_vol_bp = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    ctrl.download3D(h_vol_fdk, d_vol_fdk);
    ctrl.download3D(h_vol_bp, d_vol_bp);

    double maxDiff = 0.0, mse = 0.0;
    for (size_t i = 0; i < vol_elems; ++i) {
        double diff = std::abs((double)h_vol_fdk.cdata()[i] - (double)h_vol_bp.cdata()[i]);
        maxDiff = std::max(maxDiff, diff);
        mse += diff * diff;
    }
    mse /= (double)vol_elems;

    EXPECT_LT(maxDiff, 1e-5) << "BpReconstructor differs from FDK, maxDiff=" << maxDiff;
    EXPECT_LT(mse, 1e-10) << "MSE too large: " << mse;

    write_raw_float((test_data_dir + "bp_vec_vol_online_bpvsfdk.raw").c_str(), h_vol_bp.cdata(), vol_elems);
    write_raw_float((test_data_dir + "fdk_vec_vol_online_bpvsfdk.raw").c_str(), h_vol_fdk.cdata(), vol_elems);

    cudaStreamDestroy(s);
}
