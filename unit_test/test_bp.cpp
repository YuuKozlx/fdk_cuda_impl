#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "test_common.hpp"
#include "BP/YkBPRunner.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"

using namespace YK;
using namespace Mem;

const std::string test_data_dir = R"(G:\Code\fanproj\fdk-test\TestData\)";

static SCBCTParams makeBpParams()
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

    params.angle_list.resize(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        params.angle_list[i] = i * 2.0f * (float)CUDA_PI / params.iPAng;
    params.scan_start_angle_rad = params.angle_list[0];
    return params;
}

// ----------------------------------------------------------------
// BpTest：离线反投影（目视类，验证能跑通且结果非零）
// ----------------------------------------------------------------
TEST(BpTest, OfflineBp_NonZeroOutput)
{
    auto params = makeBpParams();
    const size_t proj_elems = (size_t)params.iPU * params.iPV * params.iPAng;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    std::vector<float> h_flt(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_flt)) {
        GTEST_SKIP() << "proj_1024x1024x360.raw not found, skipping";
    }

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    MemoryController ctrl;

    // 上传滤波投影
    auto d_flt = ctrl.allocateDevice3D<float>(
        (size_t)params.iPU * params.iPV, params.iPAng, 1, 0);
    {
        auto h_view = ctrl.allocateCpu3D<float>(
            (size_t)params.iPU * params.iPV, params.iPAng, 1, false);
        std::memcpy(h_view.data(), h_flt.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_flt, h_view);
    }

    auto d_vol = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);

    BpReconstructor recon;
    ASSERT_TRUE(recon.init(params, 32, s));
    ASSERT_TRUE(recon.feed(d_flt.data(), params, s, d_vol.data(), true));

    cudaStreamSynchronize(s);

    auto h_vol = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    ctrl.download3D(h_vol, d_vol);

    float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
    EXPECT_GT(maxv, 0.f) << "BP output is all zeros";

    write_raw_float((test_data_dir + "bp_vol_offline.raw").c_str(), h_vol.cdata(), vol_elems);

    cudaStreamDestroy(s);
}

// ----------------------------------------------------------------
// BpTest：在线分包反投影，结果与离线一致
// ----------------------------------------------------------------
TEST(BpTest, OnlineBp_MatchesOffline)
{
    auto params = makeBpParams();
    const int Ang = params.iPAng;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    std::vector<float> h_flt(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_flt)) {
        GTEST_SKIP() << "proj_1024x1024x360.raw not found, skipping";
    }

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    MemoryController ctrl;

    float* d_flt_raw = nullptr;
    ASSERT_EQ(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_flt_raw, h_flt.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    auto d_vol_offline = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);
    auto d_vol_online = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);

    auto angle_list = params.angle_list;

    // 离线
    {
        BpReconstructor recon;
        ASSERT_TRUE(recon.init(params, 32, s));
        ASSERT_TRUE(recon.feed(d_flt_raw, params, s, d_vol_offline.data(), true));
    }

    // 在线分包
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    {
        BpReconstructor recon;
        ASSERT_TRUE(recon.init(params, 32, s));

        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams bp = params;
            bp.iPAng = count;
            bp.angle_list = std::vector<float>(
                angle_list.begin() + base,
                angle_list.begin() + base + count);

            ASSERT_TRUE(recon.feed(d_flt_raw + base * view_elems,
                bp, s, d_vol_online.data(), (i == 0)));
        }
    }

    cudaStreamSynchronize(s);
    cudaFree(d_flt_raw);

    auto h_offline = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    auto h_online = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    ctrl.download3D(h_offline, d_vol_offline);
    ctrl.download3D(h_online, d_vol_online);

    double maxDiff = 0.0;
    for (size_t i = 0; i < vol_elems; ++i) {
        double diff = std::abs((double)h_offline.cdata()[i] - (double)h_online.cdata()[i]);
        maxDiff = std::max(maxDiff, diff);
    }

    EXPECT_LT(maxDiff, 1e-4) << "Online BP differs from offline, maxDiff=" << maxDiff;

    write_raw_float((test_data_dir + "bp_vol_online.raw").c_str(), h_online.cdata(), vol_elems);
    write_raw_float((test_data_dir + "bp_vol_offline.raw").c_str(), h_offline.cdata(), vol_elems);

    cudaStreamDestroy(s);
}
