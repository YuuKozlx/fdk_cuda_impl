#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include <global/YkMacro.hpp>
#include "FDK/YkFdkPipeline.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"
#include "test_common.hpp"
#include "common/YkProjectionOperators.hpp"

using namespace YK;
using namespace Mem;

const std::string test_data_dir = R"(H:\Code\fanproj\fdk-test\TestData\)";
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

// 圆轨迹测试统一先构造完整 geometry 并预计算。这样离线与在线测试都走
// Session 的正式 FdkPipeline 数据流，而非旧 runner 的独立实现。
static bool runFdkPipeline(const float* h_proj, float* d_volume,
    const SCBCTParams& params, int chunk, cudaStream_t stream, bool clear_output)
{
    FdkPipeline pipeline;
    if (!pipeline.prepareWithAngles(params, params.angle_list, chunk, stream)) return false;
    const FdkProjectionBatch batch{ h_proj, nullptr, nullptr, params.iPAng };
    return pipeline.processBatch(batch, d_volume, clear_output);
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

    bool ok = runFdkPipeline(h_proj.data(), d_vol.data(), params,
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
    ASSERT_TRUE(runFdkPipeline(h_proj.data(), d_vol_offline.data(), params,
        32, s, true));

    // 在线分包
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;
    FdkPipeline pipeline;
    ASSERT_TRUE(pipeline.prepareWithAngles(params, params.angle_list, 32, s));

    for (int i = 0; i < batch_num; ++i) {
        const int base = i * batch_size;
        const int count = std::min(batch_size, Ang - base);

        const FdkProjectionBatch batch{
            h_proj.data() + base * view_elems, nullptr, nullptr, count };
        ASSERT_TRUE(pipeline.processBatch(batch, d_vol_online.data(), i == 0));
    }
    EXPECT_TRUE(pipeline.complete());

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
// FdkTest：通用 BP operator 与 FDK 反投影阶段一致性验证
// 数值验证类
// ----------------------------------------------------------------
TEST(FdkTest, BackOperator_MatchesFdk)
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

    // FDK：测试直接读取 pipeline 暴露的最后一个 stage 设备视图，不再使用
    // 算法内部回调或在计算路径中强制同步。
    {
        FdkPipeline pipeline;
        ASSERT_TRUE(pipeline.prepareWithAngles(params, params.angle_list, params.iPAng, s));
        const FdkProjectionBatch batch{ h_proj.data(), nullptr, nullptr, params.iPAng };
        ASSERT_TRUE(pipeline.processBatch(batch, d_vol_fdk.data(), true));
        const FdkStageView& stage = pipeline.lastStage();
        ASSERT_TRUE(stage.valid());
        ASSERT_EQ(stage.first_view, 0);
        ASSERT_EQ(stage.count, params.iPAng);
        ASSERT_EQ(cudaMemcpyAsync(h_flt_all.data(), stage.d_filtered,
            proj_elems * sizeof(float), cudaMemcpyDeviceToHost, stage.stream), cudaSuccess);
    }

    cudaStreamSynchronize(s);

    // 直接调用通用 BP operator。滤波投影和目标体数据均由调用方持有，
    // 因此无 runner、无回调，也不会在算法内部强制同步。
    float* d_flt_raw = nullptr;
    ASSERT_EQ(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    {
        GeometryContext geometry;
        ResourceContext resources;
        ASSERT_TRUE(geometry.initialize(params));
        resources.attach(s, 0);
        auto bp = makeBackOperator(ETask::BP_FDK);
        ASSERT_TRUE(bp->prepare(geometry, resources));
        ASSERT_TRUE(bp->apply(d_flt_raw, params, d_vol_bp.data(), true, resources));
        bp->release();
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

    EXPECT_LT(maxDiff, 1e-5) << "BackOperator differs from FDK, maxDiff=" << maxDiff;
    EXPECT_LT(mse, 1e-10) << "MSE too large: " << mse;

    write_raw_float((test_data_dir + "bp_vec_vol_online_bpvsfdk.raw").c_str(), h_vol_bp.cdata(), vol_elems);
    write_raw_float((test_data_dir + "fdk_vec_vol_online_bpvsfdk.raw").c_str(), h_vol_fdk.cdata(), vol_elems);

    cudaStreamDestroy(s);
}
