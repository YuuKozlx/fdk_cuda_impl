#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include "test_common.hpp"

#include <common/YkVecGeo.hpp>
#include <global/YkMem3d.hpp>
#include <random>
#include "FP/YkFPRunner.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include <FP/YkFpRunnerExVec.hpp>

using namespace YK;

const std::string test_data_dir = R"(G:\Code\fanproj\fdk-test\TestData\)";

//void test_fp_runner(cudaStream_t stream)
//{
//    printf("\n[ConeProjector] test\n");
//
//    constexpr int   Nx = 512, Ny = 512, Nz = 400;
//    constexpr float vox_xy = 0.1f, vox_z = 0.1f;
//    constexpr int   Na = 480, Nu = 1024, Nv = 1024;
//    constexpr float du = 0.25f, dv = 0.25f;
//    constexpr float SID = 500.f, SDD = 1000.f;
//
//    SCBCTParams params;
//    params.iVX = Nx; params.iVY = Ny; params.iVZ = Nz;
//    params.vox_x_mm = vox_xy; params.vox_z_mm = vox_z;
//    params.vol_offset_x_mm = 0.f;
//    params.vol_offset_y_mm = 0.f;
//    params.vol_offset_z_mm = 0.f;
//    params.iPAng = Na;
//    params.iPU = Nu; params.iPV = Nv;
//    params.du_mm = du; params.dv_mm = dv;
//    params.SID = SID; params.SDD = SDD;
//    params.offsetU_mm = 0.f; params.offsetV_mm = 0.f;
//    params.tiltu_angle_rad = 0.f;
//    params.tiltn_angle_rad = 0.f;
//    params.tiltv_angle_rad = 0.f;
//
//    params.angle_list.resize(Na);
//    for (int i = 0; i < Na; ++i)
//        params.angle_list[i] = 2.f * CUDA_PI * i / 720;
//
//    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
//    if (!read_raw_float((test_data_dir + "fdk_vec_vol_offline.raw").c_str(), h_vol)) {
//        printf("  fdk_vec_vol_offline.raw not found, skip\n");
//        return;
//    }
//    printf("  volume loaded\n");
//
//    float* d_vol = nullptr;
//    YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
//    YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
//        h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));
//    h_vol.clear();
//
//    const size_t sino_elems = (size_t)Na * Nv * Nu;
//    float* d_sino = nullptr;
//    YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
//
//    ConeProjector fpr;
//    if (!fpr.init(params, ETask::FP_Joseph, 0)) {
//        YK_LOGE("ConeProjector init failed");
//        cudaFree(d_vol); cudaFree(d_sino);
//        return;
//    }
//
//    bool ok = fpr.run(d_vol, params, d_sino, stream);
//    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//    YK_LOGI("  fp_project: {}", ok ? "OK" : "FAILED");
//
//    std::vector<float> h_sino(sino_elems);
//    YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
//        sino_elems * sizeof(float), cudaMemcpyDeviceToHost));
//
//    printStats(h_sino, "sino");
//    write_raw_float((test_data_dir + "fp_reconstructor_sino.raw").c_str(), h_sino.data(), sino_elems);
//    printf("  saved: fp_reconstructor_sino.raw\n");
//
//    cudaFree(d_vol);
//    cudaFree(d_sino);
//}


void test_fp_runner(cudaStream_t stream)
{
    printf("\n[ConeProjector] test\n");

    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.1f, vox_z = 0.1f;
    constexpr int   Na = 480, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    SCBCTParams params;
    params.iVX = Nx; params.iVY = Ny; params.iVZ = Nz;
    params.vox_x_mm = vox_xy; params.vox_z_mm = vox_z;
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;
    params.iPAng = Na;
    params.iPU = Nu; params.iPV = Nv;
    params.du_mm = du; params.dv_mm = dv;
    params.SID = SID; params.SDD = SDD;
    params.offsetU_mm = 0.f; params.offsetV_mm = 0.f;
    params.tiltu_angle_rad = 0.f;
    params.tiltn_angle_rad = 0.f;
    params.tiltv_angle_rad = 0.f;

    params.angle_list.resize(Na);
    for (int i = 0; i < Na; ++i)
        params.angle_list[i] = 2.f * CUDA_PI * i / 720;

    Mem::MemoryController mc;
    auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz);

    //std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    if (!read_raw_float((test_data_dir + "fdk_vec_vol_offline.raw").c_str(), h_vol.data(), 1LL * Nx * Ny * Nz)) {
        printf("  fdk_vec_vol_offline.raw not found, skip\n");
        return;
    }
    printf("  volume loaded\n");



    // 原来: cudaMalloc + cudaMemcpy (H2D)
    auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, /*deviceId=*/0);
    auto borrowed_vol = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz);
    mc.upload3D(d_vol, borrowed_vol);
    h_vol.reset();

    // 原来: cudaMalloc (sino)
    auto d_sino = mc.allocateDevice3D<float>(Nu, Nv, Na, /*deviceId=*/0);

    ConeProjector fpr;
    if (!fpr.init(params, ETask::FP_Joseph, 0)) {
        YK_LOGE("ConeProjector init failed");
        return;  // RAII 自动释放，不需要 cudaFree
    }

    // .data() 取裸指针，兼容现有 fpr.run 签名
    bool ok = fpr.run(d_vol.data(), params, d_sino.data(), stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    YK_LOGI("  fp_project: {}", ok ? "OK" : "FAILED");

    // 原来: cudaMemcpy (D2H)
    std::vector<float> h_sino((size_t)Na * Nv * Nu);
    auto borrowed_sino = mc.borrowCpu3D(h_sino.data(), Nu, Nv, Na);
    mc.download3D(borrowed_sino, d_sino);

    printStats(h_sino, "sino");
    write_raw_float((test_data_dir + "fp_reconstructor_sino.raw").c_str(),
        h_sino.data(), h_sino.size());
    printf("  saved: fp_reconstructor_sino.raw\n");

    // d_vol, d_sino 析构自动 cudaFree
}


// 测试每个视角有相同的geo offset
void test_fp_runner_fixed_offset(cudaStream_t stream)
{
    printf("\n[ConeProjectorEx] per-frame fixed offset test\n");

    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.1f, vox_z = 0.1f;
    constexpr int   Na = 480, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    SCBCTParams params;
    params.iVX = Nx; params.iVY = Ny; params.iVZ = Nz;
    params.vox_x_mm = vox_xy; params.vox_z_mm = vox_z;
    params.vol_offset_x_mm = params.vol_offset_y_mm = params.vol_offset_z_mm = 0.f;
    params.iPAng = Na;
    params.iPU = Nu; params.iPV = Nv;
    params.du_mm = du; params.dv_mm = dv;
    params.SID = SID; params.SDD = SDD;
    params.offsetU_mm = params.offsetV_mm = 0.f;

    params.angle_list.resize(Na);
    for (int i = 0; i < Na; ++i)
        params.angle_list[i] = 2.f * CUDA_PI * i / 720;

    // ---- 生成 per-frame 随机扰动 ----
    std::mt19937 rng(42);
    std::normal_distribution<float> dist_t(0.f, 0.25f);   // 平移 sigma=0.5mm
    std::normal_distribution<float> dist_r(0.f, 0.1f);   // 旋转 sigma=0.1deg

    std::vector<float3> src_offsets(Na), det_offsets(Na);
    std::vector<float3> detTilt_degs(Na), srcCRTilt_degs(Na);

    for (int i = 0; i < Na; ++i) {
        src_offsets[i] = make_float3(0.f, 0.f, 0.f);
        det_offsets[i] = make_float3(0.25f, 0.25f, 0.f);
        detTilt_degs[i] = make_float3(0.f, 0.f, 0.1f);
        srcCRTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
    }

    // ---- 预建 geometry ----
    std::vector<SConeProjGeomVec> h_views;
    build_circular_vec_geometry_perframe(
        h_views,
        params.angle_list,
        Na, Nu, Nv, du, dv, SID, SDD-SID,
        det_offsets, src_offsets,
        detTilt_degs, srcCRTilt_degs);

    // ---- 载入体数据 ----
    Mem::MemoryController mc;
    auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz);
    if (!read_raw_float((test_data_dir + "recon_raw_save.raw").c_str(),
        h_vol.data(), 1LL * Nx * Ny * Nz)) {
        printf("  recon_raw_save.raw not found, skip\n");
        return;
    }

    auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0);
    auto d_sino = mc.allocateDevice3D<float>(Nu, Nv, Na, 0);
    auto borrow = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz);
    mc.upload3D(d_vol, borrow);
    h_vol.reset();

    // ---- 正投影 ----
    ConeProjectorEx fpr;
    if (!fpr.init(params, ETask::FP_Joseph, 0)) {
        YK_LOGE("ConeProjectorEx init failed");
        return;
    }

    bool ok = fpr.run(d_vol.data(), params, h_views, d_sino.data(), stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    YK_LOGI("  fp_project perframe: {}", ok ? "OK" : "FAILED");

    std::vector<float> h_sino((size_t)Na * Nv * Nu);
    auto borrow_sino = mc.borrowCpu3D(h_sino.data(), Nu, Nv, Na);
    mc.download3D(borrow_sino, d_sino);
    printStats(h_sino, "sino_perframe");
    write_raw_float((test_data_dir + "fp_perframe_fixedoffset_sino.raw").c_str(),
        h_sino.data(), h_sino.size());
    printf("  saved: fp_perframe_fixedoffset_sino.raw\n");
}

// 测试每个视角有不同的geo offset
void test_fp_runner_random_offset(cudaStream_t stream)
{
    printf("\n[ConeProjectorEx] per-frame random offset test\n");

    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.1f, vox_z = 0.1f;
    constexpr int   Na = 480, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    SCBCTParams params;
    params.iVX = Nx; params.iVY = Ny; params.iVZ = Nz;
    params.vox_x_mm = vox_xy; params.vox_z_mm = vox_z;
    params.vol_offset_x_mm = params.vol_offset_y_mm = params.vol_offset_z_mm = 0.f;
    params.iPAng = Na;
    params.iPU = Nu; params.iPV = Nv;
    params.du_mm = du; params.dv_mm = dv;
    params.SID = SID; params.SDD = SDD;
    params.offsetU_mm = params.offsetV_mm = 0.f;

    params.angle_list.resize(Na);
    for (int i = 0; i < Na; ++i)
        params.angle_list[i] = 2.f * CUDA_PI * i / 720;

    // ---- 生成 per-frame 随机扰动 ----
    std::mt19937 rng(42);
    std::normal_distribution<float> dist_t(0.f, 0.25f);   // 平移 sigma=0.5mm
    std::normal_distribution<float> dist_r(0.f, 0.1f);   // 旋转 sigma=0.1deg

    std::vector<float3> src_offsets(Na), det_offsets(Na);
    std::vector<float3> detTilt_degs(Na), srcCRTilt_degs(Na);

    for (int i = 0; i < Na; ++i) {
        src_offsets[i] = make_float3(0.f, 0.f, 0.f);
        det_offsets[i] = make_float3(dist_t(rng), dist_t(rng), 0.f);
        detTilt_degs[i] = make_float3(0.f, 0.f, dist_r(rng));
        srcCRTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
    }




    // ---- 预建 geometry ----
    std::vector<SConeProjGeomVec> h_views;
    build_circular_vec_geometry_perframe(
        h_views,
        params.angle_list,
        Na, Nu, Nv, du, dv, SID, SDD-SID,
        det_offsets, src_offsets,
        detTilt_degs, srcCRTilt_degs);

    // ---- 载入体数据 ----
    Mem::MemoryController mc;
    auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz);
    if (!read_raw_float((test_data_dir + "recon_raw_save.raw").c_str(),
        h_vol.data(), 1LL * Nx * Ny * Nz)) {
        printf("  recon_raw_save.raw not found, skip\n");
        return;
    }

    auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0);
    auto d_sino = mc.allocateDevice3D<float>(Nu, Nv, Na, 0);
    auto borrow = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz);
    mc.upload3D(d_vol, borrow);
    h_vol.reset();

    // ---- 正投影 ----
    ConeProjectorEx fpr;
    if (!fpr.init(params, ETask::FP_Joseph, 0)) {
        YK_LOGE("ConeProjectorEx init failed");
        return;
    }

    bool ok = fpr.run(d_vol.data(), params, h_views, d_sino.data(), stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    YK_LOGI("  fp_project perframe: {}", ok ? "OK" : "FAILED");

    std::vector<float> h_sino((size_t)Na * Nv * Nu);
    auto borrow_sino = mc.borrowCpu3D(h_sino.data(), Nu, Nv, Na);
    mc.download3D(borrow_sino, d_sino);
    printStats(h_sino, "sino_perframe");
    write_raw_float((test_data_dir + "fp_perframe_offset_sino.raw").c_str(),
        h_sino.data(), h_sino.size());
    printf("  saved: fp_perframe_offset_sino.raw\n");
}


void test_fp_runner_periodic_offset(cudaStream_t stream)
{
    printf("\n[ConeProjectorEx] periodic offset FP\n");

    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 4.0f / 3.0f;
    params.SID = 500.f; params.SDD = 1000.f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f; params.vox_y_mm = 0.1f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = params.offsetV_mm = 0.f;
    params.vol_offset_x_mm = params.vol_offset_y_mm = params.vol_offset_z_mm = 0.f;

    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> angle_list(Na);
    for (int i = 0; i < Na; ++i)
        angle_list[i] = i * 2.f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    // 每6帧一个周期，最大偏移1mm，偏转为0
    std::vector<float3> src_offsets(Na), det_offsets(Na);
    std::vector<float3> detTilt_degs(Na), srcCRTilt_degs(Na);
    for (int i = 0; i < Na; ++i) {
        const float phase = 2.f * CUDA_PI * i / 360.f;
        const float jit = 0.25f * std::sin(phase);
        const float src_jit = 30 * std::sin(phase);
        src_offsets[i] = make_float3(0, src_jit, 0.f);
        det_offsets[i] = make_float3(0, 0, 0.f);
        detTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
        srcCRTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
    }

    std::vector<SConeProjGeomVec> h_views;
    build_circular_vec_geometry_perframe(
        h_views, angle_list,
        Na, params.iPU, params.iPV,
        params.du_mm, params.dv_mm,
        params.SID, params.SDD- params.SID,
        det_offsets, src_offsets,
        detTilt_degs, srcCRTilt_degs);

    Mem::MemoryController mc;
    auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz);
    if (!read_raw_float((test_data_dir + "recon_raw_save.raw").c_str(),
        h_vol.data(), 1LL * Nx * Ny * Nz)) {
        printf("  recon_raw_save.raw not found, skip\n");
        return;
    }
    auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0);
    auto d_sino = mc.allocateDevice3D<float>(params.iPU, params.iPV, Na, 0);
    {
        auto borrow = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz);
        mc.upload3D(d_vol, borrow);
    }
    h_vol.reset();

    ConeProjectorEx fpr;
    if (!fpr.init(params, ETask::FP_Joseph, 0)) {
        YK_LOGE("ConeProjectorEx init failed"); return;
    }
    bool ok = fpr.run(d_vol.data(), params, h_views, d_sino.data(), stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    YK_LOGI("  FP periodic offset: {}", ok ? "OK" : "FAILED");

    std::vector<float> h_sino(view_elems * Na);
    {
        auto borrow_sino = mc.borrowCpu3D(h_sino.data(), params.iPU, params.iPV, Na);
        mc.download3D(borrow_sino, d_sino);
    }
    printStats(h_sino, "sino_periodic");
    write_raw_float((test_data_dir + "fp_periodic_offset_sino.raw").c_str(),
        h_sino.data(), h_sino.size());
    printf("  saved: fp_periodic_offset_sino.raw\n");
}


