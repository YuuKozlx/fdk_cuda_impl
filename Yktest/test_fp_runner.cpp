#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include "test_common.hpp"

#include <global/YkMem3d.hpp>
#include "FP/YkFPRunner.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"

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