#include "test_common.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "FP/YkFPRunner.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "YKCBCT/interface/YkTaskTypes.hpp"

using namespace YK;

void test_fp_runner(cudaStream_t stream)
{
    printf("\n[FpReconstructor] test\n");

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

    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    if (!read_raw_float("fdk_vec_vol_offline.raw", h_vol)) {
        printf("  fdk_vec_vol_offline.raw not found, skip\n");
        return;
    }
    printf("  volume loaded\n");

    float* d_vol = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                             h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));
    h_vol.clear();

    const size_t sino_elems = (size_t)Na * Nv * Nu;
    float* d_sino = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));

    FpReconstructor fpr;
    if (!fpr.init(params, ETask::FP_Joseph, 0)) {
        YK_LOGE("FpReconstructor init failed");
        cudaFree(d_vol); cudaFree(d_sino);
        return;
    }

    bool ok = fpr.run(d_vol, params, d_sino, stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    YK_LOGI("  fp_project: {}", ok ? "OK" : "FAILED");

    std::vector<float> h_sino(sino_elems);
    YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                             sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

    printStats(h_sino, "sino");
    write_raw_float("fp_reconstructor_sino.raw", h_sino.data(), sino_elems);
    printf("  saved: fp_reconstructor_sino.raw\n");

    cudaFree(d_vol);
    cudaFree(d_sino);
}
