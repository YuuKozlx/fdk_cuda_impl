#include "FP/YkFPRunner.hpp"
#include "global/YkCBCTParams.h"
#include <fstream>
#include <vector>
#include <cstdio>
#include "global/YkMacro.hpp"
#include "Fp/YkFPRunner.hpp"

// ----------------------------------------------------------------
// 读取 raw 体积文件
// ----------------------------------------------------------------
namespace fptest {
    static bool read_raw_float(const char* path, std::vector<float>& data) {
        FILE* fp = std::fopen(path, "rb");
        if (!fp) return false;
        size_t n = std::fread(data.data(), sizeof(float), data.size(), fp);
        std::fclose(fp);
        return n == data.size();
    }

    static bool write_raw_float(const char* path, const std::vector<float>& data) {
        FILE* fp = std::fopen(path, "wb");
        if (!fp) return false;
        size_t n = std::fwrite(data.data(), sizeof(float), data.size(), fp);
        std::fclose(fp);
        return n == data.size();
    }


    static bool write_raw_float(const char* path, const float* data, uint64_t element_count) {
        FILE* fp = std::fopen(path, "wb");
        if (!fp) return false;
        size_t n = std::fwrite(data, sizeof(float), element_count, fp);
        std::fclose(fp);
        return n == element_count;
    }
}


int main11()
{
    // ----------------------------------------------------------------
    // 体积参数
    // ----------------------------------------------------------------
    constexpr int   Nx = 512;
    constexpr int   Ny = 512;
    constexpr int   Nz = 400;
    constexpr float vox_xy = 0.25f;   // mm
    constexpr float vox_z = 0.25f;   // mm
    // 体素中心偏置（根据实际情况填，暂填 0）
    constexpr float offset_x = 0.f;
    constexpr float offset_y = 0.f;
    constexpr float offset_z = 0.f;

    // ----------------------------------------------------------------
    // 扫描几何参数
    // ----------------------------------------------------------------
    constexpr int   Na = 360;
    constexpr int   Nu = 1024;
    constexpr int   Nv = 1024;    // 题目给 1024，正方形探测器
    constexpr float du = 0.25f;   // mm
    constexpr float dv = 0.25f;   // mm
    constexpr float SID = 500.f;   // mm
    constexpr float SDD = 1000.f;  // mm

    // ----------------------------------------------------------------
    // 构建 SCBCTParams
    // ----------------------------------------------------------------
    SCBCTParams params;
    params.iPU = Nu;
    params.iPV = Nv;
    params.iPAng = Na;
    params.iPAngTotal = Na;
    params.du_mm = du;
    params.dv_mm = dv;
    params.offsetU_mm = 0.f;
    params.offsetV_mm = 0.f;
    params.SID = SID;
    params.SDD = SDD;
    params.scan_range_rad = 2.f * CUDA_PI;
    params.scan_start_angle_rad = 0.f;
    params.bShortScan = false;
    params.iVX = Nx;
    params.iVY = Ny;
    params.iVZ = Nz;
    params.vox_xy_mm = vox_xy;
    params.vox_z_mm = vox_z;
    params.vol_offset_x_mm = offset_x;
    params.vol_offset_y_mm = offset_y;
    params.vol_offset_z_mm = offset_z;

    // 均匀角度列表 [0, 2pi)
    params.angle_list.resize(Na);
    for (int i = 0; i < Na; ++i)
        params.angle_list[i] = 2.f * CUDA_PI * i / Na;

    // ----------------------------------------------------------------
    // 读取体积
    // ----------------------------------------------------------------
    const size_t vol_count = (size_t)Nx * Ny * Nz;
    std::vector<float> h_vol(vol_count);
    fptest::read_raw_float("fdk_vec_vol_online.raw", h_vol);
    if (h_vol.empty()) return -1;

    printf("[test] volume loaded: %dx%dx%d = %zu voxels\n",
        Nx, Ny, Nz, vol_count);

    // ----------------------------------------------------------------
    // 上传体积到 device
    // ----------------------------------------------------------------
    float* d_vol = nullptr;
    cudaMalloc(&d_vol, vol_count * sizeof(float));
    cudaMemcpy(d_vol, h_vol.data(),
        vol_count * sizeof(float), cudaMemcpyHostToDevice);
    h_vol.clear();   // host 内存释放，不再需要

    // ----------------------------------------------------------------
    // 分配 sinogram 输出（device）
    // ----------------------------------------------------------------
    const size_t sino_count = (size_t)Na * Nv * Nu;
    float* d_sino = nullptr;
    cudaMalloc(&d_sino, sino_count * sizeof(float));
    cudaMemset(d_sino, 0, sino_count * sizeof(float));

    // ----------------------------------------------------------------
    // 执行正投
    // ----------------------------------------------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    constexpr int Kchunk = 64;

    YK::Fp::FpReconstructor fp;
    if (!fp.init(params, Kchunk, stream)) {
        fprintf(stderr, "[test] FpReconstructor init failed\n");
        return -1;
    }
    if (!fp.bindVolume(d_vol)) {
        fprintf(stderr, "[test] bindVolume failed\n");
        return -1;
    }

    printf("[test] running forward projection...\n");
    if (!fp.run(d_sino, stream)) {
        fprintf(stderr, "[test] fp.run failed\n");
        return -1;
    }
    cudaStreamSynchronize(stream);
    printf("[test] forward projection done\n");

    // ----------------------------------------------------------------
    // 下载并保存 sinogram
    // ----------------------------------------------------------------
    std::vector<float> h_sino(sino_count);
    cudaMemcpy(h_sino.data(), d_sino,
        sino_count * sizeof(float), cudaMemcpyDeviceToHost);

    fptest::read_raw_float("fp_sino_out.raw", h_sino);
    printf("[test] sinogram saved: fp_sino_out.raw  [%d x %d x %d]  Na x Nv x Nu\n",
        Na, Nv, Nu);

    // ----------------------------------------------------------------
    // 清理
    // ----------------------------------------------------------------
    fp.release();
    cudaFree(d_vol);
    cudaFree(d_sino);
    cudaStreamDestroy(stream);

    return 0;
}