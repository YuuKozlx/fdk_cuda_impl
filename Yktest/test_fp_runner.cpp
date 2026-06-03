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
#include <BP/YkSiddonBPRunner.hpp>
#include <iter/YkOSSART.hpp>

using namespace YK;

const std::string test_data_dir = R"(H:\Code\fanproj\fdk-test\TestData\)";

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

void test_fp_runner_siddon_vs_joseph(cudaStream_t stream)
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
    if (!read_raw_float((test_data_dir + "fdk_vec_vol_offline.raw").c_str(),
        h_vol.data(), 1LL * Nx * Ny * Nz)) {
        printf("  fdk_vec_vol_offline.raw not found, skip\n");
        return;
    }
    printf("  volume loaded\n");

    auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0);
    {
        auto borrowed = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz);
        mc.upload3D(d_vol, borrowed);
    }
    h_vol.reset();

    auto d_sino_joseph = mc.allocateDevice3D<float>(Nu, Nv, Na, 0);
    auto d_sino_siddon = mc.allocateDevice3D<float>(Nu, Nv, Na, 0);

    // ---- Joseph FP ──────────────────────────────────────────────
    {
        ConeProjector fp;
        if (!fp.init(params, ETask::FP_Joseph, 0)) {
            YK_LOGE("ConeProjector(Joseph) init failed");
            return;
        }
        YK::Util::CudaTimer timer("joseph_fp", stream);
        bool ok = fp.run(d_vol.data(), params, d_sino_joseph.data(), stream);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        YK_LOGI("  joseph_fp: {}", ok ? "OK" : "FAILED");
    }
    {
        std::vector<float> h_sino((size_t)Na * Nv * Nu);
        auto borrowed = mc.borrowCpu3D(h_sino.data(), Nu, Nv, Na);
        mc.download3D(borrowed, d_sino_joseph);
        printStats(h_sino, "sino_joseph");
        write_raw_float((test_data_dir + "fp_joseph_sino.raw").c_str(),
            h_sino.data(), h_sino.size());
        printf("  saved: fp_joseph_sino.raw\n");
    }

    // ---- Siddon FP ──────────────────────────────────────────────
    {
        ConeProjector fp;
        if (!fp.init(params, ETask::FP_Siddon, 0)) {
            YK_LOGE("ConeProjector(Siddon) init failed");
            return;
        }
        YK::Util::CudaTimer timer("siddon_fp", stream);
        bool ok = fp.run(d_vol.data(), params, d_sino_siddon.data(), stream);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        YK_LOGI("  siddon_fp: {}", ok ? "OK" : "FAILED");
    }
    {
        std::vector<float> h_sino((size_t)Na * Nv * Nu);
        auto borrowed = mc.borrowCpu3D(h_sino.data(), Nu, Nv, Na);
        mc.download3D(borrowed, d_sino_siddon);
        printStats(h_sino, "sino_siddon");
        write_raw_float((test_data_dir + "fp_siddon_sino.raw").c_str(),
            h_sino.data(), h_sino.size());
        printf("  saved: fp_siddon_sino.raw\n");
    }

    // ---- 对比两者 ───────────────────────────────────────────────
    {
        std::vector<float> h_joseph((size_t)Na * Nv * Nu);
        std::vector<float> h_siddon((size_t)Na * Nv * Nu);
        {
            auto b1 = mc.borrowCpu3D(h_joseph.data(), Nu, Nv, Na);
            mc.download3D(b1, d_sino_joseph);
        }
        {
            auto b2 = mc.borrowCpu3D(h_siddon.data(), Nu, Nv, Na);
            mc.download3D(b2, d_sino_siddon);
        }
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

        double maxDiff = 0.0, mse = 0.0;
        double sum_joseph = 0.0, sum_siddon = 0.0;
        for (size_t i = 0; i < h_joseph.size(); ++i) {
            double diff = std::abs((double)h_joseph[i] - (double)h_siddon[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
            sum_joseph += h_joseph[i];
            sum_siddon += h_siddon[i];
        }
        mse /= (double)h_joseph.size();

        YK_LOGI("joseph sum = {:.6e}", sum_joseph);
        YK_LOGI("siddon sum = {:.6e}", sum_siddon);
        YK_LOGI("maxDiff    = {:.8f}", maxDiff);
        YK_LOGI("MSE        = {:.8e}", mse);

        double max_val = 0.0;
        for (size_t i = 0; i < h_joseph.size(); ++i)
            max_val = std::max(max_val, (double)std::abs(h_joseph[i]));

        YK_LOGI("max proj value   = {:.6f}", max_val);
        YK_LOGI("relative maxDiff = {:.4f}%", maxDiff / max_val * 100.0);
        YK_LOGI("sum ratio        = {:.6f}", sum_siddon / sum_joseph);
    }
}


// ================================================================
// test_fp_cylinder：用圆柱体模对比 Joseph vs Siddon FP
//
// 体模：均匀圆柱，轴沿 Z，半径 R_mm，衰减系数 mu
// 优点：
//   1. 投影解析值已知（弦长积分 = 2*sqrt(R²-d²)*mu，d=射线距轴距离）
//   2. 几何简单，边界清晰，一个像素的偏移立刻可见
//   3. 中心对称，便于目视检查
// ================================================================

void test_fp_cylinder_siddon_joseph(cudaStream_t stream)
{
    printf("\n[test_fp_cylinder] Joseph vs Siddon vs 解析值\n");

    // ── 几何参数（和你的实际设置一致）──────────────────────────
    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox = 0.1f;          // 各向同性体素 mm
    constexpr int   Na = 480;
    constexpr int   Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    // ── 圆柱参数 ────────────────────────────────────────────────
    constexpr float R_mm = 10.f;     // 圆柱半径 mm（约100个体素）
    constexpr float mu = 0.02f;    // 衰减系数（任意，解析值会乘它）

    SCBCTParams params;
    params.iVX = Nx; params.iVY = Ny; params.iVZ = Nz;
    params.vox_x_mm = params.vox_y_mm = params.vox_z_mm = vox;
    params.vol_offset_x_mm = params.vol_offset_y_mm = params.vol_offset_z_mm = 20.f;
    params.iPAng = Na;
    params.iPU = Nu; params.iPV = Nv;
    params.du_mm = du; params.dv_mm = dv;
    params.SID = SID; params.SDD = SDD;
    params.offsetU_mm = params.offsetV_mm = 0.f;
    params.tiltu_angle_rad = params.tiltn_angle_rad = params.tiltv_angle_rad = 0.f;
    params.scan_range_rad = 2.f * CUDA_PI;
    params.bShortScan = false;
    params.angle_list.resize(Na);
    for (int i = 0; i < Na; ++i)
        params.angle_list[i] = 2.f * CUDA_PI * i / Na;  // 均匀全圆

    Mem::MemoryController mc;

    // ── 生成圆柱体模（CPU）───────────────────────────────────────
    // 体积中心在世界原点，体素中心坐标：
    //   x_world = (ix - (Nx-1)/2) * vox
    //   y_world = (iy - (Ny-1)/2) * vox
    // 体素在圆柱内（x²+y² < R²）则为 mu，否则 0
    {
        printf("  生成圆柱体模 R=%.1fmm mu=%.4f ...\n", R_mm, mu);
        auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz);
        const float cx = (Nx - 1) * 0.5f * vox;   // 体积中心 X 世界坐标
        const float cy = (Ny - 1) * 0.5f * vox;
        for (int iz = 0; iz < Nz; ++iz)
            for (int iy = 0; iy < Ny; ++iy)
                for (int ix = 0; ix < Nx; ++ix)
                {
                    float wx = ix * vox - cx;
                    float wy = iy * vox - cy;
                    float r2 = wx * wx + wy * wy;
                    size_t idx = (size_t)iz * Ny * Nx + (size_t)iy * Nx + ix;
                    h_vol.data()[idx] = (r2 <= R_mm * R_mm) ? mu : 0.f;
                }

        auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0);
        {
            auto borrow = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz);
            mc.upload3D(d_vol, borrow);
        }

        // ── Joseph FP ──────────────────────────────────────────
        auto d_sino_joseph = mc.allocateDevice3D<float>(Nu, Nv, Na, 0);
        {
            ConeProjector fp;
            fp.init(params, ETask::FP_Joseph, 0);
            YK::Util::CudaTimer t("joseph_fp", stream);
            fp.run(d_vol.data(), params, d_sino_joseph.data(), stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // ── Siddon FP ──────────────────────────────────────────
        auto d_sino_siddon = mc.allocateDevice3D<float>(Nu, Nv, Na, 0);
        {
            ConeProjector fp;
            fp.init(params, ETask::FP_Siddon, 0);
            YK::Util::CudaTimer t("siddon_fp", stream);
            fp.run(d_vol.data(), params, d_sino_siddon.data(), stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // ── 下载并对比 ─────────────────────────────────────────
        const size_t sino_n = (size_t)Na * Nv * Nu;
        std::vector<float> h_j(sino_n), h_s(sino_n);
        {
            auto b = mc.borrowCpu3D(h_j.data(), Nu, Nv, Na);
            mc.download3D(b, d_sino_joseph);
        }
        {
            auto b = mc.borrowCpu3D(h_s.data(), Nu, Nv, Na);
            mc.download3D(b, d_sino_siddon);
        }
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

        // ── 计算解析值（中心角度，中心 V 行，扫描 U 方向）─────
        // 角度 0：源在 -Y 轴，射线沿 Y 方向，主轴 U 对应 X 方向
        // 射线到圆柱轴的距离 d = 探测器 U 偏移 × SID/SDD（放大率）
        // 解析弦长 = 2*sqrt(R²-d²)（d<R 时），投影值 = mu*弦长
        printf("\n  === 中心角度(ia=0)、中心 V 行，U 方向对比 ===\n");
        printf("  %6s  %10s  %10s  %10s  %10s  %10s\n",
            "iu", "analytic", "joseph", "siddon", "j-ana", "s-ana");

        const int ia = 0;            // 角度 0
        const int iv_c = Nv / 2;    // 中心 V 行
        const float mag = SDD / SID; // 放大倍数（源到等中心 / 源到探测器）

        // 只打印圆柱覆盖范围附近（中心 ± 几百像素）
        const int iu_center = Nu / 2;
        const int half_range = (int)(R_mm / du * mag) + 20;  // 多打 20 像素
        const int iu_lo = max(0, iu_center - half_range);
        const int iu_hi = min(Nu - 1, iu_center + half_range);

        double max_j_err = 0, max_s_err = 0;
        double sum_j = 0, sum_s = 0, sum_ana = 0;

        for (int iu = iu_lo; iu <= iu_hi; ++iu)
        {
            // 探测器 U 坐标（相对探测器中心，mm）
            float det_u_mm = (iu - (Nu - 1) * 0.5f) * du;
            // 等中心处的横向距离（射线到圆柱轴的垂直距离）
            float d_mm = det_u_mm * (SID / SDD);
            // 解析投影值
            float ana = 0.f;
            if (fabsf(d_mm) < R_mm)
                ana = 2.f * sqrtf(R_mm * R_mm - d_mm * d_mm) * mu;

            size_t idx = ((size_t)ia * Nv + iv_c) * Nu + iu;
            float jv = h_j[idx];
            float sv = h_s[idx];

            double j_err = (double)jv - (double)ana;
            double s_err = (double)sv - (double)ana;
            max_j_err = max(max_j_err, fabs(j_err));
            max_s_err = max(max_s_err, fabs(s_err));
            sum_j += jv; sum_s += sv; sum_ana += ana;

            // 只打印圆柱边缘附近（|d_mm| > R-2mm）和中心，减少输出
            if (fabsf(fabsf(d_mm) - R_mm) < 2.f || fabsf(d_mm) < 1.f)
                printf("  iu=%4d d=%6.2fmm  ana=%8.5f  j=%8.5f  s=%8.5f"
                    "  j-a=%+8.5f  s-a=%+8.5f\n",
                    iu, d_mm, ana, jv, sv, (float)j_err, (float)s_err);
        }

        printf("\n  [ia=0 全 U 范围统计]\n");
        printf("  max|joseph-analytic| = %.6f\n", max_j_err);
        printf("  max|siddon-analytic| = %.6f\n", max_s_err);
        printf("  sum analytic = %.6e\n", sum_ana);
        printf("  sum joseph   = %.6e  ratio=%.6f\n", sum_j, sum_j / sum_ana);
        printf("  sum siddon   = %.6e  ratio=%.6f\n", sum_s, sum_s / sum_ana);

        // ── 全局 Joseph vs Siddon 统计 ─────────────────────────
        printf("\n  [全局 Joseph vs Siddon]\n");
        double mse = 0, maxDiff = 0;
        double sj = 0, ss = 0;
        for (size_t i = 0; i < sino_n; ++i) {
            double d = (double)h_j[i] - (double)h_s[i];
            mse += d * d;
            maxDiff = max(maxDiff, fabs(d));
            sj += h_j[i]; ss += h_s[i];
        }
        mse /= (double)sino_n;
        printf("  joseph sum = %.6e\n", sj);
        printf("  siddon sum = %.6e  ratio=%.6f\n", ss, ss / sj);
        printf("  maxDiff    = %.8f\n", maxDiff);
        printf("  MSE        = %.8e\n", mse);

        // ── 存盘（可用 ImageJ 看投影图对比）────────────────────
        write_raw_float((test_data_dir + "cylinder_joseph.raw").c_str(),
            h_j.data(), sino_n);
        write_raw_float((test_data_dir + "cylinder_siddon.raw").c_str(),
            h_s.data(), sino_n);
        printf("\n  saved: cylinder_joseph.raw / cylinder_siddon.raw\n");
        printf("  ImageJ: 32-bit Real, W=%d H=%d imgs=%d\n", Nu, Nv, Na);
        printf("  看单帧(ia=0)中心行，Joseph 和 Siddon 应对齐，对比解析值\n");
    }
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
        Na, Nu, Nv, du, dv, SID, SDD - SID,
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
        Na, Nu, Nv, du, dv, SID, SDD - SID,
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
        params.SID, params.SDD - params.SID,
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


void test_periodic_fp_ideal_recon(cudaStream_t stream)
{
    auto params = make_default_params();
    auto fp_cfg = make_periodic_config(params, 0.f, 30.f, 0.f, 0.f, 0.f, 0.f, 360.f);
    auto recon_cfg = make_ideal_config(params);
    run_fp(params, fp_cfg, "fp_periodic_sino.raw", "recon_raw_save.raw", stream);
    run_recon(params, recon_cfg, "fp_periodic_sino.raw", "recon_periodic_ideal.raw", stream);
}

void test_periodic_fp_corrected_recon(cudaStream_t stream)
{
    auto params = make_default_params();
    auto fp_cfg = make_periodic_config(params, 0.f, 0.f, 0.f, 0.f, 30.f, 0.f, 360.f);
    // sino 已经存在，直接用同样的几何重建，不需要重新投影
    run_recon(params, fp_cfg, "fp_periodic_sino.raw", "recon_periodic_corrected.raw", stream);
}

void test_random_fp_ideal_recon(cudaStream_t stream)
{
    auto params = make_default_params();
    auto fp_cfg = make_random_config(params, 0.25f, 0.1f, 42);
    auto recon_cfg = make_ideal_config(params);
    run_fp(params, fp_cfg, "fp_random_sino.raw", "recon_raw_save.raw", stream);
    run_recon(params, recon_cfg, "fp_random_sino.raw", "recon_random_ideal.raw", stream);
}

void test_fixed_fp_ideal_recon(cudaStream_t stream)
{
    auto params = make_default_params();
    auto fp_cfg = make_fixed_offset_config(params, make_float3(0.f, 30.f, 0.f), make_float3(0.f, 0.f, 0.f));
    auto recon_cfg = make_ideal_config(params);
    run_fp(params, fp_cfg, "fp_fixed_sino.raw", "recon_raw_save.raw", stream);
    run_recon(params, recon_cfg, "fp_fixed_sino.raw", "recon_fixed_ideal.raw", stream);
}

void test_fixed_fp_corrected_recon(cudaStream_t stream)
{
    auto params = make_default_params();
    auto fp_cfg = make_fixed_offset_config(params, make_float3(0.f, 30.f, 0.f), make_float3(0.f, 0.f, 0.f));
    // sino 已经存在，直接用相同的几何重建，不需要重新投影
    run_recon(params, fp_cfg, "fp_fixed_sino.raw", "recon_fixed_corrected.raw", stream);
}

void test_insufficient_angle_fp_recon(cudaStream_t stream)
{
    auto params = make_default_params();

    // 实际扫描只有238°，但重建时按240°处理
    constexpr float actual_range_deg = 238.f;
    constexpr float assumed_range_deg = 240.f;

    const float actual_range_rad = actual_range_deg * CUDA_PI / 180.f;
    const float assumed_range_rad = assumed_range_deg * CUDA_PI / 180.f;

    // 正投影：按实际238°生成angle_list
    params.angle_list.resize(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        params.angle_list[i] = i * actual_range_rad / (params.iPAng - 1);

    // 重建params：角度列表和scan_range都认为是240°
    SCBCTParams recon_params = params;
    recon_params.scan_range_rad = assumed_range_rad;
    for (int i = 0; i < recon_params.iPAng; ++i)
        recon_params.angle_list[i] = i * assumed_range_rad / (recon_params.iPAng - 1);

    auto fp_cfg = make_ideal_config(params);       // 用实际238°几何投影
    auto recon_cfg = make_ideal_config(recon_params); // 用假设240°几何重建

    run_fp(params, fp_cfg, "fp_insufficient_angle_sino.raw", "recon_raw_save.raw", stream);
    run_recon(recon_params, recon_cfg, "fp_insufficient_angle_sino.raw", "recon_insufficient_angle.raw", stream);
}



void test_flat_detector_roty_fp(cudaStream_t stream)
{
    auto params = make_default_params();
    params.SDD = 300; params.SID = 200;
    params.iPAng = 180; params.iPAngTotal = 180;
    params.iVY = 100; params.iVZ = 512;
    params.vox_x_mm = params.vox_y_mm = params.vox_z_mm = 0.2f;
    params.bShortScan = false;
    params.scan_range_rad = 2.f * CUDA_PI;
    params.angle_list.resize(180);
    for (int i = 0; i < 180; ++i)
        params.angle_list[i] = i * 2.f * CUDA_PI / 180;
    params.scan_start_angle_rad = params.angle_list[0];

    constexpr float R = 125.f;
    std::vector<SConeProjGeomVec> h_views;
    build_planar_ct_vec_geometry(
        h_views, params.angle_list,
        params.iPAng, params.iPU, params.iPV,
        params.du_mm, params.dv_mm,
        params.SID, params.SDD - params.SID, R);




    run_fp(params, h_views, "fp_flat_det_roty_sino.raw", "pcb_phantom.raw", stream);
    params.iVX = 512;
    params.iVY = 100;
    params.iVZ = 512;
    params.vox_x_mm = 0.2f;
    params.vox_y_mm = 0.04f;
    params.vox_z_mm = 0.2f;
    // 重建用理想圆轨迹几何，但所有角度都设为0
    // 等效于：所有帧都认为是从同一个0°位置投影的
    SCBCTParams recon_params = params;
    std::fill(recon_params.angle_list.begin(), recon_params.angle_list.end(), 0.f);
    recon_params.scan_start_angle_rad = 0.f;
    run_recon(params, h_views, "fp_flat_det_roty_sino.raw",
        "recon_flat_det_roty.raw", stream);
}


void test_flat_detector_roty_fp_ellipse(cudaStream_t stream)
{
    auto params = make_default_params();
    params.SDD = 300; params.SID = 200;
    params.iPAng = 720; params.iPAngTotal = 720;
    params.scan_range_rad = 2.f * CUDA_PI;
    params.bShortScan = false;

    // 探测器参数
    params.iPU = 1024; params.iPV = 1024;
    params.du_mm = 0.1f; params.dv_mm = 0.1f;

    // 体素网格（正投影与重建共用同一套体素参数）
    params.iVX = 512; params.iVY = 100; params.iVZ = 512;
    params.vox_x_mm = 0.05f; params.vox_y_mm = 0.05f; params.vox_z_mm = 0.05f;

    // 角度列表（0 ~ 2π，720个角度）
    params.angle_list.resize(720);
    for (int i = 0; i < 720; ++i)
        params.angle_list[i] = i * 2.f * CUDA_PI / 720;
    params.scan_start_angle_rad = params.angle_list[0];

    // ========= 椭圆轨迹正投影 =========
    constexpr float a = 55.f;   // X轴半长
    constexpr float b = 54.9f;   // Z轴半长（接近圆形，但不等）
    std::vector<SConeProjGeomVec> h_views_ellipse;
    build_planar_ct_vec_geometry_ellipse(
        h_views_ellipse, params.angle_list,
        params.iPAng, params.iPU, params.iPV,
        params.du_mm, params.dv_mm,
        params.SID, params.SDD - params.SID,
        a, b);

    // 执行正投影（使用椭圆几何）
    run_fp(params, h_views_ellipse, "fp_ellipse_sino.raw", "pcb_phantom.raw", stream);

    // ========= 圆形轨迹重建 =========
    constexpr float R = a;   // 圆形半径（取椭圆的X半轴）
    std::vector<SConeProjGeomVec> h_views_circle;
    build_planar_ct_vec_geometry(
        h_views_circle, params.angle_list,
        params.iPAng, params.iPU, params.iPV,
        params.du_mm, params.dv_mm,
        params.SID, params.SDD - params.SID, R);


    // 重建参数（沿用正投影的体素网格和角度列表）
    SCBCTParams recon_params = params;  // 拷贝所有参数
    // 注意：recon_params.angle_list 已经正确，无需修改

    // 使用圆形几何重建椭圆正弦图
    run_recon(params, h_views_circle, "fp_ellipse_sino.raw",
        "recon_ellipse2circle.raw", stream);

    // （可选）使用椭圆几何重建椭圆正弦图（自洽重建，作为对比）
    run_recon(params, h_views_ellipse, "fp_ellipse_sino.raw",
        "recon_ellipse2ellipse.raw", stream);

    // 比较 h_views_circle 和 h_views_ellipse 的差异，验证几何构建函数的正确性
    //struct alignas(16) SConeProjGeomVec {
    //    float4 src;     // xyz = source position, w = unused
    //    float4 srcCR;   // xyz = center ray direction, w = unused
    //    float4 detS;    // xyz = detector (0,0) position, w = unused
    //    float4 detU;    // xyz = per-pixel U vector, w = unused
    //    float4 detV;    // xyz = per-pixel V vector, w = unused
    //    float4 angle;   // x = gantry angle, y/z = reserved, w = unused
    //};

    for (int i = 0; i < params.iPAng; ++i) {
        const auto& v_circle = h_views_circle[i];
        const auto& v_ellipse = h_views_ellipse[i];
        // 比较 src_pos

        float src_diff = length(make_float3(v_circle.src) - make_float3(v_ellipse.src));
        float det_diff = length(make_float3(v_circle.detS) - make_float3(v_ellipse.detS));
        float u_diff = length(make_float3(v_circle.detU) - make_float3(v_ellipse.detU));
        float v_diff = length(make_float3(v_circle.detV) - make_float3(v_ellipse.detV));
        float angle_diff = std::abs(v_circle.angle.x - v_ellipse.angle.x);
        float srcCR_diff = length(make_float3(v_circle.srcCR) - make_float3(v_ellipse.srcCR));

        printf("View %3d: src_diff=%.3fmm, det_diff=%.3fmm, u_diff=%.3fmm, v_diff=%.3fmm, angle_diff=%.3fdeg, srcCR_diff=%.6f\n",
            i, src_diff, det_diff, u_diff, v_diff, angle_diff * 180.f / CUDA_PI, srcCR_diff);

    }
}


void test_generate_pcb_phantom()
{
    generate_pcb_phantom(test_data_dir + "pcb_phantom.raw");
}



