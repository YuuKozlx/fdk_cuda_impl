#include "test_common.hpp"
#include "Iter/YkOSSART.hpp"
#include "Iter/YkSART.hpp"

#include <vector>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include "common/YkVecGeo.hpp"
#include "util/YkCudatimer.hpp"
#include <string>
#include <cuda_runtime.h>
#include "global/YkLog.h"
#include "global/YkCBCTParams.h"
#include "global/YkMacro.hpp"


using namespace YK::Mem;

using namespace YK;

const std::string test_data_dir = R"(H:\Code\fanproj\fdk-test\TestData\)";

// ----------------------------------------------------------------
// main_ossart_test：OS-SART 迭代重建测试
// ----------------------------------------------------------------
int main_ossart_test()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 360; params.iPAngTotal = 360;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 3.0f / 3.0f;
    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f; params.vox_y_mm = 0.1f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)CUDA_PI / 720;  // ← 720 不是 480
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // ---- 读测量正弦图 ────────────────────────────────────────────
    std::vector<float> h_sino(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read proj_1024x1024x480.raw");
        return -1;
    }
    YK_LOGI("sino loaded");

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    // 上传正弦图
    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    {
        auto h_buf = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        std::memcpy(h_buf.data(), h_sino.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_sino, h_buf);
    }

    // 分配输出体积（初始全零）
    auto d_vol = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol.data(), 0, vol_elems * sizeof(float), s));

    // ---- OS-SART 配置 ───────────────────────────────────────────
    OSSART::Config cfg;
    cfg.n_iter = 10;
    cfg.n_subset = 20;
    cfg.lambda = 1.f;
    cfg.lambda_red = 0.999f;
    cfg.eps = 1e-6f;
    cfg.use_min = true;
    cfg.min_constraint = 0.f;     // CT 值非负约束
    cfg.fp_task = ETask::FP_Joseph;       // Joseph 正投影
    cfg.bp_task = ETask::BP_Joseph_v3;    // Joseph 反投影（接近 FP_Joseph 的伴随）

    // ---- 运行 OS-SART ──────────────────────────────────────────

    {
        OSSART recon;
        if (!recon.init(params, cfg, s)) {
            YK_LOGE("OSSART init failed");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }

        YK_LOGI("OSSART start: {} iters x {} subsets = {} updates",
            cfg.n_iter, cfg.n_subset, cfg.n_iter * cfg.n_subset);

        YK::Util::CudaTimer timer("ossart_total", s);
        recon.run(d_sino.data(), d_vol.data(), params, s);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        YK_LOGI("OSSART done, total iterations = {}", recon.totalIterations());
        recon.release();
    }

    // ---- 保存结果 ───────────────────────────────────────────────
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol);

        float minv = *std::min_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        YK_LOGI("vol stats: min={:.6f} max={:.6f}", minv, maxv);

        write_raw_float((test_data_dir + "ossart_vol_joseph_v3.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: ossart_vol.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- 与 FDK 结果对比（可选）────────────────────────────────
    // 如果有 fdk_vol.raw，可以计算 MSE 作为参考
    {
        std::vector<float> h_fdk(vol_elems);
        if (read_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_fdk)) {
            auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
            ctrl.download3D(h_vol, d_vol);

            double mse = 0.0, sum_ossart = 0.0, sum_fdk = 0.0;
            for (size_t i = 0; i < vol_elems; ++i) {
                double diff = (double)h_vol.cdata()[i] - (double)h_fdk[i];
                mse += diff * diff;
                sum_ossart += h_vol.cdata()[i];
                sum_fdk += h_fdk[i];
            }
            mse /= (double)vol_elems;
            YK_LOGI("FDK  sum = {:.6e}", sum_fdk);
            YK_LOGI("OSSART sum = {:.6e}", sum_ossart);
            YK_LOGI("MSE(OSSART vs FDK) = {:.6e}", mse);
        }
        else {
            YK_LOGI("verify_vol_fdk.raw not found, skip FDK comparison");
        }
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

int main_ossart_ex_test()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 360; params.iPAngTotal = 360;
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

    auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

    // ---- 构建几何 ────────────────────────────────────────────────
    std::vector<SConeProjGeomVec> h_views(params.iPAng);
    build_circular_vec_geometry_from_theta(
        h_views, params.angle_list, params.iPAng,
        params.iPU, params.iPV,
        params.du_mm, params.dv_mm,
        params.SID, params.SDD - params.SID,
        f3(params.offsetU_mm, params.offsetV_mm, 0.f),
        f3(rad2deg(params.tiltu_angle_rad),
            rad2deg(params.tiltn_angle_rad),
            rad2deg(params.tiltv_angle_rad)));

    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // ---- 读测量正弦图 ────────────────────────────────────────────
    std::vector<float> h_sino(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }
    YK_LOGI("sino loaded");

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    {
        auto h_buf = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        std::memcpy(h_buf.data(), h_sino.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_sino, h_buf);
    }

    auto d_vol = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol.data(), 0, vol_elems * sizeof(float), s));

    // ---- OS-SART 配置 ───────────────────────────────────────────
    OSSARTEx::Config cfg;
    cfg.n_iter = 10;
    cfg.n_subset = 20;
    cfg.lambda = 1.f;
    cfg.lambda_red = 1.f;
    cfg.eps = 1e-6f;
    cfg.use_min = false;
    cfg.min_constraint = 0.f;
    cfg.fp_task = ETask::FP_Joseph;       // Joseph 正投影
    cfg.bp_task = ETask::BP_FDK;    // Joseph 反投影（接近 FP_Joseph 的伴随）

    // ---- 运行 OS-SART ───────────────────────────────────────────
    {
        OSSARTEx recon;
        if (!recon.init(params, cfg, h_views, s)) {
            YK_LOGE("OSSARTEx init failed");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }

        YK_LOGI("OSSARTEx start: {} iters x {} subsets = {} updates",
            cfg.n_iter, cfg.n_subset, cfg.n_iter * cfg.n_subset);

        YK::Util::CudaTimer timer("ossart_total", s);
        recon.run(d_sino.data(), d_vol.data(), s);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        YK_LOGI("OSSARTEx done, total iterations = {}", recon.totalIterations());
        recon.release();
    }

    // ---- 保存结果 ───────────────────────────────────────────────
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol);

        float minv = *std::min_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        YK_LOGI("vol stats: min={:.6f} max={:.6f}", minv, maxv);

        write_raw_float((test_data_dir + "ossart_vol.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: ossart_vol.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- 与 FDK 结果对比（可选）────────────────────────────────
    {
        std::vector<float> h_fdk(vol_elems);
        if (read_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_fdk)) {
            auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
            ctrl.download3D(h_vol, d_vol);

            double mse = 0.0, sum_ossart = 0.0, sum_fdk = 0.0;
            for (size_t i = 0; i < vol_elems; ++i) {
                double diff = (double)h_vol.cdata()[i] - (double)h_fdk[i];
                mse += diff * diff;
                sum_ossart += h_vol.cdata()[i];
                sum_fdk += h_fdk[i];
            }
            mse /= (double)vol_elems;
            YK_LOGI("FDK  sum = {:.6e}", sum_fdk);
            YK_LOGI("OSSART sum = {:.6e}", sum_ossart);
            YK_LOGI("MSE(OSSART vs FDK) = {:.6e}", mse);
        }
        else {
            YK_LOGI("fdk_vec_vol_online.raw not found, skip FDK comparison");
        }
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

// ----------------------------------------------------------------
// main_iter_recon_sim：模拟测试
// 流程：读体积 → Joseph FP 生成正弦图 → OSSART/SIRT 迭代重建 → 对比原始体积
// ----------------------------------------------------------------
int main_iter_recon_sim()
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

    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    // ---- 读参考体积（ground truth）────────────────────────────
    auto d_vol_gt = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    auto d_vol_rec = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    {
        std::vector<float> h_vol_gt(vol_elems);
        if (!read_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_vol_gt)) {
            YK_LOGE("cannot read fdk_vec_vol_online.raw");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        YK_LOGI("ground truth loaded");

        auto h_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        std::memcpy(h_buf.data(), h_vol_gt.data(), vol_elems * sizeof(float));
        ctrl.upload3D(d_vol_gt, h_buf);

        float minv = *std::min_element(h_vol_gt.begin(), h_vol_gt.end());
        float maxv = *std::max_element(h_vol_gt.begin(), h_vol_gt.end());
        YK_LOGI("gt stats: min={:.6f} max={:.6f}", minv, maxv);
    }

    // ---- Step1：Joseph FP 生成正弦图 ───────────────────────────
    {
        ConeProjector fp;
        fp.init(params, ETask::FP_Joseph);
        YK_CUDA_CHECK(cudaMemsetAsync(d_sino.data(), 0,
            proj_elems * sizeof(float), s));
        YK::Util::CudaTimer timer("fp_generate_sino", s);
        fp.run(d_vol_gt.data(), params, d_sino.data(), s);
    }
    cudaStreamSynchronize(s);
    {
        auto h_sino = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        ctrl.download3D(h_sino, d_sino);
        float minv = *std::min_element(h_sino.cdata(), h_sino.cdata() + proj_elems);
        float maxv = *std::max_element(h_sino.cdata(), h_sino.cdata() + proj_elems);
        YK_LOGI("sino stats: min={:.6f} max={:.6f}", minv, maxv);
        write_raw_float((test_data_dir + "sim_sino.raw").c_str(),
            h_sino.cdata(), proj_elems);
        YK_LOGI("saved: sim_sino.raw");
    }

    // ---- Step2：OS-SART 重建 ────────────────────────────────────
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol_rec.data(), 0,
        vol_elems * sizeof(float), s));
    {
        OSSART::Config cfg;
        cfg.n_iter = 5;
        cfg.n_subset = 20;
        cfg.lambda = 1.0f;
        cfg.eps = 1e-6f;
        cfg.use_min = true;
        cfg.min_constraint = 0.f;
        cfg.fp_task = ETask::FP_Joseph;
        cfg.bp_task = ETask::BP_Siddon_VoxDriven;

        OSSART recon;
        if (!recon.init(params, cfg, s)) {
            YK_LOGE("OSSART init failed");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }

        YK_LOGI("OSSART start: {} iters x {} subsets = {} updates",
            cfg.n_iter, cfg.n_subset, cfg.n_iter * cfg.n_subset);

        YK::Util::CudaTimer timer("ossart_recon", s);
        recon.run(d_sino.data(), d_vol_rec.data(), params, s);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        YK_LOGI("OSSART done, total iterations = {}", recon.totalIterations());
        recon.release();
    }
    {
        auto h_vol_rec = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_rec, d_vol_rec);
        float minv = *std::min_element(h_vol_rec.cdata(), h_vol_rec.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol_rec.cdata(), h_vol_rec.cdata() + vol_elems);
        YK_LOGI("ossart vol stats: min={:.6f} max={:.6f}", minv, maxv);
        write_raw_float((test_data_dir + "sim_ossart_vol.raw").c_str(),
            h_vol_rec.cdata(), vol_elems);
        YK_LOGI("saved: sim_ossart_vol.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- Step3：SIRT 重建 ───────────────────────────────────────
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol_rec.data(), 0,
        vol_elems * sizeof(float), s));
    {
        SIRT::Config cfg;
        cfg.n_iter = 20;
        cfg.lambda = 1.0f;
        cfg.eps = 1e-6f;
        cfg.use_min = true;
        cfg.min_constraint = 0.f;
        cfg.fp_task = ETask::FP_Joseph;
        cfg.bp_task = ETask::BP_Siddon_VoxDriven;

        SIRT recon;
        if (!recon.init(params, cfg, s)) {
            YK_LOGE("SIRT init failed");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }

        YK_LOGI("SIRT start: {} iters", cfg.n_iter);

        YK::Util::CudaTimer timer("sirt_recon", s);
        recon.run(d_sino.data(), d_vol_rec.data(), params, s);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        YK_LOGI("SIRT done");
        recon.release();
    }
    {
        auto h_vol_rec = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_rec, d_vol_rec);
        float minv = *std::min_element(h_vol_rec.cdata(), h_vol_rec.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol_rec.cdata(), h_vol_rec.cdata() + vol_elems);
        YK_LOGI("sirt vol stats: min={:.6f} max={:.6f}", minv, maxv);
        write_raw_float((test_data_dir + "sim_sirt_vol.raw").c_str(),
            h_vol_rec.cdata(), vol_elems);
        YK_LOGI("saved: sim_sirt_vol.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- Step4：对比 GT ─────────────────────────────────────────
    {
        std::vector<float> h_gt(vol_elems);
        read_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_gt);

        auto compare = [&](const char* name, const char* fname) {
            std::vector<float> h_rec(vol_elems);
            if (!read_raw_float(fname, h_rec)) return;
            double mse = 0.0, sum_rec = 0.0, sum_gt = 0.0;
            for (size_t i = 0; i < vol_elems; ++i) {
                double diff = (double)h_rec[i] - (double)h_gt[i];
                mse += diff * diff;
                sum_rec += h_rec[i];
                sum_gt += h_gt[i];
            }
            mse /= (double)vol_elems;
            YK_LOGI("{}: sum={:.6e}  GT sum={:.6e}  MSE={:.6e}",
                name, sum_rec, sum_gt, mse);
            };

        compare("OSSART",
            (test_data_dir + "sim_ossart_vol.raw").c_str());
        compare("SIRT",
            (test_data_dir + "sim_sirt_vol.raw").c_str());
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

// ----------------------------------------------------------------
// main_iter_recon_sim：模拟测试
// 流程：读体积 → Joseph FP 生成正弦图 → SIRT 迭代重建 → 对比原始体积
// ----------------------------------------------------------------
int main_iter_sirt_recon_sim()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 360; params.iPAngTotal = 360;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 3.0f / 3.0f;
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

    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // ---- 读测量正弦图 ────────────────────────────────────────────
    std::vector<float> h_sino(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read proj file");
        return -1;
    }
    YK_LOGI("sino loaded");

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    {
        auto h_buf = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        std::memcpy(h_buf.data(), h_sino.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_sino, h_buf);
    }

    auto d_vol = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol.data(), 0, vol_elems * sizeof(float), s));

    // ---- SIRT配置 ───────────────────────────────────────────────
    {
        SIRT::Config cfg;
        cfg.n_iter = 40;
        cfg.n_batch = 10;
        cfg.lambda = 1.f;
        cfg.lambda_red = 0.999f;
        cfg.eps = 1e-6f;
        cfg.use_min = true;
        cfg.min_constraint = 0.f;
        cfg.dump_debug = false;
        cfg.row_w_down = 2;
        cfg.fp_task = ETask::FP_Joseph;
        cfg.bp_task = ETask::BP_FDK;

        SIRT recon;
        if (!recon.init(params, cfg, s)) {
            YK_LOGE("SIRT init failed");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }

        YK_LOGI("SIRT start: {} iters x {} batches",
            cfg.n_iter, cfg.n_batch);

        YK::Util::CudaTimer timer("sirt_total", s);
        recon.run(d_sino.data(), d_vol.data(), params, s);
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        YK_LOGI("SIRT done");
        recon.release();
    }

    // ---- 保存结果 ───────────────────────────────────────────────
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol);

        float minv = *std::min_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        YK_LOGI("vol stats: min={:.6f} max={:.6f}", minv, maxv);

        write_raw_float((test_data_dir + "sirt_vol.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: sirt_vol.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ---- 与FDK结果对比 ──────────────────────────────────────────
    {
        std::vector<float> h_fdk(vol_elems);
        if (read_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_fdk)) {
            auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
            ctrl.download3D(h_vol, d_vol);

            double mse = 0.0, sum_sirt = 0.0, sum_fdk = 0.0;
            for (size_t i = 0; i < vol_elems; ++i) {
                double diff = (double)h_vol.cdata()[i] - (double)h_fdk[i];
                mse += diff * diff;
                sum_sirt += h_vol.cdata()[i];
                sum_fdk += h_fdk[i];
            }
            mse /= (double)vol_elems;
            YK_LOGI("FDK  sum = {:.6e}", sum_fdk);
            YK_LOGI("SIRT sum = {:.6e}", sum_sirt);
            YK_LOGI("ratio    = {:.4f}", sum_fdk > 0 ? sum_sirt / sum_fdk : 0.0);
            YK_LOGI("MSE(SIRT vs FDK) = {:.6e}", mse);
        }
        else {
            YK_LOGI("fdk_vec_vol_online.raw not found, skip comparison");
        }
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}


void test_flat_detector_roty_fp_ossart(cudaStream_t stream)
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

    // ---- 正投影 ─────────────────────────────────────────────────
    run_fp(params, h_views, "fp_flat_det_roty_sino2.raw", "pcb_phantom.raw", stream);

    // ---- 重建参数 ───────────────────────────────────────────────
    params.iVX = 512;
    params.iVY = 100;
    params.iVZ = 512;
    params.vox_x_mm = 0.2f;
    params.vox_y_mm = 0.04f;
    params.vox_z_mm = 0.2f;
    std::for_each(params.angle_list.begin(), params.angle_list.end(),
        [](float& a) { a = a; });

    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    // 读正弦图
    std::vector<float> h_sino(proj_elems);
    if (!read_raw_float((test_data_dir + "fp_flat_det_roty_sino.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read fp_flat_det_roty_sino.raw");
        return;
    }

    Mem::MemoryController ctrl;
    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    {
        auto h_buf = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        std::memcpy(h_buf.data(), h_sino.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_sino, h_buf);
    }

    auto d_vol = ctrl.allocateDevice3D<float>(params.iVX, params.iVY, params.iVZ, 0, false);
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol.data(), 0, vol_elems * sizeof(float), stream));

    //// just bp
    //{
    //    ConeBackprojectorEx recon;
    //    recon.init(params, ETask::BP_FDK, 0);
    //    recon.run(d_sino.data(), params, h_views, stream, d_vol.data(), true, 0);
    //    auto h_vol = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
    //    ctrl.download3D(h_vol, d_vol);

    //    write_raw_float((test_data_dir + "bp_only_vol.raw").c_str(),
    //        h_vol.data(), vol_elems);
    //}

    // ---- OS-SART ────────────────────────────────────────────────
    OSSARTEx::Config cfg;
    cfg.n_iter = 10;
    cfg.n_subset = 20;
    cfg.lambda = 1.f;
    cfg.lambda_red = 1.f;
    cfg.eps = 1e-6f;
    cfg.use_min = false;
    cfg.fp_task = ETask::FP_Joseph;
    cfg.bp_task = ETask::BP_Joseph_v3;

    {
        OSSARTEx recon;
        if (!recon.init(params, cfg, h_views, stream)) {
            YK_LOGE("OSSARTEx init failed");
            return;
        }

        YK_LOGI("OSSARTEx start: {} iters x {} subsets", cfg.n_iter, cfg.n_subset);
        YK::Util::CudaTimer timer("ossart_flat_det_roty", stream);
        recon.run(d_sino.data(), d_vol.data(), stream);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        YK_LOGI("OSSARTEx done, total iterations={}", recon.totalIterations());
        recon.release();
    }

    // ---- 保存 ───────────────────────────────────────────────────
    {
        auto h_vol = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
        ctrl.download3D(h_vol, d_vol);

        float minv = *std::min_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        YK_LOGI("vol stats: min={:.6f} max={:.6f}", minv, maxv);

        write_raw_float((test_data_dir + "recon_flat_det_roty_ossart.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: recon_flat_det_roty_ossart.raw ({}x{}x{})",
            params.iVX, params.iVY, params.iVZ);
    }
}

void test_flat_detector_roty_fp_independent(cudaStream_t stream)
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

    // ---- 正投影 ─────────────────────────────────────────────────
    run_fp(params, h_views, "fp_flat_det_roty_sino.raw", "pcb_phantom.raw", stream);

    // ---- 重建参数 ───────────────────────────────────────────────
    params.iVX = 512;
    params.iVY = 100;
    params.iVZ = 512;
    params.vox_x_mm = 0.2f;
    params.vox_y_mm = 0.04f;
    params.vox_z_mm = 0.2f;

    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * params.iPAng;
    const size_t vol_elems = (size_t)params.iVX * params.iVY * params.iVZ;

    // 读正弦图
    std::vector<float> h_sino(proj_elems);
    if (!read_raw_float((test_data_dir + "fp_flat_det_roty_sino.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read fp_flat_det_roty_sino.raw");
        return;
    }

    Mem::MemoryController ctrl;
    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    {
        auto h_buf = ctrl.allocateCpu3D<float>(view_elems, params.iPAng, 1, false);
        std::memcpy(h_buf.data(), h_sino.data(), proj_elems * sizeof(float));
        ctrl.upload3D(d_sino, h_buf);
    }

    auto d_vol = ctrl.allocateDevice3D<float>(params.iVX, params.iVY, params.iVZ, 0, false);
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol.data(), 0, vol_elems * sizeof(float), stream));

    // ---- OS-SART（用Ex版，传入自定义h_views）───────────────────
    OSSART::Config cfg;
    cfg.n_iter = 10;
    cfg.n_subset = 20;
    cfg.lambda = 1.f;
    cfg.lambda_red = 1.f;
    cfg.eps = 1e-6f;
    cfg.use_min = false;
    cfg.fp_task = ETask::FP_Joseph;
    cfg.bp_task = ETask::BP_FDK;

    // OSSART内部用ConeProjector/ConeBackprojector，不接受外部h_views
    // 改用Ex版手动实现迭代
    ConeProjectorEx     fp;
    ConeBackprojectorEx bp;
    fp.init(params, cfg.fp_task);
    bp.init(params, cfg.bp_task);

    const int    Na = params.iPAng;
    const int    n_subset = cfg.n_subset;
    const size_t max_K = (Na + n_subset - 1) / n_subset;

    // 构建子集
    std::vector<std::vector<int>> subsets(n_subset);
    for (int s = 0; s < n_subset; ++s)
        for (int i = s; i < Na; i += n_subset)
            subsets[s].push_back(i);

    // 分配buffer
    float* d_sino_sub = nullptr, * d_sino_fwd = nullptr;
    float* d_residual = nullptr, * d_row_w = nullptr;
    float* d_bp = nullptr, * d_ones_vol = nullptr, * d_col_w = nullptr;
    const size_t max_sino = max_K * view_elems;
    YK_CUDA_CHECK(cudaMalloc(&d_sino_sub, max_sino * sizeof(float)));
    YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd, max_sino * sizeof(float)));
    YK_CUDA_CHECK(cudaMalloc(&d_residual, max_sino * sizeof(float)));
    YK_CUDA_CHECK(cudaMalloc(&d_row_w, max_sino * sizeof(float)));
    YK_CUDA_CHECK(cudaMalloc(&d_bp, vol_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMalloc(&d_ones_vol, vol_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMalloc(&d_col_w, vol_elems * sizeof(float)));

    YK::Iter::fill_ones_launch(d_ones_vol, vol_elems, stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));

    {
        YK::Util::CudaTimer timer("ossart_flat_det_roty", stream);

        for (int iter = 0; iter < cfg.n_iter * n_subset; ++iter)
        {
            const int s = iter % n_subset;
            const auto& idx = subsets[s];
            const int K = (int)idx.size();
            const size_t sino_n = (size_t)K * view_elems;

            // 子集参数和几何
            SCBCTParams ps = params;
            ps.iPAng = K;
            ps.angle_list.resize(K);
            std::vector<SConeProjGeomVec> sub_views(K);
            for (int i = 0; i < K; ++i) {
                ps.angle_list[i] = params.angle_list[idx[i]];
                sub_views[i] = h_views[idx[i]];
            }

            // 行权重：A_s · 1_vol
            YK_CUDA_CHECK(cudaMemsetAsync(d_row_w, 0, sino_n * sizeof(float), stream));
            fp.run(d_ones_vol, ps, sub_views, d_row_w, stream);

            // 列权重：A_s^T · 1_proj
            YK::Iter::fill_ones_launch(d_residual, sino_n, stream);
            bp.run(d_residual, ps, sub_views, stream, d_col_w, true);

            // 收集子集正弦图
            for (int i = 0; i < K; ++i)
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_sino_sub + i * view_elems,
                    d_sino.data() + idx[i] * view_elems,
                    view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));

            // 正投影
            YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd, 0, sino_n * sizeof(float), stream));
            fp.run(d_vol.data(), ps, sub_views, d_sino_fwd, stream);

            // 残差
            YK::Iter::residual_launch(d_sino_sub, d_sino_fwd, d_residual, sino_n, stream);

            // R行归一化
            YK::Iter::divide_launch(d_residual, d_row_w, cfg.eps, sino_n, stream);

            // 反投影
            bp.run(d_residual, ps, sub_views, stream, d_bp, true);

            // 更新
            YK::Iter::update_launch(d_vol.data(), d_bp, d_col_w,
                cfg.lambda, cfg.eps, vol_elems, stream);

            if (cfg.use_min)
                YK::Iter::clamp_min_launch(d_vol.data(), vol_elems, cfg.min_constraint, stream);

            YK_LOGD("[ossart_flat] iter={} subset={}/{} K={}", iter + 1, s + 1, n_subset, K);
        }

        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // 保存
    {
        auto h_vol = ctrl.allocateCpu3D<float>(params.iVX, params.iVY, params.iVZ, false);
        ctrl.download3D(h_vol, d_vol);

        float minv = *std::min_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        float maxv = *std::max_element(h_vol.cdata(), h_vol.cdata() + vol_elems);
        YK_LOGI("vol stats: min={:.6f} max={:.6f}", minv, maxv);

        write_raw_float((test_data_dir + "recon_flat_det_roty_ossart.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: recon_flat_det_roty_ossart.raw ({}x{}x{})",
            params.iVX, params.iVY, params.iVZ);
    }

    cudaFree(d_sino_sub); cudaFree(d_sino_fwd);
    cudaFree(d_residual); cudaFree(d_row_w);
    cudaFree(d_bp);       cudaFree(d_ones_vol); cudaFree(d_col_w);
}