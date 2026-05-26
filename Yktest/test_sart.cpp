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
    params.iPAng = 240; params.iPAngTotal = 240;
    params.tiltn_angle_rad = 0;
    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 2.0f / 3.0f;
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
    cfg.n_iter = 5;
    cfg.n_subset = 8;
    cfg.lambda = 0.2f;
    cfg.lambda_red = 0.999f;
    cfg.eps = 1.f;
    cfg.use_min = false;
    cfg.min_constraint = 0.f;     // CT 值非负约束
    cfg.fp_task = ETask::FP_Siddon;
    cfg.bp_task = ETask::BP_Siddon_VoxDriven;

    // ---- 运行 OS-SART ───────────────────────────────────────────
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

        write_raw_float((test_data_dir + "ossart_vol.raw").c_str(),
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

    // d_sino 和 d_vol_rec 需要跨作用域，在外层声明
    auto d_sino = ctrl.allocateDevice3D<float>(view_elems, params.iPAng, 1, 0);
    auto d_vol_rec = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ---- Step1：读 phantom，FP 生成正弦图，用完释放 d_vol_gt ────
    std::vector<float> h_vol_gt(vol_elems);  // CPU 副本，保留到最后做对比
    {
        if (!read_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_vol_gt)) {
            YK_LOGE("cannot read fdk_vec_vol_online.raw");
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
        YK_LOGI("ground truth loaded");

        float minv = *std::min_element(h_vol_gt.begin(), h_vol_gt.end());
        float maxv = *std::max_element(h_vol_gt.begin(), h_vol_gt.end());
        YK_LOGI("gt stats: min={:.6f} max={:.6f}", minv, maxv);

        // d_vol_gt 在此作用域内，结束时自动释放 400MB
        auto d_vol_gt = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
        {
            auto h_buf = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
            std::memcpy(h_buf.data(), h_vol_gt.data(), vol_elems * sizeof(float));
            ctrl.upload3D(d_vol_gt, h_buf);
        }

        // FP 生成正弦图
        {
            ConeProjector fp;
            fp.init(params, ETask::FP_Joseph);
            YK_CUDA_CHECK(cudaMemsetAsync(d_sino.data(), 0,
                proj_elems * sizeof(float), s));
            YK::Util::CudaTimer timer("fp_generate_sino", s);
            fp.run(d_vol_gt.data(), params, d_sino.data(), s);
        }
        cudaStreamSynchronize(s);
    }   // ← d_vol_gt 在这里析构，释放 400MB
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

    // ---- Step2：SIRT 重建 ───────────────────────────────────────
    // 此时显存：d_sino(2GB) + d_vol_rec(400MB) = 2.4GB
    // SIRT 额外占：d_sino_fwd_(2GB) + d_bp_(400MB) + d_weight_(400MB) = 2.8GB
    // iterate 时临时 d_residual(2GB)，峰值约 7.2GB
    YK_CUDA_CHECK(cudaMemsetAsync(d_vol_rec.data(), 0,
        vol_elems * sizeof(float), s));
    {
        SIRT::Config cfg;
        cfg.n_iter = 5;
        cfg.n_batch = 10;
        cfg.lambda = 1.f;
        cfg.eps = 1e-6f;
        cfg.use_min = false;
        cfg.min_constraint = 0.f;
        cfg.use_max = false;
        cfg.max_constraint = 1e30f;
        cfg.fp_task = ETask::FP_Siddon;
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

    // ---- Step3：对比 GT（h_vol_gt 是 FP 输入，即 phantom）────────
    {
        std::vector<float> h_rec(vol_elems);
        read_raw_float((test_data_dir + "sim_sirt_vol.raw").c_str(), h_rec);

        double mse = 0.0, sum_rec = 0.0, sum_gt = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = (double)h_rec[i] - (double)h_vol_gt[i];
            mse += diff * diff;
            sum_rec += h_rec[i];
            sum_gt += h_vol_gt[i];
        }
        mse /= (double)vol_elems;

        YK_LOGI("phantom sum  = {:.6e}", sum_gt);
        YK_LOGI("SIRT    sum  = {:.6e}", sum_rec);
        YK_LOGI("ratio        = {:.4f}", sum_gt > 0 ? sum_rec / sum_gt : 0.0);
        YK_LOGI("MSE(SIRT vs phantom) = {:.6e}", mse);

        if (mse < 1e-4)
            YK_LOGI("PASS: SIRT converged close to phantom");
        else
            YK_LOGI("INFO: MSE={:.6e}, may need more iterations or lambda tuning", mse);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}