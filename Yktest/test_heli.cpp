#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHelicalReconstructor.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkMem3d.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include <test_common.hpp>
#include <util/YkCudaTimer.hpp>
#include <FDK/YkFdkReconstructor.hpp>

using namespace YK;

const std::string test_data_dir = R"(G:\Code\fanproj\fdk-test\TestData\)";

int main_helical_verify()
{
    // ----------------------------------------------------------------
    // 参数：32 排，螺旋扫描
    // ----------------------------------------------------------------
    SCBCTParams params;
    params.iPU = 512;  params.iPV = 32;
    params.iVX = 256;  params.iVY = 256; params.iVZ = 80;
    params.bShortScan = false;
    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.5f;  params.dv_mm = 0.25f;
    params.vox_x_mm = 0.5f; params.vox_y_mm = 0.5f;
    params.vox_z_mm = 0.25f;
    params.offsetU_mm = 0.f;
    params.tiltu_angle_rad = 0.f;
    params.tiltn_angle_rad = 0.f;
    params.tiltv_angle_rad = 0.f;
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;

    YK::HelicalReconstructor::SHeliFpConfig cfg;
    cfg.pitch_mm = 3.0f;
    cfg.auto_start_z = true;
    cfg.z_block_mm = 3.0f;
    cfg.z_step_mm = 1.5f;
    cfg.Kchunk = 32;
    cfg.views_per_rot = 360;
    cfg.n_rotations = 0;
    cfg.fp_task = ETask::FP_Joseph;

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    Mem::MemoryController ctrl;

    // ----------------------------------------------------------------
    // 初始化重建器
    // ----------------------------------------------------------------
    YK::HelicalReconstructor recon;
    if (!recon.init(params, cfg, s)) {
        YK_LOGE("HelicalReconstructor init failed");
        YK_CUDA_CHECK(cudaStreamDestroy(s));
        return -1;
    }

    YK_LOGI("total views={} slabs={} z=[{:.2f},{:.2f}]mm",
        recon.totalViews(), recon.totalSlabs(),
        recon.zVolStart(), recon.zVolEnd());

    // ----------------------------------------------------------------
    // 生成均匀球体体模
    // ----------------------------------------------------------------
    const size_t vol_elems =
        (size_t)params.iVX * params.iVY * params.iVZ;
    std::vector<float> h_phantom(vol_elems, 0.f);
    {
        const int cx = params.iVX / 2;
        const int cy = params.iVY / 2;
        const int cz = params.iVZ / 2;
        const int r = std::min({ params.iVX, params.iVY, params.iVZ }) / 4;

        for (int z = 0; z < params.iVZ; ++z)
            for (int y = 0; y < params.iVY; ++y)
                for (int x = 0; x < params.iVX; ++x) {
                    const int dx = x - cx, dy = y - cy, dz = z - cz;
                    if (dx * dx + dy * dy + dz * dz < r * r)
                        h_phantom[(size_t)z * params.iVY * params.iVX
                        + y * params.iVX + x] = 1.f;
                }

        write_raw_float(
            (test_data_dir + "helical_phantom.raw").c_str(),
            h_phantom.data(), vol_elems);
        YK_LOGI("saved: helical_phantom.raw ({}x{}x{})",
            params.iVX, params.iVY, params.iVZ);
    }

    // 上传体模到设备端
    auto d_phantom = ctrl.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);
    YK_CUDA_CHECK(cudaMemcpy(
        d_phantom.data(), h_phantom.data(),
        vol_elems * sizeof(float), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------------
    // 螺旋正投影
    // ----------------------------------------------------------------
    const size_t proj_elems =
        (size_t)params.iPU * params.iPV * recon.totalViews();
    std::vector<float> h_proj(proj_elems, 0.f);

    {
        Util::CudaTimer timer("helical_forward_project", s);
        if (!recon.forwardProject(d_phantom.data(), h_proj.data(), s)) {
            YK_LOGE("forwardProject failed");
            recon.release();
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
    }

    write_raw_float(
        (test_data_dir + "helical_proj.raw").c_str(),
        h_proj.data(), proj_elems);
    YK_LOGI("saved: helical_proj.raw ({}x{}x{})",
        params.iPU, params.iPV, recon.totalViews());

    {
        float minv = *std::min_element(h_proj.begin(), h_proj.end());
        float maxv = *std::max_element(h_proj.begin(), h_proj.end());
        YK_LOGI("proj stats: min={:.4f} max={:.4f}", minv, maxv);
    }


    {
        const int vpr = cfg.views_per_rot;
        std::vector<float> circ_angles(vpr);
        for (int i = 0; i < vpr; ++i)
            circ_angles[i] = i * 2.f * CUDA_PI / vpr;

        SCBCTParams circ_params = params;
        circ_params.iPAng = vpr;
        circ_params.iPAngTotal = vpr;
        circ_params.bShortScan = false;
        circ_params.scan_range_rad = 2.f * CUDA_PI;
        circ_params.scan_start_angle_rad = 0.f;
        circ_params.vol_offset_z_mm = 0.f;
        circ_params.angle_list = circ_angles;

        // 圆轨迹正投影（用 fp_project，内部用圆轨迹几何）
        const size_t circ_proj_elems =
            (size_t)params.iPU * params.iPV * vpr;
        std::vector<float> h_proj_circ(circ_proj_elems, 0.f);

        float* d_proj_circ = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_proj_circ,
            circ_proj_elems * sizeof(float)));

        fp_project(d_phantom.data(), d_proj_circ,
            circ_params, ETask::FP_Joseph, s);
        cudaStreamSynchronize(s);

        YK_CUDA_CHECK(cudaMemcpy(h_proj_circ.data(), d_proj_circ,
            circ_proj_elems * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_proj_circ);

        // 圆轨迹重建
        auto d_vol_circ = ctrl.allocateDevice3D<float>(
            params.iVX, params.iVY, params.iVZ, 0, false);

        FdkReconstructor fdk;
        fdk.init(circ_params, 32, s);
        fdk.feed(h_proj_circ.data(), circ_params, s,
            d_vol_circ.data(), true);
        cudaStreamSynchronize(s);

        auto h_vol_circ = ctrl.allocateCpu3D<float>(
            params.iVX, params.iVY, params.iVZ, false);
        ctrl.download3D(h_vol_circ, d_vol_circ);

        float maxv_circ = 0.f;
        for (size_t i = 0; i < vol_elems; ++i)
            maxv_circ = std::max(maxv_circ, h_vol_circ.cdata()[i]);
        YK_LOGI("[circ FDK reference] max={:.4f}", maxv_circ);

        write_raw_float(
            (test_data_dir + "helical_circ_ref.raw").c_str(),
            h_vol_circ.cdata(), vol_elems);
    }

    // ----------------------------------------------------------------
    // 分段螺旋 FDK 重建
    // ----------------------------------------------------------------
    std::vector<float> h_vol_out(vol_elems, 0.f);

    {
        Util::CudaTimer timer("helical_reconstruct", s);
        if (!recon.reconstruct(h_proj.data(), h_vol_out.data(), s)) {
            YK_LOGE("reconstruct failed");
            recon.release();
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
    }

    write_raw_float(
        (test_data_dir + "helical_recon.raw").c_str(),
        h_vol_out.data(), vol_elems);
    YK_LOGI("saved: helical_recon.raw ({}x{}x{})",
        params.iVX, params.iVY, params.iVZ);

    // ----------------------------------------------------------------
    // 重建统计
    // ----------------------------------------------------------------
    {
        float maxv = 0.f, mean = 0.f;
        for (size_t i = 0; i < vol_elems; ++i) {
            maxv = std::max(maxv, h_vol_out[i]);
            mean += h_vol_out[i];
        }
        mean /= (float)vol_elems;
        YK_LOGI("recon stats: max={:.4f} mean={:.6f}", maxv, mean);
    }

    // ----------------------------------------------------------------
    // 与体模对比（只比较球体内部体素）
    // ----------------------------------------------------------------
    {
        double maxDiff = 0.0, mse = 0.0;
        size_t nonzero = 0;

        for (size_t i = 0; i < vol_elems; ++i) {
            if (h_phantom[i] > 0.5f) {
                const double diff = std::abs((double)h_vol_out[i] - 1.0);
                maxDiff = std::max(maxDiff, diff);
                mse += diff * diff;
                ++nonzero;
            }
        }
        mse /= (double)std::max(nonzero, (size_t)1);

        YK_LOGI("[phantom vs recon] maxDiff={:.6f}  MSE={:.6e}  "
            "nonzero_voxels={}",
            maxDiff, mse, nonzero);

        if (maxDiff < 0.1f)
            YK_LOGI("PASS: helical reconstruction looks correct");
        else
            YK_LOGW("WARN: maxDiff={:.6f} may indicate issues", maxDiff);
    }

    recon.release();
    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: helical verify  views=%d  slabs=%d\n",
        recon.totalViews(), recon.totalSlabs());
    return 0;
}