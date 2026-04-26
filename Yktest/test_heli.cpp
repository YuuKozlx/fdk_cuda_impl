#include <FDK/YkFdkReconstructor.hpp>
#include <Heli/YkHeliCTParams.h>
#include <global/YkMacro.hpp>
#include <test_common.hpp>
#include <util/YkCudaTimer.hpp>
#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHelicalProjector.hpp"
#include "Heli/YkHelicalReconstructor.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkMem3d.hpp"



using namespace YK;

const std::string test_data_dir = "/workspace/fdk-test/TestData/";




int main_helical_from_volume()
{
    // ----------------------------------------------------------------
    // 螺旋 CT 参数
    // ----------------------------------------------------------------
    SHeliCTParam p;

    // 探测器
    p.iPU = 1024;
    p.iPV = 128;
    p.du_mm = 0.417f;
    p.dv_mm = 0.417f;
    p.offsetU_mm = 0.f;
    p.offsetV_mm = 0.f;

    // 几何
    p.SID = 500.0f;
    p.SDD = 1000.0f;

    // 体积
    p.iVX = 512;
    p.iVY = 512;
    p.iVZ = 400;
    p.vox_x_mm = 0.3f;
    p.vox_y_mm = 0.3f;
    p.vox_z_mm = 0.3f;
    p.vol_offset_x_mm = 0.f;
    p.vol_offset_y_mm = 0.f;
    p.vol_offset_z_mm = 0.f;

    // 螺旋扫描
    p.pitch_mm = 20.0f;
    p.bShortScan = false;
    YK::fillHelicalScanGeometry(p, 0.75f, 720, true);

    // 重建
    p.z_block_mm = 20.0f;
    p.z_step_mm = 10.0f;
    p.Kchunk = 32;
    p.fp_task = ETask::FP_Joseph;

    // ----------------------------------------------------------------
    const size_t vol_elems =
        (size_t)p.iVX * p.iVY * p.iVZ;

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    Mem::MemoryController ctrl;

    // ----------------------------------------------------------------
    // 从外部读取体积
    // ----------------------------------------------------------------
    std::vector<float> h_phantom(vol_elems);
    const std::string vol_path =
        test_data_dir + "arrow_phantom_512x512x400_f.raw";
    if (!read_raw_float(vol_path.c_str(), h_phantom)) {
        YK_LOGE("cannot read {}", vol_path);
        YK_CUDA_CHECK(cudaStreamDestroy(s));
        return -1;
    }
    YK_LOGI("loaded: {} ({}x{}x{})",
        vol_path, p.iVX, p.iVY, p.iVZ);

    {
        float minv = *std::min_element(h_phantom.begin(), h_phantom.end());
        float maxv = *std::max_element(h_phantom.begin(), h_phantom.end());
        YK_LOGI("phantom stats: min={:.4f} max={:.4f}", minv, maxv);
    }

    auto d_phantom = ctrl.allocateDevice3D<float>(
        p.iVX, p.iVY, p.iVZ, 0, false);
    YK_CUDA_CHECK(cudaMemcpy(
        d_phantom.data(), h_phantom.data(),
        vol_elems * sizeof(float), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------------
    // 螺旋正投影
    // ----------------------------------------------------------------
    YK::HelicalProjector projector;
    if (!projector.init(p, s)) {
        YK_LOGE("HelicalProjector init failed");
        YK_CUDA_CHECK(cudaStreamDestroy(s));
        return -1;
    }

    YK_LOGI("total views={}", projector.totalViews());

    const size_t proj_elems =
        (size_t)p.iPU * p.iPV * projector.totalViews();
    std::vector<float> h_proj(proj_elems, 0.f);

    {
        Util::CudaTimer timer("helical_fp", s);
        if (!projector.project(d_phantom.data(), h_proj.data(), s)) {
            YK_LOGE("project failed");
            projector.release();
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
    }

    write_raw_float(
        (test_data_dir + "helical_proj_512.raw").c_str(),
        h_proj.data(), proj_elems);
    YK_LOGI("saved: helical_proj_512.raw ({}x{}x{})",
        p.iPU, p.iPV, projector.totalViews());

    {
        float minv = *std::min_element(h_proj.begin(), h_proj.end());
        float maxv = *std::max_element(h_proj.begin(), h_proj.end());
        YK_LOGI("proj stats: min={:.4f} max={:.4f}", minv, maxv);
    }

    // ----------------------------------------------------------------
    // 分段螺旋 FDK 重建
    // ----------------------------------------------------------------
    YK::HelicalReconstructor recon;
    if (!recon.init(p, projector.geo(), s)) {
        YK_LOGE("HelicalReconstructor init failed");
        projector.release();
        YK_CUDA_CHECK(cudaStreamDestroy(s));
        return -1;
    }

    YK_LOGI("slabs={} z=[{:.2f},{:.2f}]mm",
        recon.totalSlabs(), recon.zVolStart(), recon.zVolEnd());

    std::vector<float> h_vol_out(vol_elems, 0.f);

    {
        Util::CudaTimer timer("helical_recon", s);
        if (!recon.reconstruct(h_proj.data(), h_vol_out.data(), s)) {
            YK_LOGE("reconstruct failed");
            recon.release();
            projector.release();
            YK_CUDA_CHECK(cudaStreamDestroy(s));
            return -1;
        }
    }

    write_raw_float(
        (test_data_dir + "helical_recon_512.raw").c_str(),
        h_vol_out.data(), vol_elems);
    YK_LOGI("saved: helical_recon_512.raw ({}x{}x{})",
        p.iVX, p.iVY, p.iVZ);

    {
        float maxv = 0.f, mean = 0.f;
        for (size_t i = 0; i < vol_elems; ++i) {
            maxv = std::max(maxv, h_vol_out[i]);
            mean += h_vol_out[i];
        }
        mean /= (float)vol_elems;
        YK_LOGI("recon stats: max={:.4f} mean={:.6f}", maxv, mean);
    }

    printf("Done: helical_from_volume  views=%d  slabs=%d\n",
        projector.totalViews(), recon.totalSlabs());

    recon.release();
    projector.release();
    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}