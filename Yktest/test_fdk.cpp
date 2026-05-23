#include "test_common.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <fstream>

#include "FDK/YkFdkReconstructor.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include <FDK/YkFdkSlabReconstructor.hpp>
#include <random>

using namespace YK;
using namespace Mem;

const std::string test_data_dir = R"(H:\Code\fanproj\fdk-test\TestData\)";

int main_fdk()
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

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    const int    Ang = params.iPAng;
    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_proj)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    auto dump = [](void* p) {
        auto* payload = static_cast<DumpPayload*>(p);
        if (payload->viewIdx != 0) return;
        float* d_buf = static_cast<float*>(payload->buf);
        const size_t n = payload->n;

        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.f, maxv = -1e30f, minv = 1e30f;
        for (auto v : h) { sum += v; maxv = std::max(maxv, v); minv = std::min(minv, v); }
        YK_LOGI("[dump][a={}][{}] n={} min={:.4f} max={:.4f} mean={:.6f}",
            payload->viewIdx, payload->stage, n, minv, maxv, sum / (float)n);

        auto path = fmt::format("dump_a{}_{}.raw", payload->viewIdx, payload->stage);
        std::ofstream f(path, std::ios::binary);
        if (f) f.write(reinterpret_cast<const char*>(h.data()), n * sizeof(float));
        else   YK_LOGE("[dump] cannot save {}", path);
        };

    MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ---- 离线重建 ----
    {
        YK::Util::CudaTimer timer("offline", s);
        YK::fdk_recon(h_proj.data(), d_vol_buf.data(), params,
            /*Kchunk=*/64, s, /*clear_vol=*/true, nullptr, nullptr);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "fdk_vec_vol_offline.raw").c_str(), h_vol.cdata(), vol_elems);
        YK_LOGI("saved: fdk_vec_vol_offline.raw");
    }

    // ---- 在线重建 ----
    const int batch_size = 32 * 2;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    FdkReconstructor recon;
    recon.init(params, /*Kchunk=*/64, s);
    {
        YK::Util::CudaTimer timer("online", s);
        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams bp = params;
            bp.iPAng = count;
            bp.angle_list = std::vector<float>(
                angle_list.begin() + base, angle_list.begin() + base + count);

            recon.feed(h_proj.data() + base * view_elems, bp, s,
                d_vol_buf.data(), (i == 0));
        }
    }
    recon.reset();
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "fdk_vec_vol_online.raw").c_str(), h_vol.cdata(), vol_elems);
        YK_LOGI("saved: fdk_vec_vol_online.raw");
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: fdk offline + online (%d x %d x %d)\n", Nx, Ny, Nz);
    return 0;
}


int main_fdk_zslab_bigdata()
{
    SCBCTParams params;
    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0;
    params.iVX = 512 * 4; params.iVY = 512 * 4; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 4.0f / 3.0f;
    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f / 4.f; params.vox_y_mm = 0.1f / 4.f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    const int    Ang = params.iPAng;
    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_proj)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    // ---- z_block_size 参数：模拟显存不足，每次只重建 50 层 ----
    const int z_block_size = 200;  // iVZ=400，共 8 个 slab
    const int batch_size = 32 * 2;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    std::vector<float> h_vol_out(vol_elems, 0.f);

    // ---- ZSlab 在线重建 ----
    YK::FdkZSlabReconstructor slab_recon;
    if (!slab_recon.init(params, /*Kchunk=*/64, s, z_block_size)) {
        YK_LOGE("FdkZSlabReconstructor init failed");
        YK_CUDA_CHECK(cudaStreamDestroy(s));
        return -1;
    }

    {
        YK::Util::CudaTimer timer("zslab_online", s);
        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams bp = params;
            bp.iPAng = count;
            bp.angle_list = std::vector<float>(
                angle_list.begin() + base, angle_list.begin() + base + count);

            if (!slab_recon.feed(h_proj.data() + base * view_elems, bp,
                h_vol_out.data(), s)) {
                YK_LOGE("FdkZSlabReconstructor feed failed at batch {}", i);
                slab_recon.release();
                YK_CUDA_CHECK(cudaStreamDestroy(s));
                return -1;
            }
        }
    }

    write_raw_float((test_data_dir + "fdk_zslab_vol.raw").c_str(),
        h_vol_out.data(), vol_elems);
    YK_LOGI("saved: fdk_zslab_vol.raw");

    //// ---- 与离线结果对比（可选）----
    //// 先跑一次标准离线重建作为 ground truth
    //{
    //    MemoryController ctrl;
    //    auto d_vol_ref = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    //    {
    //        YK::Util::CudaTimer timer("offline_ref", s);
    //        YK::fdk_recon(h_proj.data(), d_vol_ref.data(), params,
    //            /*Kchunk=*/32, s, /*clear_vol=*/true, nullptr, nullptr);
    //    }

    //    auto h_vol_ref = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    //    ctrl.download3D(h_vol_ref, d_vol_ref);

    //    // 计算最大绝对误差和均值误差
    //    float max_err = 0.f, mean_err = 0.f;
    //    for (size_t idx = 0; idx < vol_elems; ++idx) {
    //        const float diff = std::abs(h_vol_out[idx] - h_vol_ref.cdata()[idx]);
    //        max_err = std::max(max_err, diff);
    //        mean_err += diff;
    //    }
    //    mean_err /= (float)vol_elems;

    //    YK_LOGI("[zslab vs offline] max_err={:.6f}  mean_err={:.8f}", max_err, mean_err);

    //    write_raw_float((test_data_dir + "fdk_vec_vol_offline_ref.raw").c_str(),
    //        h_vol_ref.cdata(), vol_elems);
    //    YK_LOGI("saved: fdk_vec_vol_offline_ref.raw");
    //}

    slab_recon.release();
    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: fdk_zslab (%d x %d x %d, z_block=%d)\n", Nx, Ny, Nz, z_block_size);
    return 0;
}


int main_fdk_zslab()
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

    const int    Ang = params.iPAng;
    const int    Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float((test_data_dir + "proj_1024x1024x360.raw").c_str(), h_proj)) {
        YK_LOGE("cannot read proj_1024x1024x360.raw");
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    // ---- z_block_size 参数：模拟显存不足，每次只重建 100 层 ----
    const int z_block_size = 100;  // iVZ=400，共 4 个 slab
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    std::vector<float> h_vol_out(vol_elems, 0.f);

    // ---- ZSlab 在线重建 ----
    YK::FdkZSlabReconstructor slab_recon;
    if (!slab_recon.init(params, /*Kchunk=*/32, s, z_block_size)) {
        YK_LOGE("FdkZSlabReconstructor init failed");
        YK_CUDA_CHECK(cudaStreamDestroy(s));
        return -1;
    }

    {
        YK::Util::CudaTimer timer("zslab_online", s);
        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams bp = params;
            bp.iPAng = count;
            bp.angle_list = std::vector<float>(
                angle_list.begin() + base, angle_list.begin() + base + count);

            if (!slab_recon.feed(h_proj.data() + base * view_elems, bp,
                h_vol_out.data(), s)) {
                YK_LOGE("FdkZSlabReconstructor feed failed at batch {}", i);
                slab_recon.release();
                YK_CUDA_CHECK(cudaStreamDestroy(s));
                return -1;
            }
        }
    }

    write_raw_float((test_data_dir + "fdk_zslab_vol.raw").c_str(),
        h_vol_out.data(), vol_elems);
    YK_LOGI("saved: fdk_zslab_vol.raw");

    // ---- 与离线结果对比（可选）----
    // 先跑一次标准离线重建作为 ground truth
    {
        MemoryController ctrl;
        auto d_vol_ref = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

        {
            YK::Util::CudaTimer timer("offline_ref", s);
            YK::fdk_recon(h_proj.data(), d_vol_ref.data(), params,
                /*Kchunk=*/32, s, /*clear_vol=*/true, nullptr, nullptr);
        }

        auto h_vol_ref = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_ref, d_vol_ref);

        // 计算最大绝对误差和均值误差
        float max_err = 0.f, mean_err = 0.f;
        for (size_t idx = 0; idx < vol_elems; ++idx) {
            const float diff = std::abs(h_vol_out[idx] - h_vol_ref.cdata()[idx]);
            max_err = std::max(max_err, diff);
            mean_err += diff;
        }
        mean_err /= (float)vol_elems;

        YK_LOGI("[zslab vs offline] max_err={:.6f}  mean_err={:.8f}", max_err, mean_err);

        write_raw_float((test_data_dir + "fdk_vec_vol_offline_ref.raw").c_str(),
            h_vol_ref.cdata(), vol_elems);
        YK_LOGI("saved: fdk_vec_vol_offline_ref.raw");
    }

    slab_recon.release();
    YK_CUDA_CHECK(cudaStreamDestroy(s));
    printf("Done: fdk_zslab (%d x %d x %d, z_block=%d)\n", Nx, Ny, Nz, z_block_size);
    return 0;
}



void test_stdrecon_with_random_offset(cudaStream_t stream)
{
    printf("\n[FDK] recon from offset sino\n");

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

    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> angle_list(Na);
    for (int i = 0; i < Na; ++i)
        angle_list[i] = i * 2.f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    // 直接读之前正投影存下来的 sino
    std::vector<float> h_sino(view_elems * Na);
    if (!read_raw_float((test_data_dir + "fp_perframe_offset_sino.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read fp_perframe_offset_sino.raw");
        return;
    }

    YK::Mem::MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        YK::Util::CudaTimer timer("offline_recon", stream);
        YK::fdk_recon(h_sino.data(), d_vol_buf.data(), params,
            /*Kchunk=*/64, stream, /*clear_vol=*/true, nullptr, nullptr);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "stdrecon_with_random_offset.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: recon_with_offset.raw");
    }

    printf("Done\n");
}


void test_recon_with_random_offset(cudaStream_t stream)
{
    printf("\n[FDK] recon from random offset sino\n");

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

    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> angle_list(Na);
    for (int i = 0; i < Na; ++i)
        angle_list[i] = i * 2.f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    // 与投影时完全相同的随机数种子和分布
    std::mt19937 rng(42);
    std::normal_distribution<float> dist_t(0.f, 0.25f);
    std::normal_distribution<float> dist_r(0.f, 0.1f);

    std::vector<float3> src_offsets(Na), det_offsets(Na);
    std::vector<float3> detTilt_degs(Na), srcCRTilt_degs(Na);
    for (int i = 0; i < Na; ++i) {
        src_offsets[i] = make_float3(0.f, 0.f, 0.f);
        det_offsets[i] = make_float3(dist_t(rng), dist_t(rng), 0.f);
        detTilt_degs[i] = make_float3(0.f, 0.f, dist_r(rng));
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

    // 读取正投影时存下来的 sino
    std::vector<float> h_sino(view_elems * Na);
    if (!read_raw_float((test_data_dir + "fp_perframe_offset_sino.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read fp_perframe_offset_sino.raw");
        return;
    }

    YK::Mem::MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        YK::Util::CudaTimer timer("offline_recon_random_offset", stream);
        YK::fdk_recon_ex(h_sino.data(), d_vol_buf.data(), params,
            h_views, /*Kchunk=*/64, stream, /*clear_vol=*/true, nullptr, nullptr);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "recon_with_random_offset.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: recon_with_random_offset.raw");
    }

    printf("Done\n");
}


void test_stdrecon_with_periodic_offset(cudaStream_t stream)
{
    printf("\n[FDK] recon from offset sino\n");

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

    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> angle_list(Na);
    for (int i = 0; i < Na; ++i)
        angle_list[i] = i * 2.f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    // 直接读之前正投影存下来的 sino
    std::vector<float> h_sino(view_elems * Na);
    if (!read_raw_float((test_data_dir + "fp_periodic_offset_sino.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read fp_periodic_offset_sino.raw");
        return;
    }

    YK::Mem::MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        YK::Util::CudaTimer timer("offline_recon", stream);
        YK::fdk_recon(h_sino.data(), d_vol_buf.data(), params,
            /*Kchunk=*/64, stream, /*clear_vol=*/true, nullptr, nullptr);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "stdrecon_with_periodic_offset.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: recon_with_offset.raw");
    }

    printf("Done\n");
}


void test_recon_with_periodic_offset(cudaStream_t stream)
{
    printf("\n[FDK] recon from periodic offset sino\n");

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

    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    std::vector<float> angle_list(Na);
    for (int i = 0; i < Na; ++i)
        angle_list[i] = i * 2.f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    // 与投影时完全相同的周期抖动
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

    std::vector<float> h_sino(view_elems * Na);
    if (!read_raw_float((test_data_dir + "fp_periodic_offset_sino.raw").c_str(), h_sino)) {
        YK_LOGE("cannot read fp_periodic_offset_sino.raw");
        return;
    }

    YK::Mem::MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        YK::Util::CudaTimer timer("offline_recon_periodic", stream);
        YK::fdk_recon_ex(h_sino.data(), d_vol_buf.data(), params,
            h_views, /*Kchunk=*/64, stream, /*clear_vol=*/true, nullptr, nullptr);
    }
    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        write_raw_float((test_data_dir + "recon_with_periodic_offset.raw").c_str(),
            h_vol.cdata(), vol_elems);
        YK_LOGI("saved: recon_with_periodic_offset.raw");
    }

    printf("Done\n");
}