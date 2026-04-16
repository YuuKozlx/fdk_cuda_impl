
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "FDK/YkFdkReconstructor.hpp"
#include "FDK/YkVecGeo.hpp"



#include "test/Yktest_fft.hpp"
#include "test/Yktest_fdkflter.hpp"
#include "global/YkMem3d.hpp"
#include "test/Yktest_mem3d.hpp"
#include "test/Yktest_dataobject.hpp"
#include "util/YkVecOperation.hpp"
#include "util/YkCudaTimer.hpp"


#include "FP/YkFPRunner.hpp"




static bool read_raw_float(const char* path, std::vector<float>& data) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    size_t n = std::fread(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
    return n == data.size();
}

static bool read_raw_float(const char* path, float* data, uint64_t element_count) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    size_t n = std::fread(data, sizeof(float), element_count, fp);
    std::fclose(fp);
    return n == element_count;
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

int main_fdk() {
    SCBCTParams params;

    params.iPU = 1024; params.iPV = 1024; params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0 * CUDA_PI / 180;

    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)M_PI * 4.0f / 3.0f; // 270 degree short scan

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)M_PI / 720;

    params.scan_start_angle_rad = angle_list[0]; // start at -30 degree
    params.angle_list = angle_list;


    params.SID = 500.0f, params.SDD = 1000.0f;
    params.du_mm = 0.25f, params.dv_mm = 0.25f, params.vox_x_mm = 0.25f;
    params.vox_y_mm = 0.25f;
    params.vox_z_mm = 0.25f;


    params.offsetU_mm = 0 * params.du_mm;
    const int Ang = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;

    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // 读投影
    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float("proj_1024x1024x360.raw", h_proj)) {
        std::printf("Error: cannot read proj_1024x1024x360 (expect %zu floats)\n", proj_elems);
        return -1;
    }

    // CUDA 资源
    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));



    // 调用
    auto dump = [](int a, const char* tag, float* d_buf, size_t n) {
        // 只看第0帧
        if (a != 0) return;

        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.f, maxv = -1e30f, minv = 1e30f;
        for (auto v : h) {
            sum += v;
            maxv = std::max(maxv, v);
            minv = std::min(minv, v);
        }
        printf("[dump][a=%d][%s] n=%zu min=%.4f max=%.4f mean=%.6f\n",
            a, tag, n, minv, maxv, sum / (float)n);
        };

    MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    // 离线重建（一次性全量）

    {
        YK::Util::CudaTimer timer("offline", s);
        YK::fdk_recon(
            h_proj.data(), d_vol_buf.data(),
            params,
            /*Kchunk=*/32, s,
            /*clear_vol=*/true, dump);
    }



    auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);

    ctrl.download3D(h_vol, d_vol_buf);


    uint64_t total_elements = (uint64_t)Nx * Ny * Nz;
    if (!write_raw_float("fdk_vec_vol_offline.raw", h_vol.cdata(), total_elements)) {
        std::printf("Error: cannot write fdk_vec_vol_new.raw\n");
        return -2;
    }

    //    // 模拟在线重建，每次传输iPBatch个角度，进行重建。保持KChunk=30不变，测试在线重建的正确性和性能。
    //    // 待更改参数 
    //    // 1. params.iPAng = iPBatch，角度list也相应缩减为当前批次的角度
    //    // 2. 每次循环传入的数据指针起点偏置
    //    // 3. 仅在第一批时 clear_vol=true，后续批次 clear_vol=false
    //
    //    // batch_size = 60; batch_num = Ang + batch_size - 1) / batch_size;;
    //// 在线重建
    //    int batch_size = 32 * 3;
    //    int batch_num = (Ang + batch_size - 1) / batch_size;
    //
    //    FdkReconstructor recon;
    //    recon.init(params, /*Kchunk=*/32, s);
    //
    //    {
    //        YK::Util::CudaTimer timer("online", s);
    //
    //        for (int i = 0; i < batch_num; ++i) {
    //            const int base = i * batch_size;
    //            const int count = std::min(batch_size, Ang - base);
    //
    //            SCBCTParams batch_params = params;
    //            batch_params.iPAng = count;
    //            batch_params.angle_list = std::vector<float>(
    //                angle_list.begin() + base,
    //                angle_list.begin() + base + count);
    //
    //            recon.feed(
    //                h_proj.data() + base * view_elems,
    //                batch_params, s,
    //                d_vol_buf.data(),
    //                /*clear_vol=*/(i == 0));
    //        }
    //    }
    //
    //
    //    // 新一轮扫描时
    //    recon.reset();
    //
    //    auto h_vol_online = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    //    ctrl.download3D(h_vol_online, d_vol_buf);
    //
    //    if (!write_raw_float("fdk_vec_vol_online.raw",
    //        h_vol_online.cdata(), total_elements)) {  // ← 修正变量名
    //        std::printf("Error: cannot write fdk_vec_vol_online.raw\n");
    //        return -2;
    //    }




    YK_CUDA_CHECK(cudaStreamDestroy(s));

    std::printf("Done: wrote fdk_vec_vol_online.raw (%d x %d x %d)\n", Nx, Ny, Nz);
    return 0;
}

int main_fdk2() {
    SCBCTParams params;

    params.iPU = 768; params.iPV = 768; params.iPAng = 360; params.iPAngTotal = 360;
    params.tiltn_angle_rad = 0 * CUDA_PI / 180;
    params.iVX = 768; params.iVY = 768; params.iVZ = 600;
    params.bShortScan = false;
    params.scan_range_rad = (float)M_PI * 6.0f / 3.0f; // 270 degree short scan

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)M_PI / 360;

    params.scan_start_angle_rad = angle_list[0]; // start at -30 degree
    params.angle_list = angle_list;


    params.SID = 430.0f, params.SDD = 760.0f;
    params.du_mm = 0.556f, params.dv_mm = 0.556f, params.vox_x_mm = 0.3f;
    params.vox_z_mm = 0.3f;

    const int Ang = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;

    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // 读投影
    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float("proj_768x768.raw", h_proj)) {
        std::printf("Error: cannot read proj_1024x1024.raw (expect %zu floats)\n", proj_elems);
        return -1;
    }

    // CUDA 资源
    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));



    // 调用
    auto dump = [](int a, const char* tag, float* d_buf, size_t n) {
        // 只看第0帧
        if (a != 0) return;

        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.f, maxv = -1e30f, minv = 1e30f;
        for (auto v : h) {
            sum += v;
            maxv = std::max(maxv, v);
            minv = std::min(minv, v);
        }
        printf("[dump][a=%d][%s] n=%zu min=%.4f max=%.4f mean=%.6f\n",
            a, tag, n, minv, maxv, sum / (float)n);
        };

    MemoryController ctrl;
    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    // 离线重建（一次性全量）

    //{
    //    YK::Util::CudaTimer timer("offline", s);
    //    YK::fdk_recon(
    //        h_proj.data(), d_vol_buf.data(),
    //        params,
    //        /*Kchunk=*/32, s,
    //        /*clear_vol=*/true, dump);
    //}



    //auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);

    //ctrl.download3D(h_vol, d_vol_buf);



    //if (!write_raw_float("fdk_vec_vol_offline2.raw", h_vol.cdata(), vol_elems)) {
    //    std::printf("Error: cannot write fdk_vec_vol_new.raw\n");
    //    return -2;
    //}

    // 模拟在线重建，每次传输iPBatch个角度，进行重建。保持KChunk=30不变，测试在线重建的正确性和性能。
    // 待更改参数 
    // 1. params.iPAng = iPBatch，角度list也相应缩减为当前批次的角度
    // 2. 每次循环传入的数据指针起点偏置
    // 3. 仅在第一批时 clear_vol=true，后续批次 clear_vol=false

    // batch_size = 60; batch_num = Ang + batch_size - 1) / batch_size;;
// 在线重建
    int batch_size = 32;
    int batch_num = (Ang + batch_size - 1) / batch_size;

    FdkReconstructor recon;
    recon.init(params, /*Kchunk=*/32, s);

    {
        YK::Util::CudaTimer timer("online", s);

        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams batch_params = params;
            batch_params.iPAng = count;
            batch_params.angle_list = std::vector<float>(
                angle_list.begin() + base,
                angle_list.begin() + base + count);

            recon.feed(
                h_proj.data() + base * view_elems,
                batch_params, s,
                d_vol_buf.data(),
                /*clear_vol=*/(i == 0), dump);
        }
    }


    // 新一轮扫描时
    recon.reset();

    auto h_vol_online = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    ctrl.download3D(h_vol_online, d_vol_buf);

    if (!write_raw_float("fdk_vec_vol_online2.raw",
        h_vol_online.cdata(), vol_elems)) {  // ← 修正变量名
        std::printf("Error: cannot write fdk_vec_vol_online.raw\n");
        return -2;
    }




    YK_CUDA_CHECK(cudaStreamDestroy(s));

    std::printf("Done: wrote fdk_vec_vol_online.raw (%d x %d x %d)\n", Nx, Ny, Nz);
    return 0;
}


static void test_fp_runner(cudaStream_t stream)
{
    printf("\n[FpReconstructor] test\n");

    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.25f, vox_z = 0.25f;
    constexpr int   Na = 360, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    // 构建 SCBCTParams
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
        params.angle_list[i] = CUDA_PI * 3.f / 2.f + 2.f * CUDA_PI * i / Na;

    // 加载体积
    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    if (!read_raw_float("fdk_vec_vol_offline.raw", h_vol)) return;
    printf("  volume loaded\n");

    float* d_vol = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
        h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));
    h_vol.clear();

    // 分配输出
    const size_t sino_elems = (size_t)Na * Nv * Nu;
    float* d_sino = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));

    YK::FpReconstructor fpr;

    if (!fpr.init(params, 0))
    {
        YK_LOGE("FpReconstructor init failed");
    }

    // 运行
    YK_LOGI("  running fp_project...\n");
    bool ok = fpr.run(d_vol, params, d_sino, stream);
    //bool ok = YK::fp_project(d_vol, d_sino, params, stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    YK_LOGI("  fp_project: %s\n", ok ? "OK" : "FAILED");

    // 回读统计
    std::vector<float> h_sino(sino_elems);
    YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
        sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

    float maxv = *std::max_element(h_sino.begin(), h_sino.end());
    float sumv = 0.f;
    for (auto x : h_sino) sumv += x;
    printf("  sino: max=%.4f  sum=%.3e\n", maxv, sumv);

    for (int a = 0; a < Na; ++a) {
        float mv = 0.f;
        for (size_t k = 0; k < (size_t)Nv * Nu; ++k)
            mv = fmaxf(mv, h_sino[a * (size_t)Nv * Nu + k]);
        if (mv > 1e-6f || a < 3 || a >= Na - 3)
            printf("  angle %3d: max=%.4f\n", a, mv);
    }

    write_raw_float("fp_reconstructor_sino.raw", h_sino.data(), sino_elems);
    printf("  saved: fp_reconstructor_sino.raw\n");

    cudaFree(d_vol);
    cudaFree(d_sino);
}

int main() {
    Logger::instance().set_level(LogLevel::Debug);
    //YKTest::testFFT();
    main_fdk();
    test_fp_runner(0);
    //forward_project_example();
    //YKTest::testFilterWeightsSpectra_RamLak();
    //YKTest::test_gpumem3d();
    //YKTest::test_mem_data_integration_wrap();
    //YKTest::test_cpu_wrap_copy();

    return 0;
}



//#include <cmath>
//#include <cstdio>
//#include <cuda_runtime.h>
//#include <vector>
//#include <cmath>
//#include <cstdio>
//#include <vector>
//#include <cstdio>
//#include <vector>
//
//#include "YkFDKFilter.hpp"    // YK::FilterManager + YK::_kernel_pad_with_offset
//#include "YkGlobals.h"
//#include "YkFDKPreweight.hpp"        // YK::PreweightManager
//#include "YkFDKBackProjection.hpp"
//
//static bool read_raw_float(const char* path, std::vector<float>& data) {
//    FILE* fp = std::fopen(path, "rb");
//    if (!fp) return false;
//    size_t n = std::fread(data.data(), sizeof(float), data.size(), fp);
//    std::fclose(fp);
//    return n == data.size();
//}
//
//static bool write_raw_float(const char* path, const std::vector<float>& data) {
//    FILE* fp = std::fopen(path, "wb");
//    if (!fp) return false;
//    size_t n = std::fwrite(data.data(), sizeof(float), data.size(), fp);
//    std::fclose(fp);
//    return n == data.size();
//}
//
//int main() {
//    // ------------------- dims / params (from your snippet) -------------------
//    SDimensions3D dims;
//    dims.iProjU = 256; dims.iProjV = 256; dims.iProjAngles = 360;
//    dims.iVolX = 512; dims.iVolY = 512; dims.iVolZ = 100;
//
//    float SID = 500.0f;
//    float SDD = 1000.0f;
//    float du = 1.0f;
//    float dv = 1.0f;
//    float voxel_size = 0.5f;
//
//    const int Nu = (int)dims.iProjU;
//    const int Nv = (int)dims.iProjV;
//    const int Ang = (int)dims.iProjAngles;
//    const int Nx = (int)dims.iVolX;
//    const int Ny = (int)dims.iVolY;
//    const int Nz = (int)dims.iVolZ;
//
//    const size_t proj_elems = (size_t)Nu * Nv * Ang;
//    const size_t vol_elems = (size_t)Nx * Ny * Nz;
//
//    // ------------------- read projection (assume A-V-U contiguous) -------------------
//    std::vector<float> h_proj(proj_elems);
//    if (!read_raw_float("cat515_projection.raw", h_proj)) {
//        std::printf("Error: cannot read cat515_projection.raw (expected %zu floats)\n", proj_elems);
//        return -1;
//    }
//    std::printf("Read projection: %zu floats\n", proj_elems);
//
//    // ------------------- allocate volume on device -------------------
//    cudaStream_t stream;
//    YK_CUDA_CHECK(cudaStreamCreate(&stream));
//
//    float* d_vol = nullptr;
//    YK_CUDA_CHECK(cudaMalloc(&d_vol, vol_elems * sizeof(float)));
//
//    // ------------------- run streaming FDK (preweight + filter + BP) -------------------
//    const float offsetU = 0.0f;  // pixel
//    const float offsetV = 0.0f;  // pixel
//    const int   Kchunk = 8;     // 8 or 16 recommended (trade memory vs speed)
//
//    YK::fdk_recon_streaming(
//        h_proj.data(),      // host proj [Ang][Nv][Nu]
//        d_vol,              // device vol [Nz][Ny][Nx]
//        Nu, Nv, Ang,
//        Nx, Ny, Nz, voxel_size,
//        SID, SDD, du, dv,
//        offsetU, offsetV,
//        Kchunk,
//        stream
//    );
//
//    // sync once at end
//    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//
//    // ------------------- copy volume back and write -------------------
//    std::vector<float> h_vol(vol_elems);
//    YK_CUDA_CHECK(cudaMemcpy(h_vol.data(), d_vol, vol_elems * sizeof(float), cudaMemcpyDeviceToHost));
//
//    if (!write_raw_float("fdk_vol.raw", h_vol)) {
//        std::printf("Error: cannot write fdk_vol.raw\n");
//        return -2;
//    }
//    std::printf("Wrote volume: fdk_vol.raw (%u x %u x %u)\n", dims.iVolX, dims.iVolY, dims.iVolZ);
//
//    // ------------------- cleanup -------------------
//    YK_CUDA_CHECK(cudaFree(d_vol));
//    YK_CUDA_CHECK(cudaStreamDestroy(stream));
//
//    return 0;
//}

//// 如果你的滤波流程输出的是 padded 宽度，需要 crop 回 Nu（可选）
//__global__ void kernel_crop_u_2d(
//    const float* __restrict__ src, // [Nv * paddedN]
//    float* __restrict__ dst,       // [Nv * Nu]
//    int Nu, int Nv,
//    int paddedN, int start_u)
//{
//    int u = blockIdx.x * blockDim.x + threadIdx.x;
//    int v = blockIdx.y * blockDim.y + threadIdx.y;
//    if (u >= Nu || v >= Nv) return;
//
//    int su = start_u + u;
//    float val = 0.0f;
//    if (su >= 0 && su < paddedN) val = src[v * paddedN + su];
//    dst[v * Nu + u] = val;
//}
//
//int test_preweight_and_filter_row_contiguous()
//{
//    // ------------------- 你给的参数 -------------------
//    SDimensions3D dims;
//    dims.iProjU = 256; dims.iProjV = 256; dims.iProjAngles = 360;
//    dims.iVolX = 512; dims.iVolY = 512; dims.iVolZ = 100;
//
//    float SID = 500.0f;
//    float SDD = 1000.0f;
//    float du = 1.0f;
//    float dv = 1.0f;
//    (void)SID;
//
//    const int Nu = (int)dims.iProjU;
//    const int Nv = (int)dims.iProjV;
//    const int Ang = (int)dims.iProjAngles;
//
//    const size_t view_elems = (size_t)Nu * Nv;
//    const size_t proj_elems = view_elems * Ang;
//
//    // ------------------- 读入 raw（假设 A-V-U 连续） -------------------
//    std::vector<float> h_proj(proj_elems);
//    FILE* fp = std::fopen("cat515_projection.raw", "rb");
//    if (!fp) { std::printf("Error: Cannot find cat515_projection.raw\n"); return -1; }
//    size_t nread = std::fread(h_proj.data(), sizeof(float), proj_elems, fp);
//    std::fclose(fp);
//    if (nread != proj_elems) std::printf("Warning: read %zu/%zu floats\n", nread, proj_elems);
//
//    std::vector<float> h_out(proj_elems, 0.0f);
//
//    // ------------------- CUDA stream -------------------
//    cudaStream_t s;
//    YK_CUDA_CHECK(cudaStreamCreate(&s));
//
//    // ------------------- Device buffers (复用) -------------------
//    float* d_in = nullptr; // [Nv*Nu]
//    float* d_pw = nullptr; // [Nv*Nu]
//    float* d_out = nullptr; // [Nv*Nu]
//    YK_CUDA_CHECK(cudaMalloc(&d_in, view_elems * sizeof(float)));
//    YK_CUDA_CHECK(cudaMalloc(&d_pw, view_elems * sizeof(float)));
//    YK_CUDA_CHECK(cudaMalloc(&d_out, view_elems * sizeof(float)));
//
//    // ------------------- 1) 预加权 init -------------------
//    // DSD=SDD (源到探测器距离)
//    YK::PreweightManager pw;
//    pw.init(
//        Nu, Nv, /*batch=*/1,
//        du, dv,
//        /*DSD=*/SDD,
//        /*offsetU=*/0.0f, /*offsetV=*/0.0f,
//        /*power=*/1,      // 1: cos
//        s
//    );
//
//    // ------------------- 2) 滤波 init：对每一行做 1D，所以 batch=Nv -------------------
//    YK::FilterManager fm;
//    fm.init(/*Nu=*/Nu, /*du=*/du, /*batch=*/Nv, s);
//    const int paddedN = fm.getPaddedN();
//
//    float* d_padded = nullptr; // [Nv * paddedN]
//    YK_CUDA_CHECK(cudaMalloc(&d_padded, (size_t)Nv * paddedN * sizeof(float)));
//
//    // pad/crop 对齐参数（offsetX=0）
//    const float offsetX = 0.0f;
//    const float axis_idx = (Nu - 1) * 0.5f + offsetX;
//    const int start_u = (int)lrintf(paddedN * 0.5f - axis_idx);
//
//    // ------------------- 逐角处理 -------------------
//    for (int a = 0; a < Ang; ++a) {
//        const float* h_view = h_proj.data() + (size_t)a * view_elems;
//        float* h_dst = h_out.data() + (size_t)a * view_elems;
//
//        // H2D
//        YK_CUDA_CHECK(cudaMemcpyAsync(d_in, h_view, view_elems * sizeof(float),
//            cudaMemcpyHostToDevice, s));
//
//        // 预加权（cos）
//        pw.setStream(s);
//        pw.apply(d_in, d_pw);
//
//        // 行滤波：把 Nv 行当 batch，每行长度 Nu
//        // pad: [Nv*Nu] -> [Nv*paddedN]
//        fm.setStream(s);
//
//        dim3 block1(256, 1);
//        dim3 grid1((paddedN + 255) / 256, Nv);
//        YK::_kernel_pad_with_offset << <grid1, block1, 0, s >> > (d_pw, d_padded, Nu, Nv, paddedN, offsetX);
//        YK_CUDA_KERNEL_CHECK();
//
//        // FFT filter in-place on padded
//        fm.apply(d_padded);
//
//        // crop 回 Nu（如果你后面 BP 直接用 padded，也可以不 crop）
//        dim3 block2(16, 16);
//        dim3 grid2((Nu + block2.x - 1) / block2.x, (Nv + block2.y - 1) / block2.y);
//        kernel_crop_u_2d << <grid2, block2, 0, s >> > (d_padded, d_out, Nu, Nv, paddedN, start_u);
//        YK_CUDA_KERNEL_CHECK();
//
//        // D2H
//        YK_CUDA_CHECK(cudaMemcpyAsync(h_dst, d_out, view_elems * sizeof(float),
//            cudaMemcpyDeviceToHost, s));
//    }
//
//    // 统一同步（系统级同步点）
//    YK_CUDA_CHECK(cudaStreamSynchronize(s));
//
//    // 写出
//    FILE* fo = std::fopen("cat515_preweight_filter.raw", "wb");
//    if (!fo) { std::printf("Error: cannot write output\n"); return -2; }
//    std::fwrite(h_out.data(), sizeof(float), proj_elems, fo);
//    std::fclose(fo);
//
//    // cleanup
//    YK_CUDA_CHECK(cudaFree(d_in));
//    YK_CUDA_CHECK(cudaFree(d_pw));
//    YK_CUDA_CHECK(cudaFree(d_out));
//    YK_CUDA_CHECK(cudaFree(d_padded));
//    YK_CUDA_CHECK(cudaStreamDestroy(s));
//
//    std::printf("Done: cat515_preweight_filter.raw (A-V-U layout assumed)\n");
//    return 0;
//}
//
//int main() {
//    return test_preweight_and_filter_row_contiguous();
//}


//#include <cstdlib>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <cufft.h>
//#include <driver_types.h>
//#include <iomanip>
//#include <ios>
//#include <iostream>
//#include <vector>
//#include "YKtestconv.hpp"
//#include "YkConv.hpp" // 确保包含你的命名空间定义
//#include "YkFFT.hpp"
//#include "YKFDKFilter.hpp"
//
//int conv_test_std() {
//    // --- 参数设置 ---
//    const int Nu = 64;
//    const int Nv = 1;
//    const int Ntheta = 1;        // Batch 设为 1
//    const float du = 1.0f;
//    const float offsetX = 0.0f;
//
//    // --- 1. 初始化 FilterManager (核心复用对象) ---
//    YK::FilterManager filter_mgr;
//    if (!filter_mgr.init(Nu, du, Ntheta)) {
//        std::cerr << "FilterManager init failed!" << std::endl;
//        return -1;
//    }
//
//    // 获取由管理器自动计算的补零长度 (应该是 128)
//    int paddedN = filter_mgr.getPaddedN();
//
//    // --- 2. 准备数据 ---
//    std::vector<float> h_input(Nu, 0.0f);
//    h_input[Nu / 2] = 1.0f; // 中心脉冲
//
//    float* d_input, * d_output_padded;
//    cudaMalloc(&d_input, Nu * sizeof(float));
//    cudaMalloc(&d_output_padded, paddedN * sizeof(float));
//
//    cudaMemcpy(d_input, h_input.data(), Nu * sizeof(float), cudaMemcpyHostToDevice);
//
//    // --- 3. 执行物理对齐与滤波 ---
//    // A. 物理对齐补零 (这一步每次投影都需要根据 offsetX 计算)
//    dim3 block(256, 1);
//    dim3 grid_pad((paddedN + 255) / 256, Ntheta);
//    YK::_kernel_pad_with_offset << <grid_pad, block >> > (d_input, d_output_padded, Nu, Ntheta, paddedN, offsetX);
//
//    // B. 复用 Plan 和权重执行滤波 (Zero-Allocation 过程)
//    filter_mgr.apply(d_output_padded);
//
//    // --- 4. 执行 CPU 空间域验证 (保持不变) ---
//    std::vector<float> spatial_kernel = YK_Test::generateSpatialRLKernel(Nu, du);
//    std::vector<float> cpu_result = YK_Test::cpuConvolve(h_input, spatial_kernel);
//
//    // --- 5. 结果对比 ---
//    std::vector<float> gpu_result_full(paddedN);
//    cudaMemcpy(gpu_result_full.data(), d_output_padded, paddedN * sizeof(float), cudaMemcpyDeviceToHost);
//
//    int gpu_center = paddedN / 2;
//    int cpu_center = Nu / 2;
//
//    std::cout << std::fixed << std::setprecision(6);
//    std::cout << "Comparison using FilterManager (Center +/- 7 elements):" << std::endl;
//    std::cout << "Index | CPU (Spatial) | GPU (FFT-Reuse) | Difference" << std::endl;
//    std::cout << "--------------------------------------------------------" << std::endl;
//
//    for (int i = -7; i <= 7; ++i) {
//        float val_cpu = cpu_result[cpu_center + i];
//        float val_gpu = gpu_result_full[gpu_center + i];
//        std::cout << std::setw(5) << i << " | "
//            << std::setw(13) << val_cpu << " | "
//            << std::setw(14) << val_gpu << " | "
//            << std::setw(10) << std::abs(val_cpu - val_gpu) << std::endl;
//    }
//
//    // 清理
//    cudaFree(d_input);
//    cudaFree(d_output_padded);
//    // filter_mgr 会在析构时自动释放内部资源
//    return 0;
//}
//
//
//
//int main() {
//    // --- 参数设置 ---
//    const int Nu = 64;
//    const int Nv = 1;
//    const int Ntheta = 1;
//    const float du = 1.0f;
//
//    // 【修改点 1】设置偏位测试值：假设旋转轴偏离中心 +2.0 像素
//    const float offsetX = 2.0f;
//
//    // --- 1. 初始化 FilterManager ---
//    YK::FilterManager filter_mgr;
//    if (!filter_mgr.init(Nu, du, Ntheta)) {
//        std::cerr << "FilterManager init failed!" << std::endl;
//        return -1;
//    }
//
//    int paddedN = filter_mgr.getPaddedN();
//
//    // --- 2. 准备数据 (中心脉冲) ---
//    std::vector<float> h_input(Nu, 0.0f);
//    h_input[Nu / 2] = 1.0f;
//
//    float* d_input, * d_output_padded;
//    cudaMalloc(&d_input, Nu * sizeof(float));
//    cudaMalloc(&d_output_padded, paddedN * sizeof(float));
//    cudaMemcpy(d_input, h_input.data(), Nu * sizeof(float), cudaMemcpyHostToDevice);
//
//    // --- 3. 执行物理对齐与滤波 ---
//    dim3 block(256, 1);
//    dim3 grid_pad((paddedN + 255) / 256, Ntheta);
//
//    // 【修改点 2】调用带 Offset 的补零 Kernel，此时旋转轴将对齐到 paddedN/2
//    YK::_kernel_pad_with_offset << <grid_pad, block >> > (d_input, d_output_padded, Nu, Ntheta, paddedN, offsetX);
//
//    filter_mgr.apply(d_output_padded);
//
//    // --- 4. 执行 CPU 空间域验证 (作为原始无偏位参考) ---
//    std::vector<float> spatial_kernel = YK_Test::generateSpatialRLKernel(Nu, du);
//    std::vector<float> cpu_result = YK_Test::cpuConvolve(h_input, spatial_kernel);
//
//    // --- 5. 结果对比 ---
//    std::vector<float> gpu_result_full(paddedN);
//    cudaMemcpy(gpu_result_full.data(), d_output_padded, paddedN * sizeof(float), cudaMemcpyDeviceToHost);
//
//    int gpu_center = paddedN / 2;
//    int cpu_center = Nu / 2;
//
//    std::cout << std::fixed << std::setprecision(6);
//    std::cout << "Offset Verification (offsetX = " << offsetX << "):" << std::endl;
//    std::cout << "Index | CPU (Ref) | GPU (Shifted) | Note" << std::endl;
//    std::cout << "--------------------------------------------------------" << std::endl;
//
//    for (int i = -7; i <= 7; ++i) {
//        float val_cpu = cpu_result[cpu_center + i];
//        float val_gpu = gpu_result_full[gpu_center + i];
//
//        std::cout << std::setw(5) << i << " | "
//            << std::setw(9) << val_cpu << " | "
//            << std::setw(13) << val_gpu << " | ";
//
//        // 【修改点 3】逻辑判断：如果对齐成功，GPU 的峰值应该出现在 i = -offsetX 的位置
//        if (i == (int)(-offsetX)) {
//            std::cout << " <-- Shifted Peak (Aligned to Center)";
//        }
//        std::cout << std::endl;
//    }
//
//    cudaFree(d_input);
//    cudaFree(d_output_padded);
//    return 0;
//}
//
