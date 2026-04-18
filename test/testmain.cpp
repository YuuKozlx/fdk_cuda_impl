
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "FDK/YkFdkReconstructor.hpp"
#include "FDK/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"



#include "global/YkMem3d.hpp"
#include "test/Yktest_dataobject.hpp"
#include "test/Yktest_fdkflter.hpp"
#include "test/Yktest_fft.hpp"
#include "test/Yktest_mem3d.hpp"
#include "util/YkCudaTimer.hpp"
#include "util/YkVecOperation.hpp"


#include <fstream>
#include <string>
#include "FP/YkFPRunner.hpp"
#include "interface/YkTaskTypes.hpp"
#include "BP/YkBPRunner.hpp"




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
    params.du_mm = 0.25f, params.dv_mm = 0.25f, params.vox_x_mm = 0.1f;
    params.vox_y_mm = 0.1f;
    params.vox_z_mm = 0.1f;


    params.offsetU_mm = 0 * params.du_mm;
    const int Ang = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;

    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // 读投影
    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float("proj_1024x1024x360.raw", h_proj)) {
        YK_LOGE("Error: cannot read proj_1024x1024x360 (expect %zu floats)\n", proj_elems);
        return -1;
    }

    // CUDA 资源
    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));



    // 调用
    auto dump = [](void* p) {
        auto* payload = static_cast<DumpPayload*>(p);


        const int    a = payload->viewIdx;
        const char* tag = payload->stage;

        float* d_buf = static_cast<float*>(payload->buf);
        const size_t n = payload->n;

        if (a != 0) return;

        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.f, maxv = -1e30f, minv = 1e30f;
        for (auto v : h) {
            sum += v;
            maxv = std::max(maxv, v);
            minv = std::min(minv, v);
        }

        YK_LOGI("[dump][a={}][{}] n={} min={:.4f} max={:.4f} mean={:.6f}",
            a, tag, n, minv, maxv, sum / (float)n);

        auto path = fmt::format("dump_a{}_{}.raw", a, tag);
        std::ofstream f(path, std::ios::binary);
        if (f)
            f.write(reinterpret_cast<const char*>(h.data()), n * sizeof(float));
        else
            YK_LOGE("[dump] cannot save {}", path);
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
            /*clear_vol=*/true, dump, nullptr);
    }



    auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);

    ctrl.download3D(h_vol, d_vol_buf);


    uint64_t total_elements = (uint64_t)Nx * Ny * Nz;
    if (!write_raw_float("fdk_vec_vol_offline.raw", h_vol.cdata(), total_elements)) {
        YK_LOGE("Error: cannot write fdk_vec_vol_new.raw\n");
        return -2;
    }

    // 模拟在线重建，每次传输iPBatch个角度，进行重建。保持KChunk=30不变，测试在线重建的正确性和性能。
    // 待更改参数 
    // 1. params.iPAng = iPBatch，角度list也相应缩减为当前批次的角度
    // 2. 每次循环传入的数据指针起点偏置
    // 3. 仅在第一批时 clear_vol=true，后续批次 clear_vol=false

    // batch_size = 60; batch_num = Ang + batch_size - 1) / batch_size;;
    // 在线重建
    int batch_size = 32 * 3;
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
                /*clear_vol=*/(i == 0));
        }
    }


    // 新一轮扫描时
    recon.reset();

    auto h_vol_online = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
    ctrl.download3D(h_vol_online, d_vol_buf);

    if (!write_raw_float("fdk_vec_vol_online.raw",
        h_vol_online.cdata(), total_elements)) {  // ← 修正变量名
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
    constexpr float vox_xy = 0.1f, vox_z = 0.1f;
    constexpr int   Na = 480, Nu = 1024, Nv = 1024;
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
        params.angle_list[i] = 2.f * CUDA_PI * i / 720;

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

    if (!fpr.init(params, ETask::FP_Joseph, 0))
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

int main_bp_runner() {
    SCBCTParams params;

    params.iPU = 1024; params.iPV = 1024;
    params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0 * CUDA_PI / 180;

    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = false;
    params.scan_range_rad = 2.0f * (float)M_PI;

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)M_PI / params.iPAng;

    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f; params.vox_y_mm = 0.1f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;

    const int Ang = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // ---- 读已滤波投影（假设外部已完成 preweight + filter）----
    std::vector<float> h_flt(proj_elems);
    if (!read_raw_float("proj_1024x1024x360.raw", h_flt)) {
        YK_LOGE("Error: cannot read flt_1024x1024x480.raw (expect {} floats)", proj_elems);
        return -1;
    }

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    MemoryController ctrl;

    // 上传已滤波投影到 GPU
    auto d_flt_buf = ctrl.allocateDevice3D<float>(
        (size_t)params.iPU * params.iPV, Ang, 1, 0);
    {
        auto h_flt_view = ctrl.allocateCpu3D<float>(
            (size_t)params.iPU * params.iPV, Ang, 1, false);
        std::memcpy(h_flt_view.data(), h_flt.data(),
            proj_elems * sizeof(float));
        ctrl.upload3D(d_flt_buf, h_flt_view);
    }

    auto d_vol_buf = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // ----------------------------------------------------------------
    // 离线反投影（一次性全量）
    // ----------------------------------------------------------------
    {
        YK::BpReconstructor recon;
        recon.init(params, /*Kchunk=*/32, s);

        YK::Util::CudaTimer timer("bp_offline", s);
        recon.feed(
            d_flt_buf.data(),
            params, s,
            d_vol_buf.data(),
            /*clear_vol=*/true);
    }

    {
        auto h_vol = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol, d_vol_buf);
        if (!write_raw_float("bp_vol_offline.raw", h_vol.cdata(), vol_elems)) {
            YK_LOGE("Error: cannot write bp_vol_offline.raw");
            return -2;
        }
        YK_LOGI("saved: bp_vol_offline.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    // ----------------------------------------------------------------
    // 在线反投影（分包）
    // ----------------------------------------------------------------
    const int batch_size = 32 * 3;
    const int batch_num = (Ang + batch_size - 1) / batch_size;

    YK::BpReconstructor recon_online;
    recon_online.init(params, /*Kchunk=*/32, s);

    {
        YK::Util::CudaTimer timer("bp_online", s);

        for (int i = 0; i < batch_num; ++i) {
            const int base = i * batch_size;
            const int count = std::min(batch_size, Ang - base);

            SCBCTParams batch_params = params;
            batch_params.iPAng = count;
            batch_params.angle_list = std::vector<float>(
                angle_list.begin() + base,
                angle_list.begin() + base + count);

            recon_online.feed(
                d_flt_buf.data() + base * view_elems,
                batch_params, s,
                d_vol_buf.data(),
                /*clear_vol=*/(i == 0));
        }
    }

    recon_online.reset();

    {
        auto h_vol_online = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_online, d_vol_buf);
        if (!write_raw_float("bp_vol_online.raw",
            h_vol_online.cdata(), vol_elems)) {
            YK_LOGE("Error: cannot write bp_vol_online.raw");
            return -2;
        }
        YK_LOGI("saved: bp_vol_online.raw ({}x{}x{})", Nx, Ny, Nz);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

int main_bp_verify() {
    SCBCTParams params;

    params.iPU = 1024; params.iPV = 1024; params.iPAng = 480; params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0 * CUDA_PI / 180;

    params.iVX = 512; params.iVY = 512; params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)M_PI * 4.0f / 3.0f;

    std::vector<float> angle_list(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        angle_list[i] = i * 2.0f * (float)M_PI / 720;

    params.scan_start_angle_rad = angle_list[0];
    params.angle_list = angle_list;

    params.SID = 500.0f; params.SDD = 1000.0f;
    params.du_mm = 0.25f; params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f; params.vox_y_mm = 0.1f; params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;

    const int Ang = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t proj_elems = view_elems * Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));
    MemoryController ctrl;

    // ---- 读原始投影 ----
    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float("proj_1024x1024x360.raw", h_proj)) {
        YK_LOGE("Error: cannot read proj_1024x1024x480.raw (expect {} floats)", proj_elems);
        return -1;
    }
    YK_LOGI("proj loaded");

    auto d_vol_fdk = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    auto d_vol_bp = ctrl.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);

    // 预分配 CPU buffer 收集所有视图的滤波结果
    std::vector<float> h_flt_all(proj_elems, 0.f);

    // ----------------------------------------------------------------
    // 路径一：完整 FDK，同时 dump flt 阶段数据
    // ----------------------------------------------------------------
    {
        FdkReconstructor recon;
        recon.init(params, 32, s);

        struct MyDumpCtx {
            cudaStream_t stream;
            float* h_flt_all;
            size_t       view_elems;
        };

        MyDumpCtx ctx{ s, h_flt_all.data(), view_elems };

        TaskDumpCallback onDump = [](void* p) {
            auto* payload = static_cast<FdkDumpPayload*>(p);
            auto* ctx = static_cast<MyDumpCtx*>(payload->userdata);

            if (std::string(payload->stage) != "flt") return;

            cudaMemcpyAsync(
                ctx->h_flt_all + (size_t)payload->viewIdx * ctx->view_elems,
                payload->d_buf, payload->n * sizeof(float),
                cudaMemcpyDeviceToHost, ctx->stream);
            };

        recon.feed(h_proj.data(), params, s,
            d_vol_fdk.data(), true, onDump, &ctx);
    }

    // FDK 跑完立即同步并保存结果
    cudaStreamSynchronize(s);
    {
        auto h_vol_fdk = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_fdk, d_vol_fdk);
        if (!write_raw_float("verify_vol_fdk.raw", h_vol_fdk.cdata(), vol_elems)) {
            YK_LOGE("Error: cannot write verify_vol_fdk.raw");
            return -2;
        }
        YK_LOGI("saved: verify_vol_fdk.raw");

        // 顺便统计 flt dump 是否有效
        float maxv = *std::max_element(h_flt_all.begin(), h_flt_all.end());
        float minv = *std::min_element(h_flt_all.begin(), h_flt_all.end());
        YK_LOGI("h_flt_all stats: min={:.6f} max={:.6f}", minv, maxv);
    }

    // ----------------------------------------------------------------
    // 路径二：BpReconstructor，输入 flt dump 的结果
    // ----------------------------------------------------------------

    // 上传滤波投影到 GPU（用裸指针避免 3D 布局歧义）
    float* d_flt_raw = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_flt_raw, proj_elems * sizeof(float)));
    YK_CUDA_CHECK(cudaMemcpy(d_flt_raw, h_flt_all.data(),
        proj_elems * sizeof(float), cudaMemcpyHostToDevice));

    {
        YK::BpReconstructor recon;
        recon.init(params, 32, s);
        recon.feed(d_flt_raw, params, s, d_vol_bp.data(), true);
    }

    cudaStreamSynchronize(s);
    {
        auto h_vol_bp = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_bp, d_vol_bp);
        if (!write_raw_float("verify_vol_bp.raw", h_vol_bp.cdata(), vol_elems)) {
            YK_LOGE("Error: cannot write verify_vol_bp.raw");
            cudaFree(d_flt_raw);
            return -2;
        }
        YK_LOGI("saved: verify_vol_bp.raw");
    }

    cudaFree(d_flt_raw);

    // ----------------------------------------------------------------
    // 对比两个体数据
    // ----------------------------------------------------------------
    {
        auto h_vol_fdk = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        auto h_vol_bp = ctrl.allocateCpu3D<float>(Nx, Ny, Nz, false);
        ctrl.download3D(h_vol_fdk, d_vol_fdk);
        ctrl.download3D(h_vol_bp, d_vol_bp);

        double maxDiff = 0.0, mse = 0.0;
        for (size_t i = 0; i < vol_elems; ++i) {
            double diff = std::abs(
                (double)h_vol_fdk.cdata()[i] - (double)h_vol_bp.cdata()[i]);
            maxDiff = std::max(maxDiff, diff);
            mse += diff * diff;
        }
        mse /= (double)vol_elems;

        YK_LOGI("maxDiff={:.8f}  MSE={:.8e}", maxDiff, mse);

        if (maxDiff < 1e-5)
            YK_LOGI("PASS: BpReconstructor matches FdkReconstructor");
        else
            YK_LOGE("FAIL: maxDiff={:.8f} exceeds threshold", maxDiff);
    }

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}

int test_log() {
    std::string str = "world";
    YK_LOGI("hello,{}", str);

    return 0;
}

int main() {
    Logger::instance().set_level(LogLevel::Debug);
    //YKTest::testFFT();
    //main_fdk();
    //main_bp_runner();
    main_bp_verify();
    //test_fp_runner(0);
    //YKTest::testFilterWeightsSpectra_RamLak();
    //YKTest::test_gpumem3d();
    //YKTest::test_mem_data_integration_wrap();
    //YKTest::test_cpu_wrap_copy();
    //test_log();
    return 0;
}
