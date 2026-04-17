// test_Task_dll.cpp
// 测试 TaskFactory 创建 FDK 和 FP 任务，验证 DLL 接口完整性

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>


#include <cuda_runtime_api.h>
#include "../../global/YkLog.h"
#include "../../interface/IYkTask.hpp"
#include "../../interface/YkTaskFactory.hpp"
#include "../../interface/YkTaskTypes.hpp"


#pragma comment(lib, "YKCBCT.lib")

// ----------------------------------------------------------------
// 工具函数
// ----------------------------------------------------------------
static bool loadRaw(const char* path, std::vector<float>& buf, size_t count)
{
    buf.resize(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[loadRaw] cannot open %s\n", path); return false; }
    f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
    return (size_t)f.gcount() == count * sizeof(float);
}

static bool saveRaw(const char* path, const float* data, size_t count)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[saveRaw] cannot open %s\n", path); return false; }
    f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    return true;
}

static void printStats(const std::vector<float>& v, const char* tag)
{
    float mn = *std::min_element(v.begin(), v.end());
    float mx = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (auto x : v) sum += x;
    printf("  [%s] min=%.4f  max=%.4f  sum=%.3e  count=%zu\n",
        tag, mn, mx, sum, v.size());
}

// ----------------------------------------------------------------
// 公共几何参数，FDK 和 FP 共用
// ----------------------------------------------------------------
static YK::SScanParams makeScanParams()
{
    YK::SScanParams s{};
    s.Nu = 1024;
    s.Nv = 1024;
    s.du_mm = 0.25f;
    s.dv_mm = 0.25f;
    s.SOD_mm = 500.f;
    s.SDD_mm = 1000.f;
    s.scanRangeRad = 2.f * 3.14159265f;
    s.startAngleRad = 3.14159265f * 1.5f;
    s.shortScan = false;
    s.NAng = 360; // 仅 FDK 用，表示总视图数（非批次大小）
    return s;
}

static YK::SVolumeParams makeVolumeParams()
{
    YK::SVolumeParams v{};
    v.Nx = 512;
    v.Ny = 512;
    v.Nz = 400;
    v.voxX_mm = 0.25f;
    v.voxY_mm = 0.25f;
    v.voxZ_mm = 0.25f;
    return v;
}

// ----------------------------------------------------------------
// 测试1：FP 正投影
// ----------------------------------------------------------------
static void test_fp_task()
{
    printf("\n[DLL Test1] FP_Joseph forward projection\n");

    constexpr int Na = 360, Nu = 1024, Nv = 1024;
    constexpr int Nx = 512, Ny = 512, Nz = 400;

    // --- 加载体积 ---
    std::vector<float> h_vol;
    if (!loadRaw("fdk_vec_vol_offline.raw", h_vol,
        (size_t)Nx * Ny * Nz)) return;
    printf("  volume loaded\n");

    void* d_vol = nullptr;
    cudaMalloc(&d_vol, h_vol.size() * sizeof(float));
    cudaMemcpy(d_vol, h_vol.data(),
        h_vol.size() * sizeof(float), cudaMemcpyHostToDevice);
    h_vol.clear();

    // --- 分配输出 ---
    const size_t sino_elems = (size_t)Na * Nv * Nu;
    void* d_sino = nullptr;
    cudaMalloc(&d_sino, sino_elems * sizeof(float));
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));

    // --- 构建角度 ---
    std::vector<float> angles(Na);
    for (int i = 0; i < Na; ++i)
        angles[i] = 3.14159265f * 1.5f * 0 + 2.f * 3.14159265f * i / Na;

    // --- 创建任务 ---
    YK::ITask* task = YK::TaskFactory::create(YK::ETask::FP_Joseph);
    if (!task) { fprintf(stderr, "  create failed\n"); return; }

    // --- init ---
    YK::SFpAlgoParams fpAlgo{};
    fpAlgo.stepSS = YK::EFpStepSample::x1;
    fpAlgo.detSS = YK::EFpDetSample::x1;

    YK::TaskInitParams initP{};
    initP.task = YK::ETask::FP_Joseph;
    initP.scan = makeScanParams();
    initP.volume = makeVolumeParams();
    initP.algoParams = &fpAlgo;
    initP.algoParamSize = sizeof(fpAlgo);

    if (!task->init(initP)) {
        fprintf(stderr, "  init failed\n");
        YK::TaskFactory::destroy(task);
        return;
    }
    printf("  init OK\n");

    // --- run ---
    YK::TaskBatchParams batchP{};
    batchP.d_vol_in = (float*)d_vol;
    batchP.d_sino_out = (float*)d_sino;
    batchP.h_angles = angles.data();
    batchP.K = Na;
    batchP.clearOut = true;

    if (!task->run(batchP)) {
        fprintf(stderr, "  run failed\n");
        YK::TaskFactory::destroy(task);
        return;
    }
    printf("  run OK\n");

    // --- 回读统计 ---
    std::vector<float> h_sino(sino_elems);
    cudaMemcpy(h_sino.data(), d_sino,
        sino_elems * sizeof(float), cudaMemcpyDeviceToHost);
    printStats(h_sino, "fp sino");

    saveRaw("dll_test1_fp_sino.raw", h_sino.data(), sino_elems);
    printf("  saved: dll_test1_fp_sino.raw\n");

    // --- 清理 ---
    YK::TaskFactory::destroy(task);
    cudaFree(d_vol);
    cudaFree(d_sino);
}

// ----------------------------------------------------------------
// 测试2：FDK 重建
// ----------------------------------------------------------------
static void test_fdk_task()
{
    printf("\n[DLL Test2] FDK reconstruction\n");

    constexpr int Na = 360, Nu = 1024, Nv = 1024;
    constexpr int Nx = 512, Ny = 512, Nz = 400;

    // --- 加载正弦图 ---
    std::vector<float> h_sino;
    if (!loadRaw("dll_test1_fp_sino.raw", h_sino,
        (size_t)Na * Nv * Nu)) {
        fprintf(stderr, "  sino not found, run test1 first\n");
        return;
    }
    printf("  sino loaded\n");
    printStats(h_sino, "input sino");

    // --- 分配输出体积 ---
    void* d_vol = nullptr;
    cudaMalloc(&d_vol, (size_t)Nx * Ny * Nz * sizeof(float));

    // --- 构建角度 ---
    std::vector<float> angles(Na);
    for (int i = 0; i < Na; ++i)
        angles[i] = 3.14159265f * 1.5f + 2.f * 3.14159265f * i / Na;

    // --- 创建任务 ---
    YK::ITask* task = YK::TaskFactory::create(YK::ETask::FDK);
    if (!task) { fprintf(stderr, "  create failed\n"); return; }

    // --- init ---
    YK::SFdkAlgoParams fdkAlgo{};
    fdkAlgo.filter = YK::EFdkFilter::RamLak;

    YK::TaskInitParams initP{};
    initP.task = YK::ETask::FDK;
    initP.scan = makeScanParams();
    initP.volume = makeVolumeParams();
    initP.algoParams = &fdkAlgo;
    initP.algoParamSize = sizeof(fdkAlgo);

    if (!task->init(initP)) {
        fprintf(stderr, "  init failed\n");
        YK::TaskFactory::destroy(task);
        return;
    }
    printf("  init OK\n");

    // --- run ---
    YK::TaskBatchParams batchP{};
    batchP.h_proj = h_sino.data();
    batchP.d_vol_out = (float*)d_vol;
    batchP.h_angles = angles.data();
    batchP.K = Na;
    batchP.clearOut = true;

    YK::TaskDumpCallback dumpCb = [](void*, int a, const char* tag,
        float* d_buf, size_t n)
        {
            if (a != 0) return;

            std::vector<float> h(n);
            cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);

            float sum = 0.f, maxv = -1e30f, minv = 1e30f;
            for (auto v : h) {
                sum += v;
                maxv = std::max(maxv, v);
                minv = std::min(minv, v);
            }

            //YK_LOGI("[dump][a={}][{}] n={} min={:.4f} max={:.4f} mean={:.6f}",
            //    a, tag, n, minv, maxv, sum / (float)n);

            //auto path = fmt::format("dump_a{}_{}.raw", a, tag);
            auto path = std::string("dump_") + tag + ".raw";
            std::ofstream f(path, std::ios::binary);
            if (f)
                f.write(reinterpret_cast<const char*>(h.data()), n * sizeof(float));
            else
                /*YK_LOGE("[dump] cannot save {}", path);*/
                printf("  cannot save %s\n", path.c_str());
        };


    if (!task->run(batchP, dumpCb)) {
        fprintf(stderr, "  run failed\n");
        YK::TaskFactory::destroy(task);
        return;
    }
    printf("  run OK\n");

    // --- 回读统计 ---
    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    cudaMemcpy(h_vol.data(), d_vol,
        h_vol.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printStats(h_vol, "recon vol");

    saveRaw("dll_test2_fdk_vol.raw", h_vol.data(), h_vol.size());
    printf("  saved: dll_test2_fdk_vol.raw\n");

    // --- 清理 ---
    YK::TaskFactory::destroy(task);
    cudaFree(d_vol);
}

// ----------------------------------------------------------------
// 测试3：reset 后重复 run
// ----------------------------------------------------------------
static void test_reset()
{
    printf("\n[DLL Test3] reset and re-run\n");

    constexpr int Na = 10, Nu = 64, Nv = 64;
    constexpr int Nx = 64, Ny = 64, Nz = 64;

    std::vector<float> h_vol(Nx * Ny * Nz, 1.f);
    void* d_vol = nullptr;
    cudaMalloc(&d_vol, h_vol.size() * sizeof(float));
    cudaMemcpy(d_vol, h_vol.data(),
        h_vol.size() * sizeof(float), cudaMemcpyHostToDevice);

    const size_t sino_elems = (size_t)Na * Nv * Nu;
    void* d_sino = nullptr;
    cudaMalloc(&d_sino, sino_elems * sizeof(float));

    std::vector<float> angles(Na);
    for (int i = 0; i < Na; ++i)
        angles[i] = 2.f * 3.14159265f * i / Na;

    YK::ITask* task = YK::TaskFactory::create(YK::ETask::FP_Joseph);

    YK::SScanParams scan{};
    scan.Nu = Nu; scan.Nv = Nv;
    scan.du_mm = 1.f; scan.dv_mm = 1.f;
    scan.SOD_mm = 500.f; scan.SDD_mm = 1000.f;
    scan.scanRangeRad = 2.f * 3.14159265f;

    YK::SVolumeParams vol{};
    vol.Nx = Nx; vol.Ny = Ny; vol.Nz = Nz;
    vol.voxX_mm = vol.voxY_mm = vol.voxZ_mm = 0.1f;

    YK::TaskInitParams initP{};
    initP.task = YK::ETask::FP_Joseph;
    initP.scan = scan;
    initP.volume = vol;
    task->init(initP);

    YK::TaskBatchParams batchP{};
    batchP.d_vol_in = (float*)d_vol;
    batchP.d_sino_out = (float*)d_sino;
    batchP.h_angles = angles.data();
    batchP.K = Na;
    batchP.clearOut = true;

    // 第一次 run
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));
    task->run(batchP);
    std::vector<float> h_sino1(sino_elems);
    cudaMemcpy(h_sino1.data(), d_sino,
        sino_elems * sizeof(float), cudaMemcpyDeviceToHost);
    printStats(h_sino1, "run1");

    // reset 后第二次 run
    //task->reset();
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));
    task->run(batchP);
    std::vector<float> h_sino2(sino_elems);
    cudaMemcpy(h_sino2.data(), d_sino,
        sino_elems * sizeof(float), cudaMemcpyDeviceToHost);
    printStats(h_sino2, "run2");

    // 两次结果应完全一致
    double maxDiff = 0.0;
    for (size_t k = 0; k < sino_elems; ++k)
        maxDiff = std::max(maxDiff, (double)std::abs(h_sino1[k] - h_sino2[k]));
    printf("  run1 vs run2: maxDiff=%.2e  %s\n",
        maxDiff, maxDiff < 1e-4 ? "OK" : "MISMATCH");

    YK::TaskFactory::destroy(task);
    cudaFree(d_vol);
    cudaFree(d_sino);
}

// ----------------------------------------------------------------
// 主入口
// ----------------------------------------------------------------
int main()
{
    printf("=== TaskFactory DLL Interface Test ===\n");

    test_fp_task();
    test_fdk_task();
    test_reset();

    cudaDeviceReset();
    printf("\n=== all done ===\n");
    return 0;
}