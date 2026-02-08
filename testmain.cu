
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecFDKBackProjection.hpp"
#include "YkFDKBackProjection.hpp"


#include "Yktest_fft.hpp"
#include "Yktest_fdkflter.hpp"


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

int main_fdk() {
    SDimensions3D dims;
    dims.iProjU = 256; dims.iProjV = 256; dims.iProjAngles = 360;
    dims.iVolX = 512; dims.iVolY = 512; dims.iVolZ = 100;

    float SID = 500.0f;
    float SDD = 1000.0f;
    float du = 1.0f, dv = 1.0f;
    float vox = 0.5f;

    int Nu = (int)dims.iProjU;
    int Nv = (int)dims.iProjV;
    int Ang = (int)dims.iProjAngles;
    int Nx = (int)dims.iVolX;
    int Ny = (int)dims.iVolY;
    int Nz = (int)dims.iVolZ;

    const size_t view_elems = (size_t)Nu * Nv;
    const size_t proj_elems = view_elems * (size_t)Ang;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    // read proj (assume A-V-U contiguous: [a][v][u])
    std::vector<float> h_proj(proj_elems);
    if (!read_raw_float("pmma_cylinder_150cm_proj.raw", h_proj)) {
        std::printf("Error: cannot read cat515_projection.raw (expect %zu floats)\n", proj_elems);
        return -1;
    }

    // build vector geometry (circular)
    std::vector<SConeProjectionVec> geo;
    YK::build_circular_vec_geometry(
        geo, Ang, Nu, Nv,
        SID, SDD,
        du, dv,
        /*offsetU*/-5.5f, /*offsetV*/0.0f);

    // cuda
    cudaStream_t s = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    float* d_vol = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_vol, vol_elems * sizeof(float)));

    // streaming recon (preweight + filter + vec BP)
    YK::fdk_vec_recon_streaming(
        h_proj.data(),
        d_vol,
        geo,
        Nu, Nv, Ang,
        Nx, Ny, Nz, vox,
        SID, SDD, du, dv,
        /*Kchunk=*/8,
        -5.5f,0.0f,
        s
    );
    //YK::fdk_recon_streaming(
    //    h_proj.data(),
    //    d_vol,
    //    Nu, Nv, Ang,
    //    Nx, Ny, Nz, vox,
    //    SID, SDD, du, dv,
    //    -5.0f,0.0f,
    //    /*Kchunk=*/8,
    //    s
    //);

    YK_CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<float> h_vol(vol_elems);
    YK_CUDA_CHECK(cudaMemcpy(h_vol.data(), d_vol, vol_elems * sizeof(float), cudaMemcpyDeviceToHost));

    if (!write_raw_float("fdk_vec_vol_new.raw", h_vol)) {
        std::printf("Error: cannot write fdk_vec_vol.raw\n");
        return -2;
    }

    YK_CUDA_CHECK(cudaFree(d_vol));
    YK_CUDA_CHECK(cudaStreamDestroy(s));

    std::printf("Done: wrote fdk_vec_vol.raw (%d x %d x %d)\n", Nx, Ny, Nz);
    return 0;
}


int main() {
    //YKTest::testFFT();
    main_fdk();
    //YKTest::testFilterWeightsSpectra_RamLak();
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
