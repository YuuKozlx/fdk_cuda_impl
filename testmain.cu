#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <vector>
#include "YKtestconv.hpp"
#include "YkConv.hpp" // 确保包含你的命名空间定义
#include "YkFFT.hpp"
#include "YKFDKFilter.hpp"





int main() {
    // --- 参数设置 ---
    const int Nu = 64;
    const int Nv = 1;
    const int Ntheta = 1;        // Batch 设为 1
    const float du = 1.0f;
    const float offsetX = 0.0f;

    // --- 1. 初始化 FilterManager (核心复用对象) ---
    YK::FilterManager filter_mgr;
    if (!filter_mgr.init(Nu, du, Ntheta)) {
        std::cerr << "FilterManager init failed!" << std::endl;
        return -1;
    }

    // 获取由管理器自动计算的补零长度 (应该是 128)
    int paddedN = filter_mgr.getPaddedN();

    // --- 2. 准备数据 ---
    std::vector<float> h_input(Nu, 0.0f);
    h_input[Nu / 2] = 1.0f; // 中心脉冲

    float* d_input, * d_output_padded;
    cudaMalloc(&d_input, Nu * sizeof(float));
    cudaMalloc(&d_output_padded, paddedN * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), Nu * sizeof(float), cudaMemcpyHostToDevice);

    // --- 3. 执行物理对齐与滤波 ---
    // A. 物理对齐补零 (这一步每次投影都需要根据 offsetX 计算)
    dim3 block(256, 1);
    dim3 grid_pad((paddedN + 255) / 256, Ntheta);
    YK::_kernel_pad_with_offset << <grid_pad, block >> > (d_input, d_output_padded, Nu, Ntheta, paddedN, offsetX);

    // B. 复用 Plan 和权重执行滤波 (Zero-Allocation 过程)
    filter_mgr.apply(d_output_padded);

    // --- 4. 执行 CPU 空间域验证 (保持不变) ---
    std::vector<float> spatial_kernel = YK_Test::generateSpatialRLKernel(Nu, du);
    std::vector<float> cpu_result = YK_Test::cpuConvolve(h_input, spatial_kernel);

    // --- 5. 结果对比 ---
    std::vector<float> gpu_result_full(paddedN);
    cudaMemcpy(gpu_result_full.data(), d_output_padded, paddedN * sizeof(float), cudaMemcpyDeviceToHost);

    int gpu_center = paddedN / 2;
    int cpu_center = Nu / 2;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Comparison using FilterManager (Center +/- 7 elements):" << std::endl;
    std::cout << "Index | CPU (Spatial) | GPU (FFT-Reuse) | Difference" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int i = -7; i <= 7; ++i) {
        float val_cpu = cpu_result[cpu_center + i];
        float val_gpu = gpu_result_full[gpu_center + i];
        std::cout << std::setw(5) << i << " | "
            << std::setw(13) << val_cpu << " | "
            << std::setw(14) << val_gpu << " | "
            << std::setw(10) << std::abs(val_cpu - val_gpu) << std::endl;
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_output_padded);
    // filter_mgr 会在析构时自动释放内部资源
    return 0;
}