#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include "YkConv.hpp"
#include "YkFFT.hpp"

namespace YK {

    // 1. 生成空间域 RL 核：利用循环位移 (n=0 位于索引 0)
    __global__ void _kernel_gen_spatial_rl_kernel(float* kernel, int paddedN, float du) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u < paddedN) {
            // 映射到空间位移 n: [0, paddedN/2] -> [0, paddedN/2], [paddedN/2+1, paddedN-1] -> [-paddedN/2+1, -1]
            int n = (u <= paddedN / 2) ? u : u - paddedN;

            float val = 0.0f;
            if (n == 0) {
                val = 1.0f / (4.0f * du * du);
            }
            else if (abs(n) % 2 != 0) {
                // 标准离散 RL 公式
                val = -1.0f / (3.14159265358979323846f * 3.14159265358979323846f * n * n * du * du);
            }
            // IFFT 归一化：cuFFT 的 C2R 逆变换会产生 paddedN 倍增益，在此处预除以它
            kernel[u] = val / (float)paddedN;
        }
    }

    // 2. 提取实部权重 (由于核是对称的，FFT 后虚部极小，取实部可进一步稳定精度)
    __global__ void _kernel_extract_fft_weights(cufftComplex* src, float* dst, int n_complex) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u < n_complex) {
            dst[u] = src[u].x;
        }
    }

    // --- 第三步 & 第四步：物理匹配与 Padding (保持原有逻辑) ---
    __global__ void _kernel_pad_with_offset(const float* src, float* dst, int Nu, int batch, int paddedN, float offsetX) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (u >= paddedN || b >= batch) return;

        float axis_idx = (Nu - 1) * 0.5f + offsetX;
        int start_u = (int)(paddedN * 0.5f - axis_idx);

        int dst_idx = b * paddedN + u;
        if (u >= start_u && u < start_u + Nu) {
            dst[dst_idx] = src[b * Nu + (u - start_u)];
        }
        else {
            dst[dst_idx] = 0.0f;
        }
    }

    class FilterManager {
    private:
        // 1. 核心资源
        cufftHandle plan_fwd;
        cufftHandle plan_inv;
        float* d_filter_weights;    // 预计算的频域权重
        cufftComplex* d_complex_buf; // 预分配的 FFT 工作区

        // 2. 几何参数
        int paddedN;
        int n_complex;
        int current_batch;
        bool is_initialized = false;

    public:
        FilterManager() : d_filter_weights(nullptr), d_complex_buf(nullptr), is_initialized(false) {}
        ~FilterManager() { release(); }

        // --- 初始化：仅在扫描开始前执行一次 ---
        bool init(int Nu, float du, int batch) {
            release();
            current_batch = batch;

            // 计算 Padding 长度 (2 的幂)
            paddedN = 1;
            while (paddedN < 2 * Nu) paddedN <<= 1;
            n_complex = paddedN / 2 + 1;

            // 1. 预分配显存 (Zero-Allocation 核心)
            cudaMalloc(&d_filter_weights, n_complex * sizeof(float));
            cudaMalloc(&d_complex_buf, batch * n_complex * sizeof(cufftComplex));

            // 2. 预创建 FFT Plans
            // CUFFT_R2C 和 CUFFT_C2R 分别用于实转复和复转实
            if (cufftPlanMany(&plan_fwd, 1, &paddedN, NULL, 1, paddedN, NULL, 1, n_complex, CUFFT_R2C, batch) != CUFFT_SUCCESS) return false;
            if (cufftPlanMany(&plan_inv, 1, &paddedN, NULL, 1, n_complex, NULL, 1, paddedN, CUFFT_C2R, batch) != CUFFT_SUCCESS) return false;

            // 3. 高精度滤波核生成 (空间域 FFT 法)
            float* d_temp_kernel;
            cudaMalloc(&d_temp_kernel, paddedN * sizeof(float));
            _kernel_gen_spatial_rl_kernel << <(paddedN + 255) / 256, 256 >> > (d_temp_kernel, paddedN, du);

            // 执行一次核函数的 FFT 得到频域权重
            cufftHandle tmp_plan;
            cufftPlan1d(&tmp_plan, paddedN, CUFFT_R2C, 1);
            cufftComplex* d_tmp_complex;
            cudaMalloc(&d_tmp_complex, n_complex * sizeof(cufftComplex));

            cufftExecR2C(tmp_plan, (cufftReal*)d_temp_kernel, d_tmp_complex);
            _kernel_extract_fft_weights << <(n_complex + 255) / 256, 256 >> > (d_tmp_complex, d_filter_weights, n_complex);

            // 销毁临时资源
            cufftDestroy(tmp_plan);
            cudaFree(d_temp_kernel);
            cudaFree(d_tmp_complex);

            is_initialized = true;
            return true;
        }

        // --- 核心滤波函数：每帧投影执行一次 ---
        // 注意：d_padded_data 既是输入也是输出
        void apply(float* d_padded_data) {
            if (!is_initialized) return;

            // Step 1: 正向 FFT (Real to Complex)
            cufftExecR2C(plan_fwd, (cufftReal*)d_padded_data, d_complex_buf);

            // Step 2: 频域点乘权重
            dim3 block(256, 1);
            dim3 grid((n_complex + 255) / 256, current_batch);
            _kernel_pointwise_mul << <grid, block >> > (d_complex_buf, d_filter_weights, n_complex, current_batch);

            // Step 3: 逆向 FFT (Complex to Real)
            cufftExecC2R(plan_inv, d_complex_buf, (cufftReal*)d_padded_data);
        }

        void release() {
            if (is_initialized) {
                cufftDestroy(plan_fwd);
                cufftDestroy(plan_inv);
                if (d_filter_weights) cudaFree(d_filter_weights);
                if (d_complex_buf) cudaFree(d_complex_buf);
                is_initialized = false;
            }
        }

        int getPaddedN() const { return paddedN; }
    };

    // 辅助汇总函数 (简化接口)
    void executeFdkFiltering_Optimized(FilterManager& manager, float* d_input, float* d_output_padded,
        int Nu, int batch, float offsetX) {

        // 1. 带有位置匹配的 Padding (此 Kernel 仍需每次计算)
        dim3 block(256, 1);
        dim3 grid_pad((manager.getPaddedN() + 255) / 256, batch);
        _kernel_pad_with_offset << <grid_pad, block >> > (d_input, d_output_padded, Nu, batch, manager.getPaddedN(), offsetX);

        // 2. 执行预初始化的滤波
        manager.apply(d_output_padded);
    }
}