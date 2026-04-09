#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace YK_Test {
    // 生成空间域 RL 核
    std::vector<float> generateSpatialRLKernel(int size, float du) {
        std::vector<float> kernel(size);
        int center = size / 2;
        for (int i = 0; i < size; ++i) {
            int n = i - center;
            if (n == 0) {
                kernel[i] = 1.0f / (4.0f * du * du);
            }
            else if (n % 2 == 0) {
                kernel[i] = 0;
            }
            else {
                kernel[i] = -1.0f / (float)(M_PI * M_PI * n * n * du * du);
            }
        }
        return kernel;
    }

    // CPU 空间域卷积 (简单实现，用于验证)
    std::vector<float> cpuConvolve(const std::vector<float>& input, const std::vector<float>& kernel) {
        int n = input.size();
        int k = kernel.size();
        std::vector<float> output(n, 0.0f);
        int k_center = k / 2;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                int input_idx = i + (j - k_center);
                if (input_idx >= 0 && input_idx < n) {
                    output[i] += input[input_idx] * kernel[j];
                }
            }
        }
        return output;
    }
}