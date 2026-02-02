#pragma once
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>


namespace YK {
    // 辅助函数：打印显存中的一段连续数据（用于验证波形）
    void debugPrint(const char* label, float* d_data, int start, int count) {
        std::vector<float> h_data(count);
        cudaMemcpy(h_data.data(), d_data + start, count * sizeof(float), cudaMemcpyDeviceToHost);

        printf("--- %s ---\n", label);
        for (int i = 0; i < count; ++i) {
            printf("[%d]: %.6f\n", start + i, h_data[i]);
        }
        printf("------------------\n");
    }
}