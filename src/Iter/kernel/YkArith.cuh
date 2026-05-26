// YkArith.cuh
#pragma once
// 禁止被其他模块include，除非是 Iter 模块的 kernel 实现文件 YkIterKernels.cu
// 外部调用请使用 Iter 模块提供的 launch 接口（如 YkIterLaunch.cuh 中声明的那些函数）
#include <cstddef>
#include <cuda_runtime.h>

namespace YK {

    // ── 核心 kernel：接受任意 device callable ────────────────────────────────────
    template<class F>
    __global__ void elemwise_kernel(float* __restrict__ out, size_t n, F f)
    {
        const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) f(out[i], i);
    }

    // ── launch 入口 ───────────────────────────────────────────────────────────────
    template<class F>
    void elemwise(float* out, size_t n, cudaStream_t stream, F f, int block = 256)
    {
        if (n == 0) return;
        const int grid = (int)((n + block - 1) / block);
        elemwise_kernel << <grid, block, 0, stream >> > (out, n, f);
    }

} // namespace YK