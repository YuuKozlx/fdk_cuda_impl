// YkArith.cuh
#pragma once
// 禁止被其他模块include，除非是 Iter 模块的 kernel 实现文件 YkIterKernels.cu
// 外部调用请使用 Iter 模块提供的 launch 接口（如 YkIterLaunch.cuh 中声明的那些函数）
#include <cstddef>
#include <cuda_runtime.h>
#include "global/YkGlobals.h"
#include "global/YkKernelLaunchPolicy.hpp"

namespace YK {

    // ── 核心 kernel：接受任意 device callable ────────────────────────────────────
    template<class F>
    __global__ void elemwise_kernel(float* __restrict__ out, size_t n, F f)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = (size_t)gridDim.x * blockDim.x;
        for (; i < n; i += stride)
            f(out[i], i);
    }

    // ── launch 入口 ───────────────────────────────────────────────────────────────
    template<class F>
    void elemwise(float* out, size_t n, cudaStream_t stream, F f, int block = 256)
    {
        if (n == 0) return;

        // Keep the launch size bounded, then let elemwise_kernel's grid-stride
        // loop cover the remainder.  This avoids both oversized grids and the
        // size_t-to-int truncation that the old one-thread-per-element launch
        // incurred for very large volumes/sinograms.
        SKernelLaunchPolicy policy;
        policy.block_threads = block;
        const auto launch = policy.make1D(n);
        elemwise_kernel << <launch.grid, launch.block, 0, stream >> > (out, n, f);
    }

} // namespace YK
