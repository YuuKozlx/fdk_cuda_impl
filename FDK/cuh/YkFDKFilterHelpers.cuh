#pragma once
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "../YkFdkPipelineContext.hpp"   // SKernelLaunchPolicy


namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // Launch-policy normalization
            // ----------------------------------------------------------------
            YK_INLINE SKernelLaunchPolicy normalizeFilterPolicy(SKernelLaunchPolicy p)
            {
                if (p.block_threads < 32) p.block_threads = 32;
                p.block_threads = (p.block_threads + 31) & ~31;
                p.block_threads = std::min(p.block_threads, 1024);
                return p;
            }

            // ----------------------------------------------------------------
            // Padding / offset helpers  （__host__ __device__ 双用）
            // ----------------------------------------------------------------

            /// 返回满足 paddedN >= 2*Nu 的最小 2 的幂
            __host__ __device__ __forceinline__
                int fp_computePaddedN(int Nu)
            {
                int n = 1;
                while (n < 2 * Nu) n <<= 1;
                return n;
            }

            /// 计算源行在 padded 行中的起始偏移，使探测器轴居中
            __host__ __device__ __forceinline__
                int fp_computeStartU(int Nu, int paddedN, float offsetU_pix)
            {
                const float axis = (Nu - 1) * 0.5f + offsetU_pix;
                return static_cast<int>(lrintf(paddedN * 0.5f - axis));
            }

        }; // namespace YK::Fdk::detail
    };
};
