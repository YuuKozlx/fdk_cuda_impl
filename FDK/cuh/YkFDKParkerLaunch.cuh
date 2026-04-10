#pragma once
#include <vector>
#include <cstdio>

#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"              // CUDA_PI, kMaxChunkAng
#include "../../global/YkMacro.hpp"              // YK_CUDA_CHECK, YK_CUDA_KERNEL_CHECK

#include "YkFDKParkerKernels.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // pk_uploadAngles
            //   将本 chunk 的绝对角度转换为相对角度（归一化到 [0, 2π)）
            //   并上传到 constant memory gC_parker_angle。
            //
            //   注意：cudaMemcpyToSymbol 是同步调用，无需显式 stream 参数。
            // ----------------------------------------------------------------
            inline void pk_uploadAngles(
                const float* h_angles,
                int          K,
                float        fAngleBase)
            {
                std::vector<float> rel(K);
                for (int i = 0; i < K; ++i) {
                    float f = h_angles[i] - fAngleBase;
                    while (f < 0.f)            f += 2.f * CUDA_PI;
                    while (f >= 2.f * CUDA_PI)  f -= 2.f * CUDA_PI;
                    rel[i] = f;
                }
                YK_CUDA_CHECK(cudaMemcpyToSymbol(
                    gC_parker_angle,
                    rel.data(),
                    static_cast<size_t>(K) * sizeof(float),
                    0,
                    cudaMemcpyHostToDevice));
            }

            // ----------------------------------------------------------------
            // pk_launchParker
            //   原地 Parker 加权：d_data [K, Nv, Nu] device buffer。
            // ----------------------------------------------------------------
            inline void pk_launchParker(
                float* d_data,
                int    Nu, int Nv, int K,
                float  fSDD,
                float  fDetUSize,
                float  fCentralFanAngle,
                float  fScale,
                cudaStream_t stream)
            {
                const dim3 dimBlock(32, 8);
                const dim3 dimGrid(
                    (Nu + 31) / 32,
                    (K + 7) / 8);

                parker_weight_kernel << <dimGrid, dimBlock, 0, stream >> > (
                    d_data, Nu, Nv, K,
                    fSDD, fDetUSize, fCentralFanAngle, fScale);
                YK_CUDA_KERNEL_CHECK();
            }

        }
    }
} // namespace YK::Fdk::detail
