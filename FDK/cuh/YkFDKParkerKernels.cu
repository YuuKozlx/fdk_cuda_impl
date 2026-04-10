#pragma once
#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"              // CUDA_PI
#include "../../global/YkMacro.hpp"
#include "YkFDKParkerHelpers.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // print_d_out_kernel  （调试用，Release 中不调用）
            // ----------------------------------------------------------------
            __global__ void print_d_out_kernel(const float* data, int Nu, int Nv, int K)
            {
                const int u = blockIdx.x * blockDim.x + threadIdx.x;
                const int angle = blockIdx.y * blockDim.y + threadIdx.y;
                if (u >= Nu || angle >= K) return;
                for (int v = 0; v < Nv; ++v) {
                    const int idx = (angle * Nv + v) * Nu + u;
                    printf("data[%d] = %f\n", idx, data[idx]);
                }
            }

            // ----------------------------------------------------------------
            // parker_weight_kernel
            //
            //   数据布局：[K, Nv, Nu]，紧密排列，无 padding。
            //   beta  = gC_parker_angle[angle]，已归一化到 [0, 2π)。
            //   gamma = atan(u_mm / SDD)，u_mm 为探测器列物理坐标。
            //   沿 v 方向同列权重相同，一次 thread 处理整列。
            // ----------------------------------------------------------------
            __global__ void parker_weight_kernel(
                float* __restrict__ data,
                int   Nu,
                int   Nv,
                int   K,
                float fSDD,
                float fDetUSize,
                float fCentralFanAngle,
                float fScale)
            {
                const int u = blockIdx.x * blockDim.x + threadIdx.x;
                const int angle = blockIdx.y * blockDim.y + threadIdx.y;

                if (u >= Nu || angle >= K) return;

                // 探测器列物理坐标(以探测器中心为原点)
                const float u_mm = (u - 0.5f * Nu + 0.5f) * fDetUSize;
                const float gamma = atanf(u_mm / fSDD);
                const float beta = gC_parker_angle[angle];

                // Parker 权重分段函数
                const float t1 = 2.0f * (fCentralFanAngle + gamma);
                const float t2 = CUDA_PI + 2.0f * gamma;
                const float t3 = CUDA_PI + 2.0f * fCentralFanAngle;

                float w;
                if (beta <= 0.0f) {
                    w = 0.0f;
                }
                else if (beta < t1) {
                    const float arg = (CUDA_PI * 0.25f) * beta / (fCentralFanAngle + gamma);
                    const float s = sinf(arg);
                    w = s * s;
                }
                else if (beta <= t2) {
                    w = 1.0f;
                }
                else if (beta < t3) {
                    const float arg = (CUDA_PI * 0.25f) * (CUDA_PI + 2.0f * fCentralFanAngle - beta)
                        / (fCentralFanAngle - gamma);
                    const float s = sinf(arg);
                    w = s * s;
                }
                else {
                    w = 0.0f;
                }

                w *= fScale;

                // 沿 v 方向写回
                for (int v = 0; v < Nv; ++v) {
                    const int idx = (angle * Nv + v) * Nu + u;
                    data[idx] *= w;
                }
            }


            // ----------------------------------------------------------------
         // pk_uploadAngles
         //   将本 chunk 的绝对角度转换为相对角度（归一化到 [0, 2π)）
         //   并上传到 constant memory gC_parker_angle。
         //
         //   注意：cudaMemcpyToSymbol 是同步调用，无需显式 stream 参数。
         // ----------------------------------------------------------------
            void pk_uploadAngles(
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
            void pk_launchParker(
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
