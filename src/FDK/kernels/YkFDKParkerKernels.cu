#pragma once
#include <cuda_runtime.h>

#include "../../global/YkGlobals.h"              // CUDA_PI
#include "../../global/YkMacro.hpp"
#include "YkFDKParkerHelpers.cuh"
#include "../../global/YkWarpStrideCtx.cuh"


namespace YK {
    namespace Fdk {
        namespace detail {
            // ----------------------------------------------------------------
                    // Constant memory — 每 chunk 的相对扫描角（已归一化到 [0, 2π)）
                    // ----------------------------------------------------------------
            __constant__ float gC_parker_angle[kMaxChunkAng];


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
                    YK_DEV_LOGI("data[%d] = %f\n", idx, data[idx]);
                }
            }

            // ----------------------------------------------------------------
            // parker_weight_kernel
            //   原地 Parker 加权：d_data [K, Nv, Nu] device buffer。
            //
            //   fDirSign：旋转方向符号。
            //     +1.f  顺时针（角度单调递增，默认）
            //     -1.f  逆时针（角度单调递减）
            //
            //   逆时针时 gamma 取反，使扇形角符号与 beta 积分方向一致，
            //   保证 Parker 分区 [0, t1, t2, t3] 的物理含义不变。
            // ----------------------------------------------------------------
            __global__ void parker_weight_kernel(
                float* __restrict__ data,
                int Nu, int Nv, int K,
                float fSDD, float fDetUSize,
                float fCentralFanAngle, float fScale,
                int nDirSign)
            {
                WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;

                const int   total_rows = K * Nv;
                const float Gamma = fCentralFanAngle;
                const float kQuadPi = CUDA_PI * 0.25f;
                const float eps = 1e-6f;

                for (int row = ctx.warp_global; row < total_rows; row += ctx.n_warps)
                {
                    const int angle = row / Nv;
                    // beta 已在上传时折算为标准正向 [0, β_max]，不再翻转
                    const float beta = gC_parker_angle[angle];

                    for (int u = ctx.lane; u < Nu; u += 32)
                    {
                        const float u_mm = (u - 0.5f * Nu + 0.5f) * fDetUSize;
                        // gamma 按 nDirSign 翻转，与 beta 保持同坐标系
                        const float gamma = nDirSign * atanf(u_mm / fSDD);

                        const float t1 = 2.0f * (Gamma + gamma);
                        const float t2 = CUDA_PI + 2.0f * gamma;
                        const float t3 = CUDA_PI + 2.0f * Gamma;

                        float w;
                        if (beta <= 0.0f) { w = 0.0f; }
                        else if (beta < t1)
                        {
                            const float d = Gamma + gamma;
                            const float s = sinf(kQuadPi * beta / (fabsf(d) > eps ? d : eps));
                            w = s * s;
                        }
                        else if (beta <= t2) { w = 1.0f; }
                        else if (beta < t3)
                        {
                            const float d = Gamma - gamma;
                            const float s = sinf(kQuadPi * (t3 - beta) / (fabsf(d) > eps ? d : eps));
                            w = s * s;
                        }
                        else { w = 0.0f; }

                        data[(size_t)row * Nu + u] *= w * fScale;
                    }
                }
            }




            // ----------------------------------------------------------------
   // pk_uploadAngles
   //   将本 chunk 的绝对角度折算为"沿扫描方向的正向累积角 beta"，
   //   范围 [0, β_max]，单调递增，再上传到 constant memory。
   //
   //   nDirSign: +1 顺时针(角度递增), -1 逆时针(角度递减)
   //
   //   注意：cudaMemcpyToSymbol 是同步调用，无需显式 stream 参数。
   // ----------------------------------------------------------------
            void pk_uploadAngles(
                const float* h_angles,
                int          K,
                float        fAngleBase,
                int          nDirSign)
            {
                std::vector<float> rel(K);
                for (int i = 0; i < K; ++i) {
                    float f = nDirSign * (h_angles[i] - fAngleBase);
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
           //
           //   fDirSign：旋转方向符号（默认 +1.f 顺时针，兼容旧调用）。
           //   建议调用层通过角度序列自动判断：
           //     float fDirSign = (h_angles[K-1] >= h_angles[0]) ? +1.f : -1.f;
           // ----------------------------------------------------------------
            void pk_launchParker(
                float* d_data,
                int Nu, int Nv, int K,
                float fSDD, float fDetUSize,
                float fCentralFanAngle, float fScale,
                int nDirSign,
                cudaStream_t stream)
            {
                SKernelLaunchPolicy policy;
                policy.block_threads = 256;
                const auto launch = policy.makeRowWarp((size_t)K * Nv);

                parker_weight_kernel << <launch.grid, launch.block, 0, stream >> > (
                    d_data, Nu, Nv, K,
                    fSDD, fDetUSize, fCentralFanAngle, fScale,
                    nDirSign);

                YK_CUDA_KERNEL_CHECK();
            }



        }
    }
} // namespace YK::Fdk::detail


//namespace YK {
//    namespace Fdk {
//        namespace detail {
//
//            // ----------------------------------------------------------------
//            // print_d_out_kernel  （调试用，Release 中不调用）
//            // ----------------------------------------------------------------
//            __global__ void print_d_out_kernel(const float* data, int Nu, int Nv, int K)
//            {
//                const int u = blockIdx.x * blockDim.x + threadIdx.x;
//                const int angle = blockIdx.y * blockDim.y + threadIdx.y;
//                if (u >= Nu || angle >= K) return;
//                for (int v = 0; v < Nv; ++v) {
//                    const int idx = (angle * Nv + v) * Nu + u;
//                    printf("data[%d] = %f\n", idx, data[idx]);
//                }
//            }
//
//            // ----------------------------------------------------------------
//            // parker_weight_kernel
//            //
//            //   数据布局：[K, Nv, Nu]，紧密排列，无 padding。
//            //   beta  = gC_parker_angle[angle]，已归一化到 [0, 2π)。
//            //   gamma = atan(u_mm / SDD)，u_mm 为探测器列物理坐标。
//            //   沿 v 方向同列权重相同，一次 thread 处理整列。
//            // ----------------------------------------------------------------
//            __global__ void parker_weight_kernel(
//                float* __restrict__ data,
//                int   Nu,
//                int   Nv,
//                int   K,
//                float fSDD,
//                float fDetUSize,
//                float fCentralFanAngle,
//                float fScale)
//            {
//                const int u = blockIdx.x * blockDim.x + threadIdx.x;
//                const int angle = blockIdx.y * blockDim.y + threadIdx.y;
//
//                if (u >= Nu || angle >= K) return;
//
//                // 探测器列物理坐标(以探测器中心为原点)
//                const float u_mm = (u - 0.5f * Nu + 0.5f) * fDetUSize;
//                const float gamma = atanf(u_mm / fSDD);
//                const float beta = gC_parker_angle[angle];
//
//                // Parker 权重分段函数
//                const float t1 = 2.0f * (fCentralFanAngle + gamma);
//                const float t2 = CUDA_PI + 2.0f * gamma;
//                const float t3 = CUDA_PI + 2.0f * fCentralFanAngle;
//
//                float w;
//                if (beta <= 0.0f) {
//                    w = 0.0f;
//                }
//                else if (beta < t1) {
//                    const float arg = (CUDA_PI * 0.25f) * beta / (fCentralFanAngle + gamma);
//                    const float s = sinf(arg);
//                    w = s * s;
//                }
//                else if (beta <= t2) {
//                    w = 1.0f;
//                }
//                else if (beta < t3) {
//                    const float arg = (CUDA_PI * 0.25f) * (CUDA_PI + 2.0f * fCentralFanAngle - beta)
//                        / (fCentralFanAngle - gamma);
//                    const float s = sinf(arg);
//                    w = s * s;
//                }
//                else {
//                    w = 0.0f;
//                }
//
//                w *= fScale;
//
//                // 沿 v 方向写回
//                for (int v = 0; v < Nv; ++v) {
//                    const int idx = (angle * Nv + v) * Nu + u;
//                    data[idx] *= w;
//                }
//            }
//
//
//            // ----------------------------------------------------------------
//         // pk_uploadAngles
//         //   将本 chunk 的绝对角度转换为相对角度（归一化到 [0, 2π)）
//         //   并上传到 constant memory gC_parker_angle。
//         //
//         //   注意：cudaMemcpyToSymbol 是同步调用，无需显式 stream 参数。
//         // ----------------------------------------------------------------
//            void pk_uploadAngles(
//                const float* h_angles,
//                int          K,
//                float        fAngleBase)
//            {
//                std::vector<float> rel(K);
//                for (int i = 0; i < K; ++i) {
//                    float f = h_angles[i] - fAngleBase;
//                    while (f < 0.f)            f += 2.f * CUDA_PI;
//                    while (f >= 2.f * CUDA_PI)  f -= 2.f * CUDA_PI;
//                    rel[i] = f;
//                }
//
//                printf("\n[pk_upload] K=%d base=%.6f angles[0]=%.6f angles[K-1]=%.6f\n\n",
//                    K, fAngleBase, h_angles[0], h_angles[K - 1]);
//                YK_CUDA_CHECK(cudaMemcpyToSymbol(
//
//                    gC_parker_angle,
//                    rel.data(),
//                    static_cast<size_t>(K) * sizeof(float),
//                    0,
//                    cudaMemcpyHostToDevice));
//            }
//
//            // ----------------------------------------------------------------
//            // pk_launchParker
//            //   原地 Parker 加权：d_data [K, Nv, Nu] device buffer。
//            // ----------------------------------------------------------------
//            void pk_launchParker(
//                float* d_data,
//                int    Nu, int Nv, int K,
//                float  fSDD,
//                float  fDetUSize,
//                float  fCentralFanAngle,
//                float  fScale,
//                cudaStream_t stream)
//            {
//                const dim3 dimBlock(32, 8);
//                const dim3 dimGrid(
//                    (Nu + 31) / 32,
//                    (K + 7) / 8);
//
//                parker_weight_kernel << <dimGrid, dimBlock, 0, stream >> > (
//                    d_data, Nu, Nv, K,
//                    fSDD, fDetUSize, fCentralFanAngle, fScale);
//                YK_CUDA_KERNEL_CHECK();
//            }
//
//        }
//    }
//} // namespace YK::Fdk::detail
