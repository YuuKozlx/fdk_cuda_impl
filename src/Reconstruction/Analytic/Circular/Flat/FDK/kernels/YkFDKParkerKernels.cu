#pragma once
#include <cuda_runtime.h>

#include "global/YkGlobals.h"              // CUDA_PI
#include "global/YkMacro.hpp"
#include "global/YkFdkKernelTypes.hpp"
#include "global/YkKernelLaunchPolicy.hpp"
#include "YkFDKParkerHelpers.cuh"
#include "global/YkWarpStrideCtx.cuh"


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
                const SConeProjGeomVec* __restrict__ geometry,
                const SFDKGeoParamPerView* __restrict__ gv,
                float fScale,
                int nDirSign)
            {
                WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;

                const int   total_rows = K * Nv;
                const float kQuadPi = CUDA_PI * 0.25f;
                const float eps = 1e-6f;

                for (int row = ctx.warp_global; row < total_rows; row += ctx.n_warps)
                {
                    const int angle = row / Nv;
                    const SConeProjGeomVec& geo = geometry[angle];
                    const SFDKGeoParamPerView& view = gv[angle];
                    const float center_v = 0.5f * (Nv - 1);
                    const float3 radial = make_float3(
                        view.radial_ray.x, view.radial_ray.y, 0.f);
                    const auto fanAngle = [&](float detector_u) {
                        const float ray_x = geo.detS.x + detector_u * geo.detU.x +
                            center_v * geo.detV.x - geo.src.x;
                        const float ray_y = geo.detS.y + detector_u * geo.detU.y +
                            center_v * geo.detV.y - geo.src.y;
                        // atan2(ray x radial) 的符号与旧实现的探测器 U 扇角一致。
                        return atan2f(ray_x * radial.y - ray_y * radial.x,
                            ray_x * radial.x + ray_y * radial.y);
                    };
                    const float Gamma = fmaxf(
                        fabsf(fanAngle(0.f)), fabsf(fanAngle(float(Nu - 1))));
                    // beta 已在上传时折算为标准正向 [0, β_max]，不再翻转
                    const float beta = gC_parker_angle[angle];

                    for (int u = ctx.lane; u < Nu; u += 32)
                    {
                        // gamma 按 nDirSign 翻转，与 beta 保持同坐标系
                        const float gamma = nDirSign * fanAngle(float(u));

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
                const SConeProjGeomVec* d_geometry,
                const SFDKGeoParamPerView* d_gv,
                float fScale,
                int nDirSign,
                cudaStream_t stream)
            {
                SKernelLaunchPolicy policy;
                policy.block_threads = 256;
                const auto launch = policy.makeRowWarp((size_t)K * Nv);

                parker_weight_kernel << <launch.grid, launch.block, 0, stream >> > (
                    d_data, Nu, Nv, K,
                    d_geometry, d_gv, fScale,
                    nDirSign);

                YK_CUDA_KERNEL_CHECK();
            }



        }
    }
} // namespace YK::Fdk::detail
