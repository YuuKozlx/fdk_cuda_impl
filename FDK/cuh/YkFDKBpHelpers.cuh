#pragma once
#include <cuda_runtime.h>

#include "../YkVecGeo.hpp"             // SConeProjGeomVec, SFDKGeoParamPerView
#include "../YkFDKVecGeoDerived.hpp"   // FdkAffineCoeff（类型定义）
#include "../../util/YkVecOperation.hpp"       // f3_sub, f3_dot

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // project_uv_and_terms_derived
            //
            //   将体素 P 投影到探测器，输出：
            //     u_pix / v_pix — 像素坐标（未偏移 0.5）
            //     denom_c       — dir · ray_center，供 FDK 余弦权重计算
            //   返回 false 表示投影无效（法向分母过小或像素在源后方）。
            // ----------------------------------------------------------------
            __device__ __forceinline__ bool project_uv_and_terms_derived(
                const SConeProjGeomVec& g,
                const SFDKGeoParamPerView& gv,
                float3                     P,
                float& u_pix,
                float& v_pix,
                float& denom_c)
            {
                const float3 dir = f3_sub(P, g.src);

                denom_c = f3_dot(dir, gv.ray_center);

                const float denom_n = f3_dot(dir, gv.det_n);
                if (fabsf(denom_n) < 1e-8f) return false;

                const float t = __fdividef(gv.SDD_mm, denom_n);
                if (t <= 0.f) return false;

                const float DU = t * f3_dot(dir, g.detU) - gv.detS_sub_src_dot_dU;
                const float DV = t * f3_dot(dir, g.detV) - gv.detS_sub_src_dot_dV;

                // 临时验证，只打第一次调用
//#ifdef YK_DEBUG
//// 用 atomicAdd 防止多线程刷屏，只打一次
//                static __device__ int printed = 0;
//                if (atomicAdd(&printed, 1) == 0) {
//                    printf("[proj] t=%.6f DU=%.6f DV=%.6f inv_du=%.6f inv_dv=%.6f\n",
//                        t, DU, DV, gv.inv_du_mm, gv.inv_dv_mm);
//                    printf("[proj] detS_sub_src_dot_dU=%.6f\n", gv.detS_sub_src_dot_dU);
//                    printf("[proj] dir=(%.3f,%.3f,%.3f)\n", dir.x, dir.y, dir.z);
//                }
//#endif

                u_pix = (DU * gv.VV - DV * gv.UV) * gv.invDetUV;
                v_pix = (-DU * gv.UV + DV * gv.UU) * gv.invDetUV;
                return true;
            }

        }


        void bp_uploadCoeffsChunk(
            const FdkAffineCoeff* d_src, int K, cudaStream_t stream);

        void bp_readbackCoeffs(           // ← 新增
            FdkAffineCoeff* h_dst, int K);
    }
} // namespace YK::Fdk::detail
