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
            // 【几何问题】
            //   给定体素世界坐标 P，求其在探测器上的像素坐标 (u_pix, v_pix)，
            //   以及 FDK 余弦权重所需的分母项 denom_c。
            //
            // 【第一步：透视投影求交参数 t】
            //
            //   射线方程：Q(t) = s + t * d，其中 d = P - s
            //
            //   探测器平面共面条件（detS 为探测器参考点，n 为法向量）：
            //     (Q - s) · n = (detS - s) · n
            //
            //   求解 t：
            //     t = [(detS - s) · n] / (d · n) = SDD_plane_mm / (d · n)
            //
            //   注意：SDD_plane_mm = (detS - s) · n，是源点到探测器平面沿法向量的投影距离。
            //         不是主射线实际长度 SDD_mm，探测器有倾斜时二者不同。
            //
            // 【第二步：求交点相对探测器参考点的位移】
            //
            //   ΔU = t*(d · e_u) - (detS - s) · e_u
            //   ΔV = t*(d · e_v) - (detS - s) · e_v
            //
            // 【第三步：非正交基下求像素坐标（Cramer 法则）】
            //
            //   | UU  UV | | u |   | ΔU |
            //   | UV  VV | | v | = | ΔV |
            //
            //   u = (ΔU*VV - ΔV*UV) / det
            //   v = (ΔV*UU - ΔU*UV) / det
            //
            //   e_u 的尺度 du 已隐含在 Gram 矩阵求逆中，u/v 直接是像素坐标。
            //   像素坐标以探测器参考点 detS（像素(0,0)）为原点。
            //
            // 【第四步：FDK 余弦权重分母 denom_c】
            //
            //   denom_c = d · r，r 为主射线方向（由转轴决定）
            //   w = (SOD / denom_c)²
            //   r 在 Z轴=转轴 的坐标系下无 Z 分量，Z 方向锥角不参与权重计算。
            //
            // 【输出】
            //   u_pix, v_pix — 像素坐标（以 detS 为原点，未偏移 0.5）
            //   denom_c      — d · r，供 FDK 余弦权重计算
            //
            // 【返回值】
            //   false — 射线近乎平行于探测器平面（denom_n 过小），或交点在源点后方（t≤0）
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

                const float t = __fdividef(gv.SDD_plane_mm, denom_n);
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
