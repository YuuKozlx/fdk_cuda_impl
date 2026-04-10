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

                u_pix = (DU * gv.VV - DV * gv.UV) * gv.invDetUV * gv.inv_du_mm;
                v_pix = (-DU * gv.UV + DV * gv.UU) * gv.invDetUV * gv.inv_dv_mm;
                return true;
            }

        }
    }
} // namespace YK::Fdk::detail
