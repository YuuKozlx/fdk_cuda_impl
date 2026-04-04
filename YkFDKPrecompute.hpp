#pragma once
#include <cuda_runtime.h>
#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkFDKVecGeoDerived.hpp"

namespace YK {

    __global__ void fdk_precompute_coeffs_kernel(
        const SConeProjectionVec* __restrict__ d_geo,
        const SFDKGeoParamPerView* __restrict__ d_gv,
        FdkAffineCoeff* __restrict__ d_coeffs,
        int Ang)
    {
        int a = blockIdx.x * blockDim.x + threadIdx.x;
        if (a >= Ang) return;

        const SConeProjectionVec& g = d_geo[a];
        const SFDKGeoParamPerView& gv = d_gv[a];

        const float3 src = g.src;
        const float3 detU = g.detU;
        const float3 detV = g.detV;
        const float3 det_n = gv.det_n;
        const float3 detS = g.detS;
        const float  SDD = gv.SDD_mm;

        float3 detS_src = make_float3(
            detS.x - src.x,
            detS.y - src.y,
            detS.z - src.z);

        float offset_u = detS_src.x * detU.x + detS_src.y * detU.y + detS_src.z * detU.z;
        float offset_v = detS_src.x * detV.x + detS_src.y * detV.y + detS_src.z * detV.z;

        float src_dot_n = src.x * det_n.x + src.y * det_n.y + src.z * det_n.z;
        float src_dot_u = src.x * detU.x + src.y * detU.y + src.z * detU.z;
        float src_dot_v = src.x * detV.x + src.y * detV.y + src.z * detV.z;

        // 分母系数：(P-src)·det_n
        float Cd_x = det_n.x;
        float Cd_y = det_n.y;
        float Cd_z = det_n.z;
        float Cd_w = -src_dot_n;

        // u 分子系数：SDD*(P-src)·detU - offset_u*(P-src)·det_n
        float Cu_x = SDD * detU.x - offset_u * det_n.x;
        float Cu_y = SDD * detU.y - offset_u * det_n.y;
        float Cu_z = SDD * detU.z - offset_u * det_n.z;
        float Cu_w = -SDD * src_dot_u + offset_u * src_dot_n;

        // v 分子系数
        float Cv_x = SDD * detV.x - offset_v * det_n.x;
        float Cv_y = SDD * detV.y - offset_v * det_n.y;
        float Cv_z = SDD * detV.z - offset_v * det_n.z;
        float Cv_w = -SDD * src_dot_v + offset_v * src_dot_n;

        // 非正交基变换
        float VV = gv.VV, UV = gv.UV, UU = gv.UU, inv = gv.invDetUV;

        FdkAffineCoeff c;
        c.Cu = make_float4(
            (Cu_x * VV - Cv_x * UV) * inv,
            (Cu_y * VV - Cv_y * UV) * inv,
            (Cu_z * VV - Cv_z * UV) * inv,
            (Cu_w * VV - Cv_w * UV) * inv);
        c.Cv = make_float4(
            (-Cu_x * UV + Cv_x * UU) * inv,
            (-Cu_y * UV + Cv_y * UU) * inv,
            (-Cu_z * UV + Cv_z * UU) * inv,
            (-Cu_w * UV + Cv_w * UU) * inv);
        c.Cd = make_float4(Cd_x, Cd_y, Cd_z, Cd_w);
        c.dtheta = gv.dtheta;
        c.SID2 = gv.SOD_mm * gv.SOD_mm;

        d_coeffs[a] = c;
    }

    inline void launchPrecomputeCoeffs(
        const SConeProjectionVec* d_geo,
        const SFDKGeoParamPerView* d_gv,
        FdkAffineCoeff* d_coeffs,
        int Ang, cudaStream_t stream)
    {
        const int threads = 128;
        const int blocks = (Ang + threads - 1) / threads;
        fdk_precompute_coeffs_kernel << <blocks, threads, 0, stream >> > (
            d_geo, d_gv, d_coeffs, Ang);
    }

} // namespace YK