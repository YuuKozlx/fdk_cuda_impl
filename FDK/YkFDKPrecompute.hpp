#pragma once
#include <cuda_runtime.h>
#include "global/YkGlobals.h"
#include "FDK/YkVecGeo.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"

namespace YK {

    __global__ void fdk_precompute_coeffs_kernel(
        const SConeProjGeomVec* __restrict__ d_geo,
        const SFDKGeoParamPerView* __restrict__ d_gv,
        FdkAffineCoeff* __restrict__ d_coeffs,
        int Ang)
    {
        int a = blockIdx.x * blockDim.x + threadIdx.x;
        if (a >= Ang) return;

        const SConeProjGeomVec& g = d_geo[a];
        const SFDKGeoParamPerView& gv = d_gv[a];

        // 全部用 double 中间计算
        const double sx = g.src.x, sy = g.src.y, sz = g.src.z;
        const double ux = g.detU.x, uy = g.detU.y, uz = g.detU.z;
        const double vx = g.detV.x, vy = g.detV.y, vz = g.detV.z;
        const double nx = gv.det_n.x, ny = gv.det_n.y, nz = gv.det_n.z;
        const double dsx = g.detS.x, dsy = g.detS.y, dsz = g.detS.z;
        const double SDD = gv.SDD_mm;

        const double src_dot_n = sx * nx + sy * ny + sz * nz;
        const double src_dot_u = sx * ux + sy * uy + sz * uz;
        const double src_dot_v = sx * vx + sy * vy + sz * vz;

        const double offset_u = (dsx - sx) * ux + (dsy - sy) * uy + (dsz - sz) * uz;
        const double offset_v = (dsx - sx) * vx + (dsy - sy) * vy + (dsz - sz) * vz;

        // 分母系数
        const double Cd_x = nx;
        const double Cd_y = ny;
        const double Cd_z = nz;
        const double Cd_w = -src_dot_n;

        // u 分子系数
        const double Cu_x = SDD * ux - offset_u * nx;
        const double Cu_y = SDD * uy - offset_u * ny;
        const double Cu_z = SDD * uz - offset_u * nz;
        const double Cu_w = -SDD * src_dot_u + offset_u * src_dot_n;

        // v 分子系数
        const double Cv_x = SDD * vx - offset_v * nx;
        const double Cv_y = SDD * vy - offset_v * ny;
        const double Cv_z = SDD * vz - offset_v * nz;
        const double Cv_w = -SDD * src_dot_v + offset_v * src_dot_n;

        // 非正交基变换
        const double UU = gv.UU;
        const double VV = gv.VV;
        const double UV = gv.UV;
        const double inv = gv.invDetUV;

        FdkAffineCoeff c;
        c.Cu = make_float4(
            (float)((Cu_x * VV - Cv_x * UV) * inv),
            (float)((Cu_y * VV - Cv_y * UV) * inv),
            (float)((Cu_z * VV - Cv_z * UV) * inv),
            (float)((Cu_w * VV - Cv_w * UV) * inv));
        c.Cv = make_float4(
            (float)((-Cu_x * UV + Cv_x * UU) * inv),
            (float)((-Cu_y * UV + Cv_y * UU) * inv),
            (float)((-Cu_z * UV + Cv_z * UU) * inv),
            (float)((-Cu_w * UV + Cv_w * UU) * inv));
        c.Cd = make_float4(
            (float)Cd_x, (float)Cd_y,
            (float)Cd_z, (float)Cd_w);
        c.dtheta = gv.dtheta;
        c.SID2 = (float)(gv.SOD_mm * gv.SOD_mm);
        c.fScaleDTheta = gv.fScaleDTheta;

        d_coeffs[a] = c;
    }

    inline void launchPrecomputeCoeffs(
        const SConeProjGeomVec* d_geo,
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