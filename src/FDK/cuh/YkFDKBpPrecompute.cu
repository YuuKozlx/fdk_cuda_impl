#include "YkFDKBpPrecompute.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
           // fdk_precompute_coeffs_kernel
           //
           //   每个 thread 处理一个视角，将几何参数预计算为仿射系数
           //   FdkAffineCoeff{Cu, Cv, Cd, dtheta, SID2, fScaleDTheta}。
           //   所有中间量用 double 精度计算，结果截断为 float 写出。
           // ----------------------------------------------------------------
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
                //const double nx = gv.det_n.x, ny = gv.det_n.y, nz = gv.det_n.z;
                const double rcx = g.srcCR.x, rcy = g.srcCR.y, rcz = g.srcCR.z;
                const double dsx = g.detS.x, dsy = g.detS.y, dsz = g.detS.z;
                const double SDD = gv.SDD_plane_mm;

                const double src_dot_n = sx * rcx + sy * rcy + sz * rcz;
                const double src_dot_u = sx * ux + sy * uy + sz * uz;
                const double src_dot_v = sx * vx + sy * vy + sz * vz;

                const double offset_u = (dsx - sx) * ux + (dsy - sy) * uy + (dsz - sz) * uz;
                const double offset_v = (dsx - sx) * vx + (dsy - sy) * vy + (dsz - sz) * vz;

                // 分母系数
                const double Cd_x = rcx;
                const double Cd_y = rcy;
                const double Cd_z = rcz;
                const double Cd_w = -src_dot_n;

                // u 分子系数
                const double Cu_x = SDD * ux - offset_u * rcx;
                const double Cu_y = SDD * uy - offset_u * rcy;
                const double Cu_z = SDD * uz - offset_u * rcz;
                const double Cu_w = -SDD * src_dot_u + offset_u * src_dot_n;

                // v 分子系数
                const double Cv_x = SDD * vx - offset_v * rcx;
                const double Cv_y = SDD * vy - offset_v * rcy;
                const double Cv_z = SDD * vz - offset_v * rcz;
                const double Cv_w = -SDD * src_dot_v + offset_v * src_dot_n;

                // 非正交基变换
                const double UU = gv.UU;
                const double VV = gv.VV;
                const double UV = gv.UV;
                const double inv = gv.invDetUV;



                FdkAffineCoeff c;
                c.Cu_x = (float)((Cu_x * VV - Cv_x * UV) * inv);
                c.Cu_y = (float)((Cu_y * VV - Cv_y * UV) * inv);
                c.Cu_z = (float)((Cu_z * VV - Cv_z * UV) * inv);
                c.Cu_w = (float)((Cu_w * VV - Cv_w * UV) * inv);
                c.Cv_x = (float)((-Cu_x * UV + Cv_x * UU) * inv);
                c.Cv_y = (float)((-Cu_y * UV + Cv_y * UU) * inv);
                c.Cv_z = (float)((-Cu_z * UV + Cv_z * UU) * inv);
                c.Cv_w = (float)((-Cu_w * UV + Cv_w * UU) * inv);
                c.Cd_x = (float)Cd_x;
                c.Cd_y = (float)Cd_y;
                c.Cd_z = (float)Cd_z;
                c.Cd_w = (float)Cd_w;

                c.dtheta = gv.dtheta;
                c.SID2 = (float)(gv.SOD_mm * gv.SOD_mm);
                c.fScaleDTheta = gv.fScaleDTheta;
                c.SDD2 = (float)(SDD * SDD);
                c.du_mm = gv.du_mm;
                c.dv_mm = gv.dv_mm;

                d_coeffs[a] = c;
#ifdef YK_DEBUG
                if (a == 0) {
                    printf("[geo0] src=(%f,%f,%f)\n", sx, sy, sz);
                    printf("[geo0] detU=(%f,%f,%f)\n", ux, uy, uz);
                    printf("[geo0] detV=(%f,%f,%f)\n", vx, vy, vz);
                    printf("[geo0] detS=(%f,%f,%f)\n", dsx, dsy, dsz);
                    printf("[geo0] srcCR=(%f,%f,%f)\n", rcx, rcy, rcz);
                    printf("[geo0] SDD=%.3f UU=%.6f VV=%.6f UV=%.6f inv=%.10f\n",
                        SDD, UU, VV, UV, inv);
                    printf("[geo0] offset_u=%.6f offset_v=%.6f\n", offset_u, offset_v);
                }
#endif // DEBUG


            }


        }

        // ----------------------------------------------------------------
        // bp_launchPrecomputeCoeffs
        // ----------------------------------------------------------------
        void bp_launchPrecomputeCoeffs(
            const SConeProjGeomVec* d_geo,
            const SFDKGeoParamPerView* d_gv,
            FdkAffineCoeff* d_coeffs,
            int Ang, cudaStream_t stream)
        {
            constexpr int threads = 128;
            const int blocks = (Ang + threads - 1) / threads;
            detail::fdk_precompute_coeffs_kernel << <blocks, threads, 0, stream >> > (
                d_geo, d_gv, d_coeffs, Ang);
            YK_CUDA_KERNEL_CHECK();
        }
    }
} // namespace YK::Fdk::detail
