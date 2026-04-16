#include <cuda_runtime_api.h>
#include "../../global/YkGlobals.h"
#include "../../global/YkMacro.hpp"
#include "YkFDKBpHelpers.cuh"
#include "YkFDKBpLaunch.cuh"

__constant__ YK::FdkAffineCoeff gC_coeffs[YK::kMaxChunkAng];

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
            // 【第四步：FDK 深度权重分母 denom_c】反投影加权因子
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
                // 交点
                const point3 Pi = f3_add(g.src, f3_scale(f3_sub(P, g.src), t));
                const float3 D = f3_sub(Pi, g.detS);

                const float  DU = f3_dot(D, gv.det_u);
                const float  DV = f3_dot(D, gv.det_v);


                u_pix = DU * gv.inv_du_mm;
                v_pix = DV * gv.inv_dv_mm;
                // 临时验证，只打第一次调用
#ifdef YK_DEBUG
                // 用 atomicAdd 防止多线程刷屏，只打一次
                static __device__ int printed = 0;
                if (atomicAdd(&printed, 1) == 0) {
                    YK_DEV_LOGD("[proj] t=%.6f DU=%.6f DV=%.6f inv_du=%.6f inv_dv=%.6f\n",
                        t, DU, DV, gv.inv_du_mm, gv.inv_dv_mm);
                    YK_DEV_LOGD("[proj] detS_sub_src_dot_dU=%.6f\n", gv.detS_sub_src_dot_dU);
                    YK_DEV_LOGD("[proj] dir=(%.3f,%.3f,%.3f)\n", dir.x, dir.y, dir.z);
                }
#endif

                return true;
            }
        }

        namespace detail {

            // ----------------------------------------------------------------
            // fdk_bp_kernel<ZSIZE>  — 预计算版本（读 gC_coeffs constant memory）
            // ----------------------------------------------------------------
            template<int ZSIZE>
            __global__ void fdk_bp_kernel(
                const cudaTextureObject_t* __restrict__ tex_views,
                float* __restrict__ vol,
                SVolGeom vg,
                int K)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vg.Nx || y >= vg.Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= vg.Nz) return;

                const float fX = vg.origin().x + x * vg.vox_x;
                const float fY = vg.origin().y + y * vg.vox_y;
                const float fZ = vg.origin().z + startZ * vg.vox_z;

#ifdef YK_DEBUG
                const bool is_debug_thread = (x == 0 && y == 0 && startZ == 0);
                // ── 层1：体素世界坐标和 K ────────────────────────────────────
                if (is_debug_thread)
                    YK_DEV_LOGD("[bp_pre][voxel0] world=(%.3f,%.3f,%.3f) K=%d\n",
                        fX, fY, fZ, K);
#endif

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int i = 0; i < K; ++i) {
                    const FdkAffineCoeff& c = gC_coeffs[i];

                    float uNum = c.Cu_w + fX * c.Cu_x + fY * c.Cu_y + fZ * c.Cu_z;
                    float vNum = c.Cv_w + fX * c.Cv_x + fY * c.Cv_y + fZ * c.Cv_z;
                    float den = c.Cd_w + fX * c.Cd_x + fY * c.Cd_y + fZ * c.Cd_z;

                    const float uStep = c.Cu_z * vg.vox_z;
                    const float vStep = c.Cv_z * vg.vox_z;
                    const float dStep = c.Cd_z * vg.vox_z;
                    const float w_base = c.SID2 * c.dtheta * c.fScaleDTheta;

#ifdef YK_DEBUG
                    // ── 层2：几何参数和权重基础值 ────────────────────────────
                    if (is_debug_thread && i < 2)
                        YK_DEV_LOGD("[bp_pre][view%d] den=%.6f u0=%.3f v0=%.3f "
                            "w_base=%.8f SID2=%.3f dtheta=%.6f\n",
                            i, den, uNum / den, vNum / den,
                            w_base, c.SID2, c.dtheta);
#endif

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz) {
                        const float fr = __fdividef(1.f, den);
                        const float u = uNum * fr;
                        const float v = vNum * fr;
                        const float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                        const float contrib = p * (w_base * fr * fr);

#ifdef YK_DEBUG
                        // ── 层3：fetch 值和贡献量（iz=0）────────────────────
                        if (is_debug_thread && i < 2 && iz == 0) {
                            const float worldZ = fZ + iz * vg.vox_z;
                            YK_DEV_LOGD("[bp_pre][view%d][iz0] world=(%.3f,%.3f,%.3f) "
                                "u=%.3f v=%.3f p=%.6f den=%.6f contrib=%.8f\n",
                                i, fX, fY, worldZ, u, v, p, den, contrib);
                            if (isnan(contrib) || isinf(contrib))
                                YK_DEV_LOGD("[bp_pre][NaN!][view%d] den=%.8f w_base=%.8f p=%.6f\n",
                                    i, den, w_base, p);
                        }
#endif

                        Z[iz] += contrib;
                        uNum += uStep;
                        vNum += vStep;
                        den += dStep;
                    }
                }

                const int endZ = min(startZ + ZSIZE, vg.Nz);
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) {
                    if (startZ + iz < endZ) {
#ifdef YK_DEBUG
                        if (is_debug_thread && iz == 0)
                            YK_DEV_LOGD("[bp_pre][final] world=(%.3f,%.3f,%.3f) Z[0]=%.8f\n",
                                fX, fY, fZ, Z[0]);
#endif
                        const size_t idx = (size_t)(startZ + iz) * vg.Ny * vg.Nx
                            + (size_t)y * vg.Nx + x;
                        vol[idx] += Z[iz];
                    }
                }
            }

            // ----------------------------------------------------------------
            // fdk_bp_kernel<ZSIZE>  — 非预计算版本（重载，多 d_geo/d_gv 参数）
            // ----------------------------------------------------------------
            template<int ZSIZE>
            __global__ void fdk_bp_kernel(
                const cudaTextureObject_t* __restrict__ tex_views,
                const SConeProjGeomVec* __restrict__ d_geo,
                const SFDKGeoParamPerView* __restrict__ d_gv,
                float* __restrict__ vol,
                SVolGeom vg,
                int K)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vg.Nx || y >= vg.Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= vg.Nz) return;

                const float fX = vg.origin().x + x * vg.vox_x;
                const float fY = vg.origin().y + y * vg.vox_y;
                const float fZ = vg.origin().z + startZ * vg.vox_z;

#ifdef YK_DEBUG
                const bool is_debug_thread = (x == 0 && y == 0 && startZ == 0);
                // ── 层1：体素世界坐标和 K ────────────────────────────────────
                if (is_debug_thread)
                    YK_DEV_LOGD("[bp_dir][voxel0] world=(%.3f,%.3f,%.3f) K=%d\n",
                        fX, fY, fZ, K);
#endif

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int i = 0; i < K; ++i) {
                    const SConeProjGeomVec& g = d_geo[i];
                    const SFDKGeoParamPerView& gv = d_gv[i];

                    const float w_base = gv.SOD_mm * gv.SOD_mm * gv.dtheta * gv.fScaleDTheta;

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz) {
                        const int    zIdx = startZ + iz;
                        if (zIdx >= vg.Nz) continue;

                        const float  worldZ = vg.origin().z + zIdx * vg.vox_z;
                        const float3 P = make_float3(fX, fY, worldZ);

                        float u, v, denom_c;
                        if (!project_uv_and_terms_derived(g, gv, P, u, v, denom_c)) {
#ifdef YK_DEBUG
                            if (is_debug_thread && i < 2 && iz == 0)
                                YK_DEV_LOGD("[bp_dir][view%d][iz0] project FAILED "
                                    "world=(%.3f,%.3f,%.3f)\n", i, fX, fY, worldZ);
#endif
                            continue;
                        }

                        const float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                        const float inv_denom_c = __fdividef(1.f, denom_c);
                        const float contrib = p * w_base * inv_denom_c * inv_denom_c;
                        Z[iz] += contrib;

#ifdef YK_DEBUG
                        // ── 层2/3：几何参数、fetch 值和贡献量（iz=0）────────
                        if (is_debug_thread && i < 2 && iz == 0) {
                            YK_DEV_LOGD("[bp_dir][view%d][iz0] world=(%.3f,%.3f,%.3f) "
                                "u=%.3f v=%.3f p=%.6f den=%.6f contrib=%.8f "
                                "w_base=%.8f SOD=%.3f dtheta=%.6f\n",
                                i, fX, fY, worldZ,
                                u, v, p, denom_c, contrib,
                                w_base, gv.SOD_mm, gv.dtheta);
                            if (isnan(contrib) || isinf(contrib))
                                YK_DEV_LOGD("[bp_dir][NaN!][view%d] SOD=%.3f denom_c=%.8f p=%.6f\n",
                                    i, gv.SOD_mm, denom_c, p);
                        }
#endif
                    }
                }

                const int endZ = min(startZ + ZSIZE, vg.Nz);
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) {
                    if (startZ + iz < endZ) {
#ifdef YK_DEBUG
                        if (is_debug_thread && iz == 0)
                            YK_DEV_LOGD("[bp_dir][final] world=(%.3f,%.3f,%.3f) Z[0]=%.8f\n",
                                fX, fY, fZ, Z[0]);
#endif
                        const size_t idx = (size_t)(startZ + iz) * vg.Ny * vg.Nx
                            + (size_t)y * vg.Nx + x;
                        vol[idx] += Z[iz];
                    }
                }
            }
        };
    };
};



namespace YK {
    namespace Fdk {

        void bp_uploadCoeffsChunk(
            const FdkAffineCoeff* d_src, int K, cudaStream_t stream)
        {
            YK_CUDA_CHECK(cudaMemcpyToSymbolAsync(
                gC_coeffs,
                d_src,
                K * sizeof(FdkAffineCoeff),
                0, cudaMemcpyDeviceToDevice, stream));
        }

        void bp_readbackCoeffs(FdkAffineCoeff* h_dst, int K)
        {
            YK_CUDA_CHECK(cudaMemcpyFromSymbol(
                h_dst, gC_coeffs,
                K * sizeof(FdkAffineCoeff)));
        }



        void bp_launchBpPrecomputed(
            const cudaTextureObject_t* d_texObjs,
            float* d_vol,
            const SVolGeom& vol_geom,
            int K,
            cudaStream_t stream)
        {
            const dim3 block(16, 16, 1);

            constexpr int zsize = 4;  // ⚠️ 如果你未来要外部控制，可以改成参数

            const dim3 grid(
                (vol_geom.Nx + block.x - 1) / block.x,
                (vol_geom.Ny + block.y - 1) / block.y,
                (vol_geom.Nz + zsize - 1) / zsize);

            switch (zsize)
            {
            case 1:
                detail::fdk_bp_kernel<1> << <grid, block, 0, stream >> > (
                    d_texObjs, d_vol, vol_geom, K);
                break;

            case 2:
                detail::fdk_bp_kernel<2> << <grid, block, 0, stream >> > (
                    d_texObjs, d_vol, vol_geom, K);
                break;

            case 4:
                detail::fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                    d_texObjs, d_vol, vol_geom, K);
                break;

            case 8:
                detail::fdk_bp_kernel<8> << <grid, block, 0, stream >> > (
                    d_texObjs, d_vol, vol_geom, K);
                break;
            default:
                YK_LOGW("Unsupported zsize %d, fallback to 4\n", zsize);
                detail::fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                    d_texObjs, d_vol, vol_geom, K);

            }

            YK_CUDA_KERNEL_CHECK();
        }

        // ----------------------------------------------------------------
        // bp_launchBpDirect
        //   反投影（非预计算版本）：kernel 内实时计算投影坐标。
        // ----------------------------------------------------------------
        void bp_launchBpDirect(
            const cudaTextureObject_t* d_texObjs,
            const SConeProjGeomVec* d_geo,
            const SFDKGeoParamPerView* d_gv,
            float* d_vol,
            const SVolGeom& vol_geom,
            int K,
            cudaStream_t stream)
        {
            const dim3 block(16, 16, 1);

            constexpr int zsize = 4;

            const dim3 grid(
                (vol_geom.Nx + block.x - 1) / block.x,
                (vol_geom.Ny + block.y - 1) / block.y,
                (vol_geom.Nz + zsize - 1) / zsize);

            switch (zsize)
            {
            case 1:
                detail::fdk_bp_kernel<1> << <grid, block, 0, stream >> > (
                    d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                break;

            case 2:
                detail::fdk_bp_kernel<2> << <grid, block, 0, stream >> > (
                    d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                break;

            case 4:
                detail::fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                    d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                break;

            case 8:
                detail::fdk_bp_kernel<8> << <grid, block, 0, stream >> > (
                    d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                break;

            default:
                YK_LOGW("Unsupported zsize %d, fallback to 4\n", zsize);
                detail::fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                    d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
            }

            YK_CUDA_KERNEL_CHECK();
        }
    };
}; // namespace YK::Fdk::detail
