#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include <vector_functions.hpp>

#include "../FDK/YkVecGeo.hpp"
#include "../global/YkGlobals.h"
#include "../util/YkVecOperation.hpp" // f3_len, f3_cross, f3_dot, f3_sub, f3_mul, f3_add
#include "../util/helper_math.h"

namespace YK {

    // ============================================================
    // GeoDerivedManagerVec (stateless builder)
    // 目标：只写 gv（引用赋值），不返回冗余 result 结构体
    //
    // 关键约束/定义：
    //   - 世界中心固定为 (0,0,0)（isocenter 目前仅保留字段，不参与计算）
    //   - 主射线始终在 XY 平面：d(theta)=(cos, sin, 0)
    //   - principal point：主射线与探测器平面的交点
    //   - offsetU/V：principal point 对应像素坐标相对中心像素的偏移（pixel）
    //
    // SID（你固定的定义）：
    //   - 平面 Π(theta)：经过 Z 轴，法线方向 ∥ d(theta)
    //   - d(theta)=(cos,sin,0), |d|=1 时，Π(theta): d·X=0
    //   - SID = | d·src |
    //
    // 导出顺序（严格遵从你的要求）：
    //   1) theta
    //   2) dtheta
    //   3) du/dv
    //   4) (1) 主射线方向 + SID
    //      (2) principal point -> SDD + offsetU/V + ray0hat
    //   5) nhat + DSD_n + detector basis cache + invU2/invV2
    //
    // 注意：
    //   - 求交时允许“局部翻转法线以保证 t>0”，不改变你存储的法线定义逻辑。
    // ============================================================
    class GeoDerivedManagerVec
    {
    public:
        struct GeoDerivedOptions
        {
            float dtheta_eps = 1e-8f;
            float3 isocenter = make_float3(0.0f, 0.0f, 0.0f);
            bool force_DSD_positive = false; // optional
        };

        explicit GeoDerivedManagerVec(GeoDerivedOptions opt = GeoDerivedOptions{})
            : opt_(opt) {
        }

        bool build_geo_params(
            int detector_pixels_u, int detector_pixels_v, float scan_angle_rad,
            const std::vector<SConeProjGeomVec>& host_geo_per_view,
            std::vector<SFDKGeoParamPerView>& out_view_params) const
        {
            if (detector_pixels_u <= 0 || detector_pixels_v <= 0) return false;
            const int view_count = (int)host_geo_per_view.size();
            if (view_count <= 0) return false;

            out_view_params.assign(view_count, SFDKGeoParamPerView{});

            // ---- unwrap theta ----
            std::vector<float> unwrapped_theta(view_count);
            for (int view = 0; view < view_count; ++view) {
                unwrapped_theta[view] = host_geo_per_view[view].angle.x;
            }
            unwrap_theta_inplace(unwrapped_theta);

            // ---- per view ----
            for (int view = 0; view < view_count; ++view) {
                const SConeProjGeomVec& geo = host_geo_per_view[view];
                SFDKGeoParamPerView& gv = out_view_params[view];

                // 0) meta
                gv.Nu = detector_pixels_u;
                gv.Nv = detector_pixels_v;

                // =========================================================
                // 1) theta
                // =========================================================
                gv.theta = unwrapped_theta[view];

                // =========================================================
                // 2) dtheta
                // =========================================================
                gv.dtheta = compute_dtheta(unwrapped_theta, view, opt_.dtheta_eps);
                gv.fScaleDTheta = 2.f * CUDA_PI / scan_angle_rad;

                // =========================================================
                // 3) du/dv
                // =========================================================
                gv.du_mm = f3_len(f4_to_f3(geo.detU));
                gv.dv_mm = f3_len(f4_to_f3(geo.detV));
                gv.inv_du_mm = 1.0f / gv.du_mm;
                gv.inv_dv_mm = 1.0f / gv.dv_mm;

                // =========================================================
                // 4) (1) central ray dir + SID
                // =========================================================
                const float3 central_ray_dir = f4_to_f3(geo.srcCR);
                gv.SOD_mm = sid_mm_from_source_to_zaxis(f4_to_f3(geo.src)); // SOD_mm field stores SID by your definition

                // =========================================================
                // 4) (2) principal point -> SDD + offsetU/V + ray0hat
                // =========================================================
                // IMPORTANT:
                //   - We compute principal point by intersecting central ray with detector plane.
                //   - We compute pixel coordinate (u,v) of that point and offsets from detector center pixel.
                //   - ray0hat is unit vector from source to principal point.
                compute_SDD_offsets(
                    geo,
                    detector_pixels_u, detector_pixels_v,
                    gv.offsetU_pix,
                    gv.offsetV_pix,
                    gv.SDD_mm,
                    gv.SDD_plane_mm);



                // 5) detector basis cache (for device-side u/v solve)
                compute_detector_basis_cache(
                    geo,
                    gv.UU, gv.VV, gv.UV, gv.invDetUV);

                gv.ray_center = geo.srcCR;
                gv.det_n = f4_normalize(f4_cross(geo.detV, geo.detU));
                gv.det_u = f4_normalize(geo.detU);
                gv.det_v = f4_normalize(geo.detV);
                float3 detS_src = f4_to_f3(geo.detS) - f4_to_f3(geo.src);
                gv.detS_sub_src_dot_dU = dot(detS_src, f4_to_f3(geo.detU));
                gv.detS_sub_src_dot_dV = dot(detS_src, f4_to_f3(geo.detV));
            }

            return true;
        }

        GeoDerivedOptions& options() { return opt_; }
        const GeoDerivedOptions& options() const { return opt_; }

    private:
        // -----------------------------
        // theta unwrap
        // -----------------------------
        static void unwrap_theta_inplace(std::vector<float>& theta)
        {
            for (size_t i = 1; i < theta.size(); ++i) {
                float t = theta[i];
                float p = theta[i - 1];
                while (t - p > CUDA_PI)  t -= 2.f * CUDA_PI;
                while (t - p < -CUDA_PI) t += 2.f * CUDA_PI;
                theta[i] = t;
            }
        }

        // -----------------------------
        // dtheta
        // -----------------------------
        static float compute_dtheta(const std::vector<float>& theta, int index, float eps)
        {
            const int n = (int)theta.size();
            float dt = 0.f;
            if (n <= 1) dt = eps;
            else if (index == 0) dt = theta[1] - theta[0];
            else if (index == n - 1) dt = theta[n - 1] - theta[n - 2];
            else dt = 0.5f * (theta[index + 1] - theta[index - 1]);

            if (dt < 0.f) dt = -dt;
            if (dt < eps) dt = eps;
            return dt;
        }





        // -----------------------------
        // SID by your definition: SID = | d · src |
        // (plane through Z-axis with normal || d, and |d|=1)
        // -----------------------------
        static float sid_mm_from_source_to_zaxis(const float3& src)
        {
            return f3_len(f3(src.x, src.y, 0.f));
        }




        // -----------------------------
        // detector basis cache for device-side solve
        //   Cache UU,VV,UV,invDet for solving D = u*U + v*V
        // -----------------------------
        static void compute_detector_basis_cache(
            const SConeProjGeomVec& geo,
            float& UU, float& VV, float& UV, float& invDetUV)
        {
            float3 detU = f4_to_f3(geo.detU);
            float3 detV = f4_to_f3(geo.detV);

            UU = dot(detU, detU);
            VV = dot(detV, detV);
            UV = dot(detU, detV);

            const float det = UU * VV - UV * UV;
            if (fabsf(det) < 1e-20f) { invDetUV = 0.f; return; }
            invDetUV = 1.f / det;
        }



        // ----------------------------------------------------------------
        // compute_SDD_offsets
        //
        // 计算主射线与探测器平面的交点（principal point），并由此得出：
        //   1. SDD_mm       — 源点沿主射线到 principal point 的实际距离
        //   2. SDD_plane_mm — 源点到探测器平面沿法向量的投影距离，用于求交参数 t
        //   3. offsetU/V    — principal point 相对探测器物理中心的像素偏移
        //
        // 输入：
        //   geo                — 向量几何参数（src, srcCR, detS, detU, detV）
        //   detector_pixels_u  — 探测器 U 方向像素数
        //   detector_pixels_v  — 探测器 V 方向像素数
        //
        // 输出：
        //   out_offsetU_pix — principal point 的 U 像素坐标 - 探测器中心 U 坐标
        //   out_offsetV_pix — principal point 的 V 像素坐标 - 探测器中心 V 坐标
        //   out_SDD_mm      — 主射线实际长度 |principal_point - src|
        //   out_SDD_plane_mm— (detS - src) · det_n，求交分子项
        //
        // 返回 false：探测器法向量退化，或主射线平行于探测器平面
        // ----------------------------------------------------------------
        static bool compute_SDD_offsets(
            const SConeProjGeomVec& geo,
            int detector_pixels_u, int detector_pixels_v,
            float& out_offsetU_pix, float& out_offsetV_pix,
            float& out_SDD_mm, float& out_SDD_plane_mm)
        {
            out_offsetU_pix = out_offsetV_pix = out_SDD_mm = out_SDD_plane_mm = 0.f;

            const float3 src = f4_to_f3(geo.src);
            const float3 detS = f4_to_f3(geo.detS);
            const float3 detU = f4_to_f3(geo.detU);
            const float3 detV = f4_to_f3(geo.detV);

            // 探测器法向量
            float3 n = cross(detV, detU);
            const float n2 = dot(n, n);
            if (n2 < 1e-24f) return false;
            const float3 det_n = n * rsqrtf(n2);

            // 求交参数
            const float denom = dot(det_n, det_n);
            if (fabsf(denom) < 1e-24f) return false;

            const float numer = dot(detS - src, det_n);
            out_SDD_plane_mm = numer;

            // principal point
            float3 P = make_float3(0.f, 0.f, src.z);
            float t = out_SDD_plane_mm / dot(P - src, det_n);
            const float3 principal_point = src + (P - src) * t;

            // SDD_mm
            const float3 ray0 = principal_point - src;
            const float  ray0_len2 = dot(ray0, ray0);
            if (ray0_len2 < 1e-20f) return false;
            out_SDD_mm = sqrtf(ray0_len2);

            // principal point 像素坐标（Cramer）
            const float3 D = principal_point - detS;
            const float  UU = dot(detU, detU);
            const float  VV = dot(detV, detV);
            const float  UV = dot(detU, detV);
            const float  DU = dot(D, detU);
            const float  DV = dot(D, detV);

            const float det_gram = UU * VV - UV * UV;
            if (fabsf(det_gram) < 1e-20f) return false;

            const float invdet = 1.f / det_gram;
            const float u_pix = (DU * VV - DV * UV) * invdet;
            const float v_pix = (-DU * UV + DV * UU) * invdet;

            out_offsetU_pix = u_pix - 0.5f * (detector_pixels_u - 1);
            out_offsetV_pix = v_pix - 0.5f * (detector_pixels_v - 1);

            return true;
        }

    private:
        GeoDerivedOptions opt_;
    };

} // namespace YK
