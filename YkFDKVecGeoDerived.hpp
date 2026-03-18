#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include <vector_functions.hpp>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp" // f3_len, f3_cross, f3_dot, f3_sub, f3_mul, f3_add

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
            int detector_pixels_u, int detector_pixels_v,
            const std::vector<SConeProjectionVec>& host_geo_per_view,
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
                const SConeProjectionVec& geo = host_geo_per_view[view];
                SFDKGeoParamPerView& gv = out_view_params[view];

                // 0) meta
                gv.Nu = detector_pixels_u;
                gv.Nv = detector_pixels_v;
                gv.offset_mode = 0;

                // =========================================================
                // 1) theta
                // =========================================================
                gv.theta = unwrapped_theta[view];

                // =========================================================
                // 2) dtheta
                // =========================================================
                gv.dtheta = compute_dtheta(unwrapped_theta, view, opt_.dtheta_eps);

                // =========================================================
                // 3) du/dv
                // =========================================================
                compute_detector_pixel_size_mm(geo, gv.du_mm, gv.dv_mm);

                // =========================================================
                // 4) (1) central ray dir + SID
                // =========================================================
                const float3 central_ray_dir = central_ray_direction_xy(gv.theta);
                gv.SOD_mm = sid_mm_from_dir_and_source(central_ray_dir, geo.src); // SOD_mm field stores SID by your definition

                // =========================================================
                // 4) (2) principal point -> SDD + offsetU/V + ray0hat
                // =========================================================
                // IMPORTANT:
                //   - We compute principal point by intersecting central ray with detector plane.
                //   - We compute pixel coordinate (u,v) of that point and offsets from detector center pixel.
                //   - ray0hat is unit vector from source to principal point.
                compute_principal_point_SDD_offsets_ray0hat(
                    geo,
                    detector_pixels_u, detector_pixels_v,
                    central_ray_dir,
                    gv.offset_valid,
                    gv.offsetU_pix,
                    gv.offsetV_pix,
                    gv.SDD_mm,
                    gv.ray0hat);

                // =========================================================
                // 5) nhat + DSD_n
                // =========================================================
                compute_detector_plane_normal_and_DSDn(
                    geo, gv.nhat, gv.DSD_n, opt_.force_DSD_positive);

                // 5) detector basis cache (for device-side u/v solve)
                compute_detector_basis_cache(
                    geo,
                    gv.basis_valid,
                    gv.UU, gv.VV, gv.UV, gv.invDetUV);

            }

            return true;
        }

        GeoDerivedOptions& options() { return opt_; }
        const GeoDerivedOptions& options() const { return opt_; }

    private:
        // -----------------------------
        // theta unwrap
        // -----------------------------
        static inline void unwrap_theta_inplace(std::vector<float>& theta)
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
        static inline float compute_dtheta(const std::vector<float>& theta, int index, float eps)
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
        // du/dv
        // -----------------------------
        static inline void compute_detector_pixel_size_mm(
            const SConeProjectionVec& geo,
            float& pixel_u_mm,
            float& pixel_v_mm)
        {
            pixel_u_mm = f3_len(geo.detU);
            if (!(pixel_u_mm > 0.f)) pixel_u_mm = 1.f;

            pixel_v_mm = f3_len(geo.detV);
            if (!(pixel_v_mm > 0.f)) pixel_v_mm = 1.f;
        }

        // -----------------------------
        // central ray direction (XY)
        // -----------------------------
        static inline float3 central_ray_direction_xy(float theta_rad)
        {
            return make_float3(cosf(theta_rad), sinf(theta_rad), 0.f);
        }

        // -----------------------------
        // SID by your definition: SID = | d · src |
        // (plane through Z-axis with normal || d, and |d|=1)
        // -----------------------------
        static inline float sid_mm_from_dir_and_source(const float3& central_ray_dir_unit, const float3& source_world)
        {
            return fabsf(f3_dot(central_ray_dir_unit, source_world));
        }

        // -----------------------------
        // detector plane normal + DSD_n
        //   nhat = normalize(detU x detV)
        //   DSD_n = (detS - src) · nhat
        // -----------------------------
        static inline bool compute_detector_plane_normal_and_DSDn(
            const SConeProjectionVec& geo,
            float3& detector_plane_normal_unit,
            float& source_to_detector_plane_distance_along_normal,
            bool force_DSD_positive)
        {
            float3 n = f3_cross(geo.detU, geo.detV);
            const float n2 = f3_dot(n, n);
            if (n2 < 1e-24f) {
                detector_plane_normal_unit = make_float3(0, 0, 0);
                source_to_detector_plane_distance_along_normal = 0.f;
                return false;
            }
            const float invn = rsqrtf(n2);
            detector_plane_normal_unit = make_float3(n.x * invn, n.y * invn, n.z * invn);

            source_to_detector_plane_distance_along_normal =
                f3_dot(f3_sub(geo.detS, geo.src), detector_plane_normal_unit);

            if (force_DSD_positive && source_to_detector_plane_distance_along_normal < 0.f) {
                detector_plane_normal_unit = f3_mul(detector_plane_normal_unit, -1.f);
                source_to_detector_plane_distance_along_normal =
                    -source_to_detector_plane_distance_along_normal;
            }
            return true;
        }

        // -----------------------------
        // detector basis cache for device-side solve
        //   Cache UU,VV,UV,invDet for solving D = u*U + v*V
        // -----------------------------
        static inline void compute_detector_basis_cache(
            const SConeProjectionVec& geo,
            int& basis_valid,
            float& UU, float& VV, float& UV, float& invDetUV)
        {
            UU = f3_dot(geo.detU, geo.detU);
            VV = f3_dot(geo.detV, geo.detV);
            UV = f3_dot(geo.detU, geo.detV);

            const float det = UU * VV - UV * UV;
            if (fabsf(det) < 1e-20f) {
                basis_valid = 0;
                invDetUV = 0.0f;
                return;
            }

            basis_valid = 1;
            invDetUV = 1.0f / det;
        }

        // -----------------------------
        // principal point + SDD + offsets + ray0hat
        //
        // principal point:
        //   intersection of central ray X(t)=src+t*d with detector plane
        // detector plane:
        //   passes through detS, normal computed as normalize(detU x detV)
        //
        // outputs (write into gv fields via refs):
        //   offset_valid, offsetU_pix, offsetV_pix, SDD_mm, ray0hat
        // -----------------------------
        static inline void compute_principal_point_SDD_offsets_ray0hat(
            const SConeProjectionVec& geo,
            int detector_pixels_u,
            int detector_pixels_v,
            const float3& central_ray_dir_unit,
            int& out_offset_valid,
            float& out_offsetU_pix,
            float& out_offsetV_pix,
            float& out_SDD_mm,
            float3& out_ray0hat)
        {
            out_offset_valid = 0;
            out_offsetU_pix = 0.f;
            out_offsetV_pix = 0.f;
            out_SDD_mm = 0.f;
            out_ray0hat = make_float3(0, 0, 0);

            // local plane normal from U x V (independent; may be locally flipped for t>0)
            float3 nh_local;
            float  DSD_n_local = 0.f;
            if (!compute_detector_plane_normal_and_DSDn(geo, nh_local, DSD_n_local, /*force*/false)) return;

            float denom = f3_dot(central_ray_dir_unit, nh_local);
            if (fabsf(denom) < 1e-12f) return;

            float numer = f3_dot(f3_sub(geo.detS, geo.src), nh_local);
            float t = numer / denom;

            // local flip to enforce forward intersection (t>0)
            if (t <= 0.f) {
                nh_local = make_float3(-nh_local.x, -nh_local.y, -nh_local.z);
                denom = -denom;
                numer = -numer;
                if (fabsf(denom) < 1e-12f) return;
                t = numer / denom;
                if (t <= 0.f) return;
            }

            // principal point on detector plane
            const float3 principal_point_world = f3_add(geo.src, f3_mul(central_ray_dir_unit, t));

            // SDD and ray0hat: ray0 = principal - src
            const float3 ray0 = f3_sub(principal_point_world, geo.src);
            const float  ray0_len2 = f3_dot(ray0, ray0);
            if (ray0_len2 < 1e-20f) return;

            const float inv_len = rsqrtf(ray0_len2);
            out_ray0hat = make_float3(ray0.x * inv_len, ray0.y * inv_len, ray0.z * inv_len);
            out_SDD_mm = sqrtf(ray0_len2);

            // pixel coordinate of principal point:
            //   D = principal - detS = u*detU + v*detV
            const float3 D = f3_sub(principal_point_world, geo.detS);

            const float UU = f3_dot(geo.detU, geo.detU);
            const float VV = f3_dot(geo.detV, geo.detV);
            const float UV = f3_dot(geo.detU, geo.detV);
            const float DU = f3_dot(D, geo.detU);
            const float DV = f3_dot(D, geo.detV);

            const float det = UU * VV - UV * UV;
            if (fabsf(det) < 1e-20f) return;

            const float invdet = 1.f / det;
            const float u_pix = (DU * VV - DV * UV) * invdet;
            const float v_pix = (-DU * UV + DV * UU) * invdet;

            // offsets relative to detector center pixel
            const float center_u = 0.5f * (detector_pixels_u - 1);
            const float center_v = 0.5f * (detector_pixels_v - 1);

            out_offsetU_pix = u_pix - center_u;
            out_offsetV_pix = v_pix - center_v;
            out_offset_valid = true;
        }

    private:
        GeoDerivedOptions opt_;
    };

} // namespace YK
