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
    // GeoDerivedManagerVec
    // 约定：
    //   - 外部输入 g.angle.x = gantry angle (rad)
    //   - theta = 0 时，主射线方向为世界坐标 +X
    //   - 任意角度 theta：主射线方向 d(theta) = (cos(theta), sin(theta), 0)
    //   - 主射线与探测器平面交点 C 为 principal point
    //   - offsetU/offsetV 为 principal point 相对“探测器中心像素”的偏移（单位：pixel）
    //
    // 由此计算 per-view:
    //   du_mm, dv_mm, theta, dtheta, offsetU_pix, offsetV_pix, SID_mm, SDD_mm
    //
    // 注意：这里假定 SConeProjectionVec：
    //   detS = 像素 (0,0) 的世界坐标
    //   detU = u 方向每像素步长向量（长度=du）
    //   detV = v 方向每像素步长向量（长度=dv）
    // ============================================================
    class GeoDerivedManagerVec
    {
    public:
        enum class EOffsetMode { PerView, Median };

        struct GeoDerivedOptions
        {
            EOffsetMode offset_mode = EOffsetMode::PerView;
            float3 isocenter = make_float3(0.0f, 0.0f, 0.0f);

            // dtheta 最小值，避免 0
            float dtheta_eps = 1e-8f;

            // 允许 SID 采用绝对值；如需强制 isocenter 在射线前方，可在代码里检查 sid_signed>0
            bool sid_abs = true;
        };

        bool init(int Nu, int Nv, GeoDerivedOptions opt = GeoDerivedOptions{})
        {
            Nu_ = Nu;
            Nv_ = Nv;
            opt_ = opt;
            inited_ = (Nu_ > 0 && Nv_ > 0);
            du0_mm_ = 1.0f;
            dv0_mm_ = 1.0f;
            return inited_;
        }

        // 输入 vec 几何，输出 per-view 派生参数数组
        bool build_geo_params(const std::vector<SConeProjectionVec>& h_geo)
        {
            if (!inited_) return false;
            const int Ang = (int)h_geo.size();
            if (Ang <= 0) return false;

            views_.assign(Ang, SFDKGeoParamPerView{});

            // ---- (1) du/dv ----
            for (int a = 0; a < Ang; ++a) {
                float du = f3_len(h_geo[a].detU);
                float dv = f3_len(h_geo[a].detV);
                if (!(du > 0.0f)) du = 1.0f;
                if (!(dv > 0.0f)) dv = 1.0f;
                views_[a].du_mm = du;
                views_[a].dv_mm = dv;
            }
            du0_mm_ = views_[0].du_mm;
            dv0_mm_ = views_[0].dv_mm;

            // ---- (2) theta：直接使用外部输入角度 + unwrap ----
            std::vector<float> theta(Ang);
            for (int a = 0; a < Ang; ++a) {
                theta[a] = h_geo[a].angle.x;  // external gantry angle [rad]
            }
            for (int a = 1; a < Ang; ++a) {
                float t = theta[a];
                float p = theta[a - 1];
                while (t - p > CUDA_PI)  t -= 2.0f * CUDA_PI;
                while (t - p < -CUDA_PI) t += 2.0f * CUDA_PI;
                theta[a] = t;
            }
            for (int a = 0; a < Ang; ++a) views_[a].theta = theta[a];

            // ---- (3) dtheta ----
            for (int a = 0; a < Ang; ++a) {
                float dt = 0.0f;
                if (Ang == 1) dt = opt_.dtheta_eps;
                else if (a == 0) dt = theta[1] - theta[0];
                else if (a == Ang - 1) dt = theta[Ang - 1] - theta[Ang - 2];
                else dt = 0.5f * (theta[a + 1] - theta[a - 1]);

                if (dt < 0.0f) dt = -dt;
                if (dt < opt_.dtheta_eps) dt = opt_.dtheta_eps;
                views_[a].dtheta = dt;
            }

            // ---- (4) offsets：主射线与探测器平面交点 -> principal point -> offset ----
            if (opt_.offset_mode == EOffsetMode::PerView) {
                for (int a = 0; a < Ang; ++a) {
                    float ou = 0.0f, ov = 0.0f;
                    const bool ok = compute_offset_uv_one(h_geo[a], views_[a].theta, ou, ov);
                    views_[a].offset_valid = ok;
                    views_[a].offsetU_pix = ok ? ou : 0.0f;
                    views_[a].offsetV_pix = ok ? ov : 0.0f;
                }
            }
            else { // Median
                float mu = 0.0f, mv = 0.0f;
                const bool ok = compute_offset_uv_median(h_geo, theta, mu, mv);
                for (int a = 0; a < Ang; ++a) {
                    views_[a].offset_valid = ok;
                    views_[a].offsetU_pix = ok ? mu : 0.0f;
                    views_[a].offsetV_pix = ok ? mv : 0.0f;
                }
            }

            // ---- (5) SDD_mm：source -> principal point 距离 ----
            // principal point = detector center pixel + offset
            for (int a = 0; a < Ang; ++a) {
                const auto& g = h_geo[a];

                float u0 = 0.5f * (float)(Nu_ - 1);
                float v0 = 0.5f * (float)(Nv_ - 1);

                if (views_[a].offset_valid) {
                    u0 += views_[a].offsetU_pix;
                    v0 += views_[a].offsetV_pix;
                }

                // principal point world coord
                float3 detC = f3_add(g.detS, f3_add(f3_mul(g.detU, u0), f3_mul(g.detV, v0)));

                // SDD = |detC - src|
                views_[a].SDD_mm = f3_len(f3_sub(detC, g.src));
            }

            // ---- (6) SID_mm：source -> isocenter 沿主射线方向的距离 ----
            // d(theta)=Rz(theta)*(+X)=(cos, sin, 0)
            for (int a = 0; a < Ang; ++a) {
                const auto& g = h_geo[a];
                float3 dir = dir_from_angle_z(views_[a].theta);

                float3 iso_minus_src = f3_sub(opt_.isocenter, g.src);
                float sod_signed = f3_dot(iso_minus_src, dir);

                float sod = opt_.sid_abs ? fabsf(sod_signed) : sod_signed;
                // 你也可以选择：若 sid_signed<=0 视为无效
                // views_[a].sid_valid = (sid_signed > 0);

                // 若你的 SFDKGeoParamPerView 没有 SID_mm 字段：
                //   - 请改成你自己的字段名（例如 SOD_mm）
                views_[a].SOD_mm = sod;
            }

            return true;
        }

        // ---- getters ----
        const std::vector<SFDKGeoParamPerView>& views() const { return views_; }
        float du0_mm() const { return du0_mm_; }
        float dv0_mm() const { return dv0_mm_; }
        int Nu() const { return Nu_; }
        int Nv() const { return Nv_; }

    private:
        static inline __host__ __device__
            float3 dir_from_angle_z(float a_rad)
        {
            // 约定：theta=0 -> +X
            float ca = cosf(a_rad);
            float sa = sinf(a_rad);
            return make_float3(ca, sa, 0.0f);
        }

        // 计算单 view offset：用主射线方向 d(theta) 与探测器平面求交，得到 principal point
        bool compute_offset_uv_one(const SConeProjectionVec& g, float theta_rad, float& ou, float& ov) const
        {
            // ---- detector plane normal ----
            float3 n = f3_cross(g.detU, g.detV);
            float nlen2 = f3_dot(n, n);
            if (nlen2 < 1e-24f) return false;

            float invn = rsqrtf(nlen2);
            float3 nh = make_float3(n.x * invn, n.y * invn, n.z * invn);

            // ---- central ray direction from external angle ----
            float3 dir = dir_from_angle_z(theta_rad);

            // ray: X(t) = src + t * dir
            // plane: (X - detS)·nh = 0
            float denom = f3_dot(dir, nh);
            if (fabs(denom) < 1e-12f) return false;

            float numer = f3_dot(f3_sub(g.detS, g.src), nh);
            float t = numer / denom;

            // robust: flip normal if intersection behind the source due to nh orientation
            if (t <= 0.0f) {
                nh = make_float3(-nh.x, -nh.y, -nh.z);
                denom = -denom;
                numer = -numer;
                if (fabs(denom) < 1e-12f) return false;
                t = numer / denom;
                if (t <= 0.0f) return false;
            }

            float3 C = f3_add(g.src, f3_mul(dir, t)); // principal point in world
            float3 D = f3_sub(C, g.detS);             // vector from detS to principal point

            // ---- solve D = cu*detU + cv*detV (general 2D basis on detector plane) ----
            float UU = f3_dot(g.detU, g.detU);
            float VV = f3_dot(g.detV, g.detV);
            float UV = f3_dot(g.detU, g.detV);
            float DU = f3_dot(D, g.detU);
            float DV = f3_dot(D, g.detV);

            float det = UU * VV - UV * UV;
            if (fabs(det) < 1e-20f) return false;

            float cu = (DU * VV - DV * UV) / det;
            float cv = (-DU * UV + DV * UU) / det;

            // offset in pixel relative to detector center pixel
            ou = cu - 0.5f * (Nu_ - 1);
            ov = cv - 0.5f * (Nv_ - 1);

            return std::isfinite(ou) && std::isfinite(ov);
        }

        bool compute_offset_uv_median(const std::vector<SConeProjectionVec>& h_geo,
            const std::vector<float>& theta,
            float& ou_med, float& ov_med) const
        {
            std::vector<float> ou, ov;
            ou.reserve(h_geo.size());
            ov.reserve(h_geo.size());

            for (size_t a = 0; a < h_geo.size(); ++a) {
                float u = 0.0f, v = 0.0f;
                if (compute_offset_uv_one(h_geo[a], theta[a], u, v)) {
                    ou.push_back(u);
                    ov.push_back(v);
                }
            }
            if (ou.empty()) return false;

            auto median = [](std::vector<float>& x) -> float {
                const size_t n = x.size();
                const size_t mid = n / 2;
                std::nth_element(x.begin(), x.begin() + mid, x.end());
                float m = x[mid];
                if ((n & 1u) == 0u) {
                    std::nth_element(x.begin(), x.begin() + (mid - 1), x.end());
                    m = 0.5f * (m + x[mid - 1]);
                }
                return m;
                };

            ou_med = median(ou);
            ov_med = median(ov);
            return true;
        }

    private:
        int Nu_ = 0;
        int Nv_ = 0;
        GeoDerivedOptions opt_;
        bool inited_ = false;

        std::vector<SFDKGeoParamPerView> views_;
        float du0_mm_ = 1.0f;
        float dv0_mm_ = 1.0f;
    };

} // namespace YK
