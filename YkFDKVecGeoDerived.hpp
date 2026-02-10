#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp" // f3_len, f3_cross, f3_dot, f3_sub, f3_mul, f3_add

namespace YK {

    struct FDKGeoParamGeoDerived
    {
        float du_mm = 1.0f;     // |detU|
        float dv_mm = 1.0f;     // |detV|

        float offsetU_pix = 0.0f;
        float offsetV_pix = 0.0f;
        bool  offset_valid = true;

        float theta = 0.0f;     // unwrapped atan2(src.x, -src.y)
        float dtheta = 0.0f;    // >= eps
    };

    class GeoDerivedManagerVec
    {
    public:
        enum class EOffsetMode { PerView, Median };

        struct GeoDerivedOptions
        {
            EOffsetMode offset_mode = EOffsetMode::PerView;
            float3 isocenter = make_float3(0.0f, 0.0f, 0.0f);
            float dtheta_eps = 1e-8f;
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
            views_.assign(Ang, FDKGeoParamGeoDerived{});
            if (Ang <= 0) return false;

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

            // ---- (2) theta + unwrap ----
            std::vector<float> theta(Ang);
            for (int a = 0; a < Ang; ++a) {
                theta[a] = std::atan2(h_geo[a].src.x, -h_geo[a].src.y);
            }
            for (int a = 1; a < Ang; ++a) {
                float t = theta[a];
                float p = theta[a - 1];
                while (t - p > CUDA_PI) t -= 2.0f * CUDA_PI;
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

            // ---- (4) offsets ----
            if (opt_.offset_mode == EOffsetMode::PerView) {
                for (int a = 0; a < Ang; ++a) {
                    float ou = 0.0f, ov = 0.0f;
                    bool ok = compute_offset_uv_one(h_geo[a], ou, ov);
                    views_[a].offset_valid = ok;
                    views_[a].offsetU_pix = ok ? ou : 0.0f;
                    views_[a].offsetV_pix = ok ? ov : 0.0f;
                }
            }
            else { // Median
                float mu = 0.0f, mv = 0.0f;
                bool ok = compute_offset_uv_median(h_geo, mu, mv);
                for (int a = 0; a < Ang; ++a) {
                    views_[a].offset_valid = ok;
                    views_[a].offsetU_pix = ok ? mu : 0.0f;
                    views_[a].offsetV_pix = ok ? mv : 0.0f;
                }
            }

            return true;
        }

        // ---- getters ----
        const std::vector<FDKGeoParamGeoDerived>& views() const { return views_; }
        float du0_mm() const { return du0_mm_; }
        float dv0_mm() const { return dv0_mm_; }
        int Nu() const { return Nu_; }
        int Nv() const { return Nv_; }

    private:
        static inline __host__ __device__
            float3 dir_from_angle_z(float a_rad)
        {
            // 0°: +x
            float ca = cosf(a_rad);
            float sa = sinf(a_rad);
            return make_float3(ca, sa, 0.0f);
        }

        bool compute_offset_uv_one(const SConeProjectionVec& g, float& ou, float& ov) const
        {
            // ---- detector plane normal ----
            float3 n = f3_cross(g.detU, g.detV);
            float nlen2 = f3_dot(n, n);
            if (nlen2 < 1e-24f) return false;

            float invn = rsqrtf(nlen2);
            float3 nh = make_float3(n.x * invn, n.y * invn, n.z * invn);

            // ---- FIX: central ray direction comes from measured angle (ideal) ----
            const float a_rad = g.angle.x;          // measured gantry angle [rad]
            float3 dir = dir_from_angle_z(a_rad); // <-- choose axis version here

            float denom = f3_dot(dir, nh);
            if (fabs(denom) < 1e-12f) return false;

            float numer = f3_dot(f3_sub(g.detS, g.src), nh);
            float t = numer / denom;

            // robust: flip normal if t <= 0 due to nh orientation
            if (t <= 0.0f) {
                nh = make_float3(-nh.x, -nh.y, -nh.z);
                denom = -denom;
                numer = -numer;
                if (fabs(denom) < 1e-12f) return false;
                t = numer / denom;
                if (t <= 0.0f) return false;
            }

            float3 C = f3_add(g.src, f3_mul(dir, t));
            float3 D = f3_sub(C, g.detS);

            // ---- solve D = cu*detU + cv*detV (general basis) ----
            float UU = f3_dot(g.detU, g.detU);
            float VV = f3_dot(g.detV, g.detV);
            float UV = f3_dot(g.detU, g.detV);
            float DU = f3_dot(D, g.detU);
            float DV = f3_dot(D, g.detV);

            float det = UU * VV - UV * UV;
            if (fabs(det) < 1e-20f) return false;

            float cu = (DU * VV - DV * UV) / det;
            float cv = (-DU * UV + DV * UU) / det;

            ou = cu - 0.5f * (Nu_ - 1);
            ov = cv - 0.5f * (Nv_ - 1);

            return std::isfinite(ou) && std::isfinite(ov);
        }

        bool compute_offset_uv_median(const std::vector<SConeProjectionVec>& h_geo,
            float& ou_med, float& ov_med) const
        {
            std::vector<float> ou, ov;
            ou.reserve(h_geo.size());
            ov.reserve(h_geo.size());

            for (size_t a = 0; a < h_geo.size(); ++a) {
                float u = 0.0f, v = 0.0f;
                if (compute_offset_uv_one(h_geo[a], u, v)) {
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

        std::vector<FDKGeoParamGeoDerived> views_;
        float du0_mm_ = 1.0f;
        float dv0_mm_ = 1.0f;
    };

} // namespace YK
