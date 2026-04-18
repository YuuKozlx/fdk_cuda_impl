#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include <vector_functions.hpp>

#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "util/YkVecOperation.hpp" // f3_len, f3_cross, f3_dot, f3_sub, f3_mul, f3_add
#include "util/helper_math.h"

namespace YK {

    // ============================================================
    // GeoDerivedManagerVec (stateless builder)
    // Ŀ�꣺ֻд gv�����ø�ֵ�������������� result �ṹ��,����FDK��Kernel�������м��������
    //
    // �ؼ�Լ��/���壺
    //   - �������Ĺ̶�Ϊ (0,0,0)��isocenter Ŀǰ�������ֶΣ���������㣩
    //   - ������ʼ���� XY ƽ�棺d(theta)=(cos, sin, 0)
    //   - principal point����������̽����ƽ��Ľ���
    //   - offsetU/V��principal point ��Ӧ������������������ص�ƫ�ƣ�pixel��
    //
    // SID����̶��Ķ��壩��
    //   - ƽ�� ��(theta)������ Z �ᣬ���߷��� �� d(theta)
    //   - d(theta)=(cos,sin,0), |d|=1 ʱ����(theta): d��X=0
    //   - SID = | d��src |
    //
    // ����˳���ϸ�������Ҫ�󣩣�
    //   1) theta
    //   2) dtheta
    //   3) du/dv
    //   4) (1) �����߷��� + SID
    //      (2) principal point -> SDD + offsetU/V + ray0hat
    //   5) nhat + DSD_n + detector basis cache + invU2/invV2
    //
    // ע�⣺
    //   - ��ʱ�������ֲ���ת�����Ա�֤ t>0�������ı���洢�ķ��߶����߼���
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
        // SID by your definition: SID = | d �� src |
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
        // ������������̽����ƽ��Ľ��㣨principal point�������ɴ˵ó���
        //   1. SDD_mm       �� Դ���������ߵ� principal point ��ʵ�ʾ���
        //   2. SDD_plane_mm �� Դ�㵽̽����ƽ���ط�������ͶӰ���룬�����󽻲��� t
        //   3. offsetU/V    �� principal point ���̽�����������ĵ�����ƫ��
        //
        // ���룺
        //   geo                �� �������β�����src, srcCR, detS, detU, detV��
        //   detector_pixels_u  �� ̽���� U ����������
        //   detector_pixels_v  �� ̽���� V ����������
        //
        // �����
        //   out_offsetU_pix �� principal point �� U �������� - ̽�������� U ����
        //   out_offsetV_pix �� principal point �� V �������� - ̽�������� V ����
        //   out_SDD_mm      �� ������ʵ�ʳ��� |principal_point - src|
        //   out_SDD_plane_mm�� (detS - src) �� det_n���󽻷�����
        //
        // ���� false��̽�����������˻�����������ƽ����̽����ƽ��
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

            // ̽����������
            float3 n = cross(detV, detU);
            const float n2 = dot(n, n);
            if (n2 < 1e-24f) return false;
            const float3 det_n = n * rsqrtf(n2);

            // �󽻲���
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

            // principal point �������꣨Cramer��
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
