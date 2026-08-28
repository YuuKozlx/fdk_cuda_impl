#pragma once

#include <cuda_runtime.h>

namespace YK {

struct alignas(16) SFDKGeoParamPerView {
    float theta = 0.f;
    float dtheta = 0.f;
    float fScaleDTheta = 1.f;
    float source_to_axis_mm = 0.f;
    float source_to_radial_detector_mm = 0.f;
    float source_to_detector_plane_mm = 0.f;
    float offsetU_pix = 0.f;
    float offsetV_pix = 0.f;
    float du_mm = 1.f;
    float dv_mm = 1.f;
    float inv_du_mm = 1.f;
    float inv_dv_mm = 1.f;
    int Nu = 0;
    int Nv = 0;
    float UU = 0.f;
    float VV = 0.f;
    float UV = 0.f;
    float invDetUV = 0.f;
    float detS_sub_src_dot_dU = 0.f;
    float detS_sub_src_dot_dV = 0.f;
    float4 radial_ray = {0, 0, 0, 0}; // source -> closest point on rotation axis
    float4 det_n = {0, 0, 0, 0};
    float4 det_u = {0, 0, 0, 0};
    float4 det_v = {0, 0, 0, 0};
};

// 用于 __constant__ 数组，不能增加动态初始化器。
struct alignas(16) FdkAffineCoeff {
    float Cu_x, Cu_y, Cu_z, Cu_w;
    float Cv_x, Cv_y, Cv_z, Cv_w;
    float Cd_x, Cd_y, Cd_z, Cd_w;
    // 深度权重分母：dot(P - src, radial_ray)。
    // Cd 仍专用于与探测器平面求交，二者在探测器倾斜时不相同。
    float Cr_x, Cr_y, Cr_z, Cr_w;
    float dtheta;
    float source_to_axis_sq;
    float fScaleDTheta;
    float source_to_detector_plane_sq;
    float du_mm;
    float dv_mm;
    float inv_SDD_plane;
    float L2_0;
    float L2_u;
    float L2_v;
    float L2_uu;
    float L2_uv;
    float L2_vv;
};

// FdkAffineCoeff 为 128 字节，32 个视图仅占 4 KiB 常量内存。
inline constexpr int kMaxChunkAng = 32;

} // namespace YK
