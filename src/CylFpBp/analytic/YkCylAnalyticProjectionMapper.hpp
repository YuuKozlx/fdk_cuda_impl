#pragma once

#include <cmath>

#include "CylFpBp/analytic/kernels/YkCylAnalyticProjectionMapLaunch.cuh"
#include "global/YkKernelLaunchPolicy.hpp"

namespace YK::CylFpBp::Analytic {

// 一般同轴圆柱投影到源中心等角虚拟圆柱的映射参数。输入和输出布局均为
// [view][row][channel]；映射只重采样已经取对数的线积分，不附加解析滤波
// 所需的 Jacobian 或权重，那些量应由后续算法在虚拟等角坐标中计算。
struct ProjectionMapConfig {
    int channels = 0;
    int rows = 0;
    int views = 0;

    float source_to_detector_mm = 0.f;       // 虚拟圆柱半径，即 SDD
    float physical_radius_mm = 0.f;          // 物理圆柱曲率半径 R
    float physical_arc_step_mm = 0.f;        // 物理相邻通道弧长
    float physical_row_step_mm = 0.f;
    float physical_principal_u = 0.f;
    float physical_principal_v = 0.f;

    float target_angle_step_rad = 0.f;       // 虚拟等角通道步长
    float target_row_step_mm = 0.f;
    float target_principal_u = 0.f;
    float target_principal_v = 0.f;
    SKernelLaunchPolicy launch{};
};

class ProjectionMapper {
public:
    bool prepare(const ProjectionMapConfig& config)
    {
        release();
        if (config.channels < 2 || config.rows < 2 || config.views <= 0 ||
            !(config.source_to_detector_mm > 0.f) ||
            !(config.physical_radius_mm > 0.f) ||
            !(config.physical_arc_step_mm > 0.f) ||
            !(config.physical_row_step_mm > 0.f) ||
            !(config.target_angle_step_rad > 0.f) ||
            !(config.target_row_step_mm > 0.f))
            return false;

        // 整个目标扇角必须落在物理圆柱相对于源点可见的同一单调分支。
        const float center_distance = config.source_to_detector_mm -
            config.physical_radius_mm;
        const auto visible = [&](float beta) {
            const float sine = std::sin(beta);
            const float cosine = std::cos(beta);
            const float discriminant = config.physical_radius_mm *
                config.physical_radius_mm - center_distance * center_distance *
                sine * sine;
            if (discriminant < 0.f) return false;
            const float rho = center_distance * cosine + std::sqrt(discriminant);
            return rho > 0.f;
        };
        const float left = (0.f - config.target_principal_u) *
            config.target_angle_step_rad;
        const float right = (config.channels - 1.f - config.target_principal_u) *
            config.target_angle_step_rad;
        if (!visible(left) || !visible(right)) return false;

        config_ = config;
        prepared_ = true;
        return true;
    }

    bool apply(const float* physical_projection, float* equiangular_projection,
        cudaStream_t stream) const
    {
        if (!prepared_ || !physical_projection || !equiangular_projection ||
            !stream) return false;
        detail::launch_cylindrical_to_equiangular_map(physical_projection,
            equiangular_projection, config_, stream);
        return true;
    }

    void release()
    {
        config_ = {};
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }
    const ProjectionMapConfig& config() const { return config_; }

private:
    ProjectionMapConfig config_{};
    bool prepared_ = false;
};

} // namespace YK::CylFpBp::Analytic
