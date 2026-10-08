#pragma once

#include <limits>

namespace YK::Iter {

// 迭代重建的通用停止条件。所有浮点阈值取 0 时禁用对应条件，
// iterations 始终作为硬上限，因此默认配置保持原有固定次数行为。
struct IterativeConvergenceConfig {
    float relative_residual_tolerance = 0.f;    // ||b-Ax||_2 / ||b||_2
    float relative_update_tolerance = 0.f;      // ||x_k-x_prev||_2 / ||x_prev||_2
    float relative_improvement_tolerance = 0.f; // 相对残差改善量不足阈值
    int minimum_iterations = 1;
    int check_interval = 1;
    int patience = 1; // 条件连续满足多少次才停止
};

struct IterativeConvergenceStatistics {
    int completed_iterations = 0;
    int convergence_checks = 0;
    float projection_residual_l2 = std::numeric_limits<float>::quiet_NaN();
    float relative_projection_residual = std::numeric_limits<float>::quiet_NaN();
    float relative_volume_update = std::numeric_limits<float>::quiet_NaN();
    float relative_residual_improvement = std::numeric_limits<float>::quiet_NaN();
    bool stopped_by_relative_residual = false;
    bool stopped_by_relative_update = false;
    bool stopped_by_stagnation = false;
    bool stopped_by_breakdown = false;
    bool stopped_by_divergence = false;
};

inline bool convergenceEnabled(const IterativeConvergenceConfig& config)
{
    return config.relative_residual_tolerance > 0.f ||
        config.relative_update_tolerance > 0.f ||
        config.relative_improvement_tolerance > 0.f;
}

inline bool validConvergenceConfig(const IterativeConvergenceConfig& config)
{
    return config.relative_residual_tolerance >= 0.f &&
        config.relative_update_tolerance >= 0.f &&
        config.relative_improvement_tolerance >= 0.f &&
        config.minimum_iterations >= 0 && config.check_interval > 0 &&
        config.patience > 0;
}

} // namespace YK::Iter
