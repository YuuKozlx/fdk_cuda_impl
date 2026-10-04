#pragma once

#include "cbct_calibration/types.hpp"

#include <functional>

namespace cbct::calibration {

struct OptimizerOptions {
    int max_iterations = 100;
    double robust_scale = 0.15;
    double function_tolerance = 1.0e-11;
    double parameter_tolerance = 1.0e-10;
};

struct OptimizerResult {
    Eigen::VectorXd parameters;
    double cost = 0.0;
    int iterations = 0;
    bool converged = false;
};

using ResidualFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

OptimizerResult levenbergMarquardt(const ResidualFunction& residual,
                                   const Eigen::VectorXd& initial,
                                   const Eigen::VectorXd& scales,
                                   const OptimizerOptions& options = {});

Eigen::Matrix3d rotationFromVector(const Eigen::Vector3d& rotation_vector);
Eigen::Vector3d rotationToVector(const Eigen::Matrix3d& rotation);

}  // namespace cbct::calibration

