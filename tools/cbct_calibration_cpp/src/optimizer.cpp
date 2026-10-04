#include "cbct_calibration/optimizer.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

namespace cbct::calibration {

Eigen::Matrix3d rotationFromVector(const Eigen::Vector3d& v) {
    const double angle = v.norm();
    if (angle < 1.0e-14) {
        const Eigen::Matrix3d skew((Eigen::Matrix3d() <<
            0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0).finished());
        return Eigen::Matrix3d::Identity() + skew;
    }
    return Eigen::AngleAxisd(angle, v / angle).toRotationMatrix();
}

Eigen::Vector3d rotationToVector(const Eigen::Matrix3d& r) {
    Eigen::AngleAxisd angle_axis(r);
    return angle_axis.angle() * angle_axis.axis();
}

OptimizerResult levenbergMarquardt(const ResidualFunction& residual,
                                   const Eigen::VectorXd& initial,
                                   const Eigen::VectorXd& scales,
                                   const OptimizerOptions& options) {
    const Eigen::VectorXd scale = scales.array().max(1.0e-9);
    Eigen::VectorXd z = initial.cwiseQuotient(scale);
    auto robustResidual = [&](const Eigen::VectorXd& scaled_value) {
        Eigen::VectorXd r = residual(scaled_value.cwiseProduct(scale));
        // Keep the core optimizer strictly least-squares. Robust losses belong
        // in the measurement layer; applying an ad-hoc residual transform here
        // changes the camera objective and can hide a bad initialization.
        return r;
    };
    Eigen::VectorXd r = robustResidual(z);
    double cost = 0.5 * r.squaredNorm();
    double damping = 1.0e-3;
    OptimizerResult result;
    result.parameters = initial;
    result.cost = cost;
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        Eigen::MatrixXd jacobian(r.size(), z.size());
        for (Eigen::Index j = 0; j < z.size(); ++j) {
            const double step = 1.0e-6 * std::max(1.0, std::abs(z(j)));
            Eigen::VectorXd xp = z;
            xp(j) += step;
            jacobian.col(j) = (robustResidual(xp) - r) / step;
        }
        const Eigen::MatrixXd normal = jacobian.transpose() * jacobian;
        const Eigen::VectorXd gradient = jacobian.transpose() * r;
        Eigen::MatrixXd damped = normal;
        damped.diagonal().array() += damping * (normal.diagonal().array().abs() + 1.0);
        const Eigen::VectorXd delta = damped.ldlt().solve(-gradient);
        if (!delta.allFinite()) break;
        Eigen::VectorXd candidate = z + delta;
        Eigen::VectorXd candidate_residual = robustResidual(candidate);
        const double candidate_cost = 0.5 * candidate_residual.squaredNorm();
        if (candidate_cost < cost) {
            const double improvement = cost - candidate_cost;
            z = candidate;
            r = candidate_residual;
            cost = candidate_cost;
            damping = std::max(1.0e-12, damping * 0.35);
            result.parameters = z.cwiseProduct(scale);
            result.cost = cost;
            result.iterations = iteration + 1;
            if (improvement < options.function_tolerance ||
                delta.norm() < options.parameter_tolerance * (1.0 + z.norm())) {
                result.converged = true;
                return result;
            }
        } else {
            damping = std::min(1.0e12, damping * 8.0);
        }
    }
    result.parameters = z.cwiseProduct(scale);
    result.cost = cost;
    return result;
}

}  // namespace cbct::calibration
