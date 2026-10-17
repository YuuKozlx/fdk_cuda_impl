#include "cbct_calibration/optimizer.hpp"
#include "cbct_calibration/single_row.hpp"
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace cbct::calibration {
namespace {
    constexpr double pi = 3.14159265358979323846;
    std::vector<Point2> projectRow(const Eigen::VectorXd& x, const PipelineConfig& c, int direction)
    {
        const double sid = std::exp(x[0]), sdd = sid + std::exp(x[1]);
        Point3 axis(x[5], x[6], 1);
        axis.normalize();
        const Point3 translation = x.segment<3>(7);
        std::vector<Point2> points;
        points.reserve(size_t(c.tracker.views) * c.row_beads);
        for (int view = 0; view < c.tracker.views; ++view) {
            double a = direction * 2 * pi * view / c.tracker.views - pi / 2;
            Point3 radial(std::cos(a), std::sin(a), 0), u(-std::sin(a), std::cos(a), 0), v = Point3::UnitZ();
            u = Eigen::AngleAxisd(x[4], radial) * u;
            v = Eigen::AngleAxisd(x[4], radial) * v;
            for (int j = 0; j < c.row_beads; ++j) {
                Point3 world = translation + axis * ((j - (c.row_beads - 1) * .5) * c.row_spacing_mm);
                double denominator = sid - world.dot(radial);
                if (denominator <= 1e-6)
                    denominator = 1e-6;
                double magnification = sdd / denominator;
                points.emplace_back((c.tracker.image.cols - 1) * .5 + (magnification * world.dot(u) - x[2]) / c.pixel.u_mm,
                    (c.tracker.image.rows - 1) * .5 + (magnification * world.dot(v) - x[3]) / c.pixel.v_mm);
            }
        }
        return points;
    }
}
CalibrationResult calibrateSingleRow(const std::vector<Point2>& measured, const std::vector<unsigned char>& usable,
    const PipelineConfig& c)
{
    if (c.row_beads < 3 || c.tracker.views < 12 || !std::isfinite(c.row_spacing_mm) || c.row_spacing_mm <= 0 || !std::isfinite(c.initial_sid_mm) || !std::isfinite(c.initial_sdd_mm) || c.initial_sid_mm <= 0 || c.initial_sdd_mm <= c.initial_sid_mm || !std::isfinite(c.pixel.u_mm) || !std::isfinite(c.pixel.v_mm) || c.pixel.u_mm <= 0 || c.pixel.v_mm <= 0 || c.tracker.image.rows <= 0 || c.tracker.image.cols <= 0 || c.joint_max_iterations <= 0 || measured.size() != size_t(c.row_beads) * c.tracker.views || usable.size() != measured.size())
        throw std::invalid_argument("Invalid single-row calibration input");
    std::vector<unsigned char> mask = usable;
    int retained = 0;
    for (int j = 0; j < c.row_beads; ++j) {
        int count = 0;
        for (int i = 0; i < c.tracker.views; ++i) {
            size_t k = size_t(i) * c.row_beads + j;
            if (mask[k] && !measured[k].allFinite())
                throw std::invalid_argument("Nonfinite usable single-row point");
            if (mask[k])
                ++count;
        }
        if (count < std::max(12, c.tracker.views / 2)) {
            for (int i = 0; i < c.tracker.views; ++i)
                mask[size_t(i) * c.row_beads + j] = 0;
        } else
            ++retained;
    }
    if (retained < 3)
        throw std::runtime_error("Single-row fit needs at least three beads tracked over half a turn");
    const int count = int(std::count_if(mask.begin(), mask.end(), [](unsigned char b) { return b != 0; }));
    CalibrationResult best;
    best.model = "single_row_five_parameter_no_out_of_plane_tilt";
    double bestCondition = 0;
    for (int direction : { 1, -1 }) {
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(10);
        seed[0] = std::log(c.initial_sid_mm);
        seed[1] = std::log(c.initial_sdd_mm - c.initial_sid_mm);
        // First harmonic provides a starting pose only, not a geometry constraint.
        double sx = 0, sy = 0, sz = 0;
        int frames = 0;
        for (int view = 0; view < c.tracker.views; ++view) {
            Point2 center = Point2::Zero();
            int n = 0;
            for (int j = 0; j < c.row_beads; ++j) {
                size_t k = size_t(view) * c.row_beads + j;
                if (mask[k]) {
                    center += measured[k];
                    ++n;
                }
            }
            if (n != c.row_beads)
                continue;
            center /= n;
            double u = (center.x() - (c.tracker.image.cols - 1) * .5) * c.pixel.u_mm * c.initial_sid_mm / c.initial_sdd_mm;
            double v = (center.y() - (c.tracker.image.rows - 1) * .5) * c.pixel.v_mm * c.initial_sid_mm / c.initial_sdd_mm;
            double a = direction * 2 * pi * view / c.tracker.views;
            sx += 2 * u * std::cos(a);
            sy += 2 * u * std::sin(a);
            sz += v;
            ++frames;
        }
        if (frames) {
            seed[7] = sx / frames;
            seed[8] = sy / frames;
            seed[9] = sz / frames;
        }
        auto residual = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            auto prediction = projectRow(x, c, direction);
            Eigen::VectorXd r(2 * count);
            int n = 0;
            for (size_t k = 0; k < mask.size(); ++k)
                if (mask[k]) {
                    r.segment<2>(2 * n++) = prediction[k] - measured[k];
                }
            return r;
        };
        Eigen::VectorXd scales(10);
        scales << 1, 1, 2, 2, .02, .03, .03, 10, 10, 10;
        OptimizerOptions options;
        options.max_iterations = c.joint_max_iterations;
        auto fit = levenbergMarquardt(residual, seed, scales, options);
        auto r = residual(fit.parameters);
        double rmse = std::sqrt(r.squaredNorm() / count);
        if (!std::isfinite(rmse) || rmse >= best.joint_rmse_px)
            continue;
        Eigen::MatrixXd jac(r.size(), 10);
        for (int j = 0; j < 10; ++j) {
            auto x = fit.parameters;
            double h = 1e-5 * scales[j];
            x[j] += h;
            jac.col(j) = (residual(x) - r) / h;
        }
        for (int j = 0; j < 10; ++j) {
            double n = jac.col(j).norm();
            if (n > 1e-12)
                jac.col(j) /= n;
        }
        auto singular = jac.jacobiSvd().singularValues();
        bestCondition = singular[singular.size() - 1] / singular[0];
        const auto& x = fit.parameters;
        best.machine.sid_mm = std::exp(x[0]);
        best.machine.sdd_mm = best.machine.sid_mm + std::exp(x[1]);
        best.machine.offset_u_mm = x[2];
        best.machine.offset_v_mm = x[3];
        best.machine.tilt_n_rad = x[4];
        Point3 axis(x[5], x[6], 1);
        axis.normalize();
        best.phantom_rotation_vector_rad = rotationToVector(Eigen::Quaterniond::FromTwoVectors(Point3::UnitZ(), axis).toRotationMatrix());
        best.phantom_translation_mm = x.segment<3>(7);
        best.rotation_direction = direction;
        best.joint_rmse_px = rmse;
        best.max_point_error_px = 0;
        for (int j = 0; j < count; ++j)
            best.max_point_error_px = std::max(best.max_point_error_px, r.segment<2>(2 * j).norm());
        best.joint_iterations = fit.iterations;
        best.joint_converged = fit.converged || rmse < 1e-8;
        best.source_circle.radius_mm = best.machine.sid_mm;
    }
    if (!std::isfinite(best.joint_rmse_px) || bestCondition < 1e-7)
        throw std::runtime_error("Single-row geometry is unobservable or ill-conditioned; move the bead line off the rotation axis and provide a full turn");
    return best;
}
}
