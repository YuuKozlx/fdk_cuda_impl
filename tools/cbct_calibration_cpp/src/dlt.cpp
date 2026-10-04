#include "cbct_calibration/dlt.hpp"
#include "cbct_calibration/optimizer.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cbct::calibration {
namespace {

struct Normalization2D {
    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    std::vector<Point2> points;
};
struct Normalization3D {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    std::vector<Point3> points;
};

Normalization2D normalize2d(const std::vector<Point2>& source) {
    Normalization2D result;
    Point2 mean = Point2::Zero();
    for (const auto& p : source) mean += p;
    mean /= static_cast<double>(source.size());
    double rms = 0.0;
    for (const auto& p : source) rms += (p - mean).squaredNorm();
    rms = std::sqrt(rms / static_cast<double>(source.size()));
    const double scale = rms > 1.0e-12 ? std::sqrt(2.0) / rms : 1.0;
    result.transform << scale, 0.0, -scale * mean.x(),
                        0.0, scale, -scale * mean.y(), 0.0, 0.0, 1.0;
    result.points.reserve(source.size());
    for (const auto& p : source) {
        const Eigen::Vector3d h = result.transform * p.homogeneous();
        result.points.emplace_back(h.x() / h.z(), h.y() / h.z());
    }
    return result;
}

Normalization3D normalize3d(const std::vector<Point3>& source) {
    Normalization3D result;
    Point3 mean = Point3::Zero();
    for (const auto& p : source) mean += p;
    mean /= static_cast<double>(source.size());
    double rms = 0.0;
    for (const auto& p : source) rms += (p - mean).squaredNorm();
    rms = std::sqrt(rms / static_cast<double>(source.size()));
    const double scale = rms > 1.0e-12 ? std::sqrt(3.0) / rms : 1.0;
    result.transform << scale, 0.0, 0.0, -scale * mean.x(),
                        0.0, scale, 0.0, -scale * mean.y(),
                        0.0, 0.0, scale, -scale * mean.z(),
                        0.0, 0.0, 0.0, 1.0;
    result.points.reserve(source.size());
    for (const auto& p : source) {
        const Eigen::Vector4d h = result.transform * p.homogeneous();
        result.points.emplace_back(h.head<3>() / h.w());
    }
    return result;
}

}  // namespace

ProjectionMatrix normalizedDlt(const std::vector<Point3>& world,
                               const std::vector<Point2>& image) {
    if (world.size() != image.size() || world.size() < 6) {
        throw std::invalid_argument("DLT needs at least six 3-D/2-D pairs");
    }
    const auto nw = normalize3d(world);
    const auto ni = normalize2d(image);
    Eigen::MatrixXd a(2 * static_cast<Eigen::Index>(world.size()), 12);
    a.setZero();
    for (std::size_t i = 0; i < world.size(); ++i) {
        const auto& x = nw.points[i];
        const auto& q = ni.points[i];
        const double X[4] = {x.x(), x.y(), x.z(), 1.0};
        const double u = q.x();
        const double v = q.y();
        for (int k = 0; k < 4; ++k) {
            a(2 * static_cast<Eigen::Index>(i), k) = -X[k];
            a(2 * static_cast<Eigen::Index>(i), 8 + k) = u * X[k];
            a(2 * static_cast<Eigen::Index>(i) + 1, 4 + k) = -X[k];
            a(2 * static_cast<Eigen::Index>(i) + 1, 8 + k) = v * X[k];
        }
    }
    const Eigen::VectorXd h = a.jacobiSvd(Eigen::ComputeFullV).matrixV().col(11);
    ProjectionMatrix pn;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) pn(r, c) = h(4 * r + c);
    ProjectionMatrix result = ni.transform.inverse() * pn * nw.transform;
    const double scale = result.norm();
    if (scale <= 1.0e-15) throw std::runtime_error("degenerate DLT matrix");
    return result / scale;
}

Point2 projectPoint(const ProjectionMatrix& projection, const Point3& point) {
    const Eigen::Vector3d q = projection * point.homogeneous();
    if (std::abs(q.z()) < 1.0e-15) throw std::runtime_error("point projects at infinity");
    return q.head<2>() / q.z();
}

double reprojectionRmse(const ProjectionMatrix& projection,
                        const std::vector<Point3>& world,
                        const std::vector<Point2>& image) {
    double sum = 0.0;
    for (std::size_t i = 0; i < world.size(); ++i)
        sum += (projectPoint(projection, world[i]) - image[i]).squaredNorm();
    return std::sqrt(sum / static_cast<double>(world.size()));
}

DltCamera calibratePhysicalCamera(const std::vector<Point3>& world,
                                  const std::vector<Point2>& image,
                                  PixelSize pixel,
                                  int max_iterations) {
    const ProjectionMatrix p = normalizedDlt(world, image);
    DltCamera camera;
    camera.projection = p;
    camera.reprojection_rmse_px = reprojectionRmse(p, world, image);
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(p, Eigen::ComputeFullV);
    const Eigen::Vector4d c = svd.matrixV().col(3);
    camera.source_phantom_mm = c.head<3>() / c.w();

    // RQ decomposition of the left 3x3 block gives a pixel-space camera matrix.
    Eigen::Matrix3d m = p.leftCols<3>();
    const Eigen::Matrix3d j = (Eigen::Matrix3d() << 0, 0, 1, 0, 1, 0, 1, 0, 0).finished();
    Eigen::HouseholderQR<Eigen::Matrix3d> qr(j * m.transpose() * j);
    Eigen::Matrix3d q = j * qr.householderQ().transpose() * j;
    Eigen::Matrix3d k = j * qr.matrixQR().template triangularView<Eigen::Upper>().transpose() * j;
    for (int i = 0; i < 3; ++i) {
        if (k(i, i) < 0.0) { k.col(i) *= -1.0; q.row(i) *= -1.0; }
    }
    if (q.determinant() < 0.0) { k *= -1.0; q *= -1.0; }
    if (std::abs(k(2, 2)) < 1.0e-15)
        throw std::runtime_error("degenerate intrinsic matrix");
    k /= k(2, 2);
    const double focal_u = std::abs(k(0, 0));
    const double focal_v = std::abs(k(1, 1));
    const double sdd0 = 0.5 * (focal_u * pixel.u_mm + focal_v * pixel.v_mm);
    const Point2 principal0(k(0, 2), k(1, 2));

    // DLT gives an excellent projective initialization, but its RQ factors
    // are very sensitive to a one-pixel point error. Refit the same points
    // with the physical pinhole model so SDD is not read directly from a
    // noisy projective factorization.
    Eigen::VectorXd initial(9);
    initial(0) = std::log(std::max(1.0, sdd0));
    initial.segment<2>(1) = principal0;
    initial.segment<3>(3) = rotationToVector(q);
    initial.segment<3>(6) = camera.source_phantom_mm;
    Eigen::VectorXd scales(9);
    scales << 1.0, 100.0, 100.0, 0.05, 0.05, 0.05, 100.0, 100.0, 100.0;

    const auto physicalProjection = [&](const Eigen::VectorXd& x) {
        const double sdd = std::exp(x(0));
        const double fx = sdd / pixel.u_mm;
        const double fy = sdd / pixel.v_mm;
        const Point2 principal(x(1), x(2));
        const Eigen::Matrix3d rotation = rotationFromVector(x.segment<3>(3));
        const Point3 center = x.segment<3>(6);
        ProjectionMatrix projection = ProjectionMatrix::Zero();
        Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Identity();
        intrinsic(0, 0) = fx; intrinsic(1, 1) = fy;
        intrinsic(0, 2) = principal.x(); intrinsic(1, 2) = principal.y();
        projection.leftCols<3>() = intrinsic * rotation;
        projection.col(3) = intrinsic * (-rotation * center);
        return projection;
    };
    std::vector<std::size_t> active;
    active.reserve(world.size());
    for (std::size_t i = 0; i < world.size(); ++i) active.push_back(i);
    const auto residual = [&](const Eigen::VectorXd& x) {
        const ProjectionMatrix projection = physicalProjection(x);
        Eigen::VectorXd values(2 * static_cast<Eigen::Index>(active.size()));
        for (std::size_t k = 0; k < active.size(); ++k) {
            const std::size_t i = active[k];
            const Point2 predicted = projectPoint(projection, world[i]);
            const Point2 error = predicted - image[i];
            values(2 * static_cast<Eigen::Index>(k)) = error.x();
            values(2 * static_cast<Eigen::Index>(k) + 1) = error.y();
        }
        return values;
    };
    OptimizerOptions options;
    options.max_iterations = std::max(0, max_iterations);
    options.function_tolerance = 1.0e-12;
    options.parameter_tolerance = 1.0e-10;
    Eigen::VectorXd parameters = initial;
    // A merged bead or a clipped bead can move one centroid by several
    // pixels. Refit after removing only those point measurements. The frame
    // remains present and the final camera is still evaluated on all points.
    for (int pass = 0; pass < 3; ++pass) {
        const auto optimized = levenbergMarquardt(residual, parameters, scales, options);
        parameters = optimized.parameters;
        if (active.size() < 8 || pass == 2) break;
        const ProjectionMatrix current = physicalProjection(parameters);
        std::vector<double> errors; errors.reserve(active.size());
        for (const std::size_t i : active)
            errors.push_back((projectPoint(current, world[i]) - image[i]).norm());
        std::vector<double> sorted = errors;
        std::sort(sorted.begin(), sorted.end());
        const double median_error = sorted[sorted.size() / 2];
        const double cutoff = std::max(0.75, 3.0 * median_error);
        std::vector<std::size_t> kept;
        kept.reserve(active.size());
        for (std::size_t k = 0; k < active.size(); ++k)
            if (errors[k] <= cutoff) kept.push_back(active[k]);
        if (kept.size() == active.size() || kept.size() < 8) break;
        active.swap(kept);
    }
    const ProjectionMatrix fitted = physicalProjection(parameters);
    const double fitted_rmse = reprojectionRmse(fitted, world, image);
    if (std::isfinite(fitted_rmse) && fitted_rmse < 1.0e6) {
        camera.projection = fitted;
        camera.reprojection_rmse_px = fitted_rmse;
        camera.sdd_mm = std::exp(parameters(0));
        camera.principal_point_px = parameters.segment<2>(1);
    } else {
        camera.sdd_mm = sdd0;
        camera.principal_point_px = principal0;
    }
    if (!std::isfinite(camera.sdd_mm) || camera.sdd_mm <= 0.0) camera.sdd_mm = 770.0;
    return camera;
}

}  // namespace cbct::calibration
