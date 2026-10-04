#include "cbct_calibration/source_circle.hpp"

#include <Eigen/Geometry>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cbct::calibration {

SourceCircle fitSourceCircle(const std::vector<Point3>& sources) {
    if (sources.size() < 3) throw std::invalid_argument("source circle needs three points");
    Point3 centroid = Point3::Zero();
    for (const auto& p : sources) centroid += p;
    centroid /= static_cast<double>(sources.size());
    Eigen::MatrixXd centered(static_cast<Eigen::Index>(sources.size()), 3);
    for (std::size_t i = 0; i < sources.size(); ++i) centered.row(static_cast<Eigen::Index>(i)) = (sources[i] - centroid).transpose();
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
    Point3 axis = svd.matrixV().col(2);
    if ((sources[1] - centroid).cross(sources[0] - centroid).dot(axis) < 0.0) axis = -axis;
    Point3 radial = sources[0] - centroid - axis * (sources[0] - centroid).dot(axis);
    radial.normalize();
    Point3 tangent = axis.cross(radial).normalized();
    Eigen::MatrixXd xy(static_cast<Eigen::Index>(sources.size()), 2);
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const Point3 d = sources[i] - centroid;
        xy(static_cast<Eigen::Index>(i), 0) = d.dot(radial);
        xy(static_cast<Eigen::Index>(i), 1) = d.dot(tangent);
    }
    Eigen::MatrixXd design(static_cast<Eigen::Index>(sources.size()), 3);
    design << xy.col(0), xy.col(1), Eigen::VectorXd::Ones(static_cast<Eigen::Index>(sources.size()));
    const Eigen::VectorXd rhs = -(xy.array().square().rowwise().sum()).matrix();
    const Eigen::Vector3d fit = design.colPivHouseholderQr().solve(rhs);
    Point3 center = centroid - 0.5 * fit.x() * radial - 0.5 * fit.y() * tangent;
    double radius = 0.0;
    std::vector<double> radii;
    radii.reserve(sources.size());
    double axial_sum = 0.0;
    for (const auto& p : sources) {
        const Point3 d = p - center;
        const double axial = d.dot(axis);
        axial_sum += axial * axial;
        radii.push_back((d - axial * axis).norm());
        radius += radii.back();
    }
    radius /= static_cast<double>(radii.size());
    double radial_error = 0.0;
    for (const double value : radii) radial_error += (value - radius) * (value - radius);
    radial = (sources[0] - center) - axis * (sources[0] - center).dot(axis);
    radial.normalize();
    tangent = axis.cross(radial).normalized();
    return {center, axis, radial, tangent, radius,
            std::sqrt(radial_error / radii.size()),
            std::sqrt(axial_sum / sources.size())};
}

void scannerAlignment(const SourceCircle& circle, Eigen::Matrix3d& rotation,
                      Point3& translation) {
    Point3 axis = circle.axis_phantom;
    Point3 radial = circle.radial0_phantom;
    if (axis.z() < 0.0) { axis = -axis; radial = -radial; }
    Point3 tangent = axis.cross(radial).normalized();
    radial = tangent.cross(axis).normalized();
    Eigen::Matrix3d phantom_basis;
    phantom_basis.col(0) = radial;
    phantom_basis.col(1) = tangent;
    phantom_basis.col(2) = axis;
    Eigen::Matrix3d scanner_basis;
    scanner_basis << 0.0, 1.0, 0.0,
                    -1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0;
    rotation = scanner_basis * phantom_basis.transpose();
    translation = -rotation * circle.center_phantom_mm;
}

}  // namespace cbct::calibration
