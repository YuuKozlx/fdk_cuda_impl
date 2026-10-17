#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace cbct::calibration {

using Point2 = Eigen::Vector2d;
using Point3 = Eigen::Vector3d;
using ProjectionMatrix = Eigen::Matrix<double, 3, 4>;

struct ImageShape {
    int rows = 1024;
    int cols = 1024;
};

struct PixelSize {
    double u_mm = 0.417;
    double v_mm = 0.417;
};

struct PhantomSpec {
    bool markers_for_numbering_only = false;
    // An empty list means an unmarked phantom; one or two entries cover the
    // usual marked variants.
    int beads_per_ring = 12;
    double upper_radius_mm = 50.0;
    double lower_radius_mm = 40.0;
    double ring_half_spacing_mm = 50.0;
    double lower_phase_deg = 15.0;
    double marker_axial_mm = 80.0;
    std::vector<Point3> marker_points_mm{{50.0, 0.0, 80.0}};
};

struct TrackerConfig {
    std::filesystem::path raw_path;
    int views = 360;
    ImageShape image;
    float threshold = 5.0F;
    double gate_radius_px = 28.0;
    int min_pixels = 4;
    double marker_area_ratio = 1.45;
    double max_dlt_rmse_px = 1.0;
};

struct Component {
    Point2 centroid = Point2::Zero();
    int area = 0;
    float peak = 0.0F;
};

struct TrackingResult {
    int views = 0;
    int targets = 0;
    int anchor_frame = -1;
    std::vector<Point2> points;
    std::vector<unsigned char> observed;
    // A merged target remains trackable, but is excluded from the current
    // frame's DLT and joint residual through usable.
    std::vector<unsigned char> merged;
    std::vector<unsigned char> usable;
    std::vector<double> dlt_rmse_px;

    const Point2& at(int view, int target) const {
        return points.at(static_cast<std::size_t>(view * targets + target));
    }
    bool isObserved(int view, int target) const {
        return observed.at(static_cast<std::size_t>(view * targets + target)) != 0;
    }
    bool isUsable(int view, int target) const {
        return usable.at(static_cast<std::size_t>(view * targets + target)) != 0;
    }
};

struct DltCamera {
    ProjectionMatrix projection = ProjectionMatrix::Zero();
    Point3 source_phantom_mm = Point3::Zero();
    double sdd_mm = 0.0;
    Point2 principal_point_px = Point2::Zero();
    double reprojection_rmse_px = std::numeric_limits<double>::infinity();
};

struct SourceCircle {
    Point3 center_phantom_mm = Point3::Zero();
    Point3 axis_phantom = Point3::UnitZ();
    Point3 radial0_phantom = -Point3::UnitY();
    Point3 tangent0_phantom = Point3::UnitX();
    double radius_mm = 0.0;
    double radial_rms_mm = 0.0;
    double axial_rms_mm = 0.0;
};

struct MachineGeometry {
    double sid_mm = 0.0;
    double sdd_mm = 0.0;
    double offset_u_mm = 0.0;
    double offset_v_mm = 0.0;
    double tilt_u_rad = 0.0;
    double tilt_v_rad = 0.0;
    double tilt_n_rad = 0.0;
    // Gauge choices, not estimated variables.
    Point3 source_offset_mm = Point3::Zero();
    double offset_n_mm = 0.0;
};

struct CalibrationResult {
    std::string model = "spatial_seven_parameter";
    TrackingResult tracking;
    std::vector<DltCamera> cameras;
    SourceCircle source_circle;
    MachineGeometry machine;
    Eigen::Vector3d phantom_rotation_vector_rad = Eigen::Vector3d::Zero();
    Eigen::Vector3d phantom_translation_mm = Eigen::Vector3d::Zero();
    int rotation_direction = 1;
    double joint_rmse_px = std::numeric_limits<double>::infinity();
    double max_point_error_px = std::numeric_limits<double>::infinity();
    int joint_iterations = 0;
    bool joint_converged = false;
};

struct PipelineConfig {
    enum class PhantomKind { DoubleRing, Spiral, SingleRow };
    PhantomKind phantom_kind = PhantomKind::DoubleRing;
    int spiral_beads = 24;
    double spiral_radius_mm = 50, spiral_spacing_mm = 5, spiral_step_deg = 30;
    Point3 spiral_marker_mm{50, 0, -67.5};
    int row_beads = 12;
    double row_spacing_mm = 10;
    double initial_sid_mm = 440, initial_sdd_mm = 770;
    TrackerConfig tracker;
    PhantomSpec phantom;
    PixelSize pixel;
    int joint_max_iterations = 120;
    double robust_scale_px = 0.15;
};

}  // namespace cbct::calibration
