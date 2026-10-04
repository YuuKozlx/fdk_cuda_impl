#include "cbct_calibration/pipeline.hpp"

#include "cbct_calibration/dlt.hpp"
#include "cbct_calibration/io.hpp"
#include "cbct_calibration/optimizer.hpp"
#include "cbct_calibration/phantom.hpp"
#include "cbct_calibration/source_circle.hpp"
#include "cbct_calibration/tracker.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cbct::calibration {
namespace {
constexpr double kPi = 3.14159265358979323846;

Point3 rotateAround(const Point3& value, const Point3& axis, double angle) {
    return Eigen::AngleAxisd(angle, axis.normalized()) * value;
}

std::vector<Point2> projectJoint(const Eigen::VectorXd& state,
                                 const std::vector<Point3>& local,
                                 int views, int direction, PixelSize pixel,
                                 ImageShape shape) {
    const double sid = state(0), sdd = state(1), ou = state(2), ov = state(3);
    const double tu = state(4), tv = state(5), tn = state(6);
    const Eigen::Matrix3d phantom_rotation = rotationFromVector(state.segment<3>(7));
    const Point3 phantom_translation = state.segment<3>(10);
    std::vector<Point3> world; world.reserve(local.size());
    for (const auto& p : local) world.push_back(phantom_rotation * p + phantom_translation);
    std::vector<Point2> projected(static_cast<std::size_t>(views) * local.size());
    const Point2 center((shape.cols - 1) * 0.5, (shape.rows - 1) * 0.5);
    for (int view = 0; view < views; ++view) {
        const double gantry = direction * 2.0 * kPi * view / views;
        const double angle = gantry - kPi * 0.5;
        Point3 radial(std::cos(angle), std::sin(angle), 0.0);
        Point3 u(-std::sin(angle), std::cos(angle), 0.0);
        Point3 v = Point3::UnitZ();
        Point3 n = radial;
        v = rotateAround(v, u, tu); n = rotateAround(n, u, tu);
        u = rotateAround(u, v, tv); n = rotateAround(n, v, tv);
        u = rotateAround(u, n, tn); v = rotateAround(v, n, tn);
        const Point3 source = sid * radial;
        const Point3 detector_center = -(sdd - sid) * radial + ou * u + ov * v;
        const double plane = (detector_center - source).dot(n);
        for (std::size_t target = 0; target < world.size(); ++target) {
            const Point3 ray = world[target] - source;
            const double denominator = ray.dot(n);
            const double scale = plane / denominator;
            const Point3 detector_point = source + scale * ray;
            const Point3 relative = detector_point - detector_center;
            projected[static_cast<std::size_t>(view) * local.size() + target] =
                Point2(center.x() + relative.dot(u) / pixel.u_mm,
                       center.y() + relative.dot(v) / pixel.v_mm);
        }
    }
    return projected;
}

struct JointCandidate {
    Eigen::VectorXd state;
    int direction = 1;
    double rmse = std::numeric_limits<double>::infinity();
    double max_error = std::numeric_limits<double>::infinity();
    int iterations = 0;
    bool converged = false;
};

JointCandidate fitJoint(const std::vector<Point2>& measured,
                        const std::vector<Point3>& local,
                        const std::vector<DltCamera>& cameras,
                        const SourceCircle& circle,
                        int views, const PipelineConfig& config) {
    Eigen::Matrix3d rotation0; Point3 translation0;
    scannerAlignment(circle, rotation0, translation0);
    std::vector<double> sdds; std::vector<double> pu, pv;
    for (const auto& camera : cameras) {
        if (std::isfinite(camera.sdd_mm) && camera.sdd_mm > 0.0) sdds.push_back(camera.sdd_mm);
        pu.push_back(camera.principal_point_px.x()); pv.push_back(camera.principal_point_px.y());
    }
    auto median = [](std::vector<double> values) {
        if (values.empty()) return 0.0;
        const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
        std::nth_element(values.begin(), middle, values.end()); return *middle;
    };
    const Point2 image_center((config.tracker.image.cols - 1) * 0.5,
                              (config.tracker.image.rows - 1) * 0.5);
    double sdd0 = median(sdds);
    if (!std::isfinite(sdd0) || sdd0 < circle.radius_mm || sdd0 > 10.0 * circle.radius_mm) sdd0 = 1.75 * circle.radius_mm;
    // The fitted source circle determines the scanner axis, but its in-plane
    // basis still has an arbitrary phase.  Resolve that phase from the first
    // indexed frame before starting the local joint optimizer.  This matters
    // especially for sparse six-bead rings, where LM otherwise has too little
    // local leverage to leave the wrong phase basin.
    double best_phase_score = std::numeric_limits<double>::infinity();
    Eigen::Matrix3d best_rotation = rotation0;
    Point3 best_translation = translation0;
    for (int phase_index = 0; phase_index < 72; ++phase_index) {
        const double phase = -kPi + 2.0 * kPi * phase_index / 72.0;
        const Eigen::Matrix3d phase_rotation =
            Eigen::AngleAxisd(phase, Point3::UnitZ()).toRotationMatrix();
        const Eigen::Matrix3d trial_rotation = phase_rotation * rotation0;
        const Point3 trial_translation = -trial_rotation * circle.center_phantom_mm;
        Eigen::VectorXd trial(13);
        trial << circle.radius_mm, sdd0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            rotationToVector(trial_rotation), trial_translation;
        const auto prediction = projectJoint(trial, local, 1, 1, config.pixel,
                                             config.tracker.image);
        double squared = 0.0;
        for (std::size_t target = 0; target < local.size(); ++target) {
            squared += (prediction[target] - measured[target]).squaredNorm();
        }
        const double score = std::sqrt(squared / local.size());
        if (score < best_phase_score) {
            best_phase_score = score;
            best_rotation = trial_rotation;
            best_translation = trial_translation;
        }
    }
    Eigen::VectorXd state0(13);
    // The projective decomposition's principal point is only an initializer.
    // Starting offsets at the panel center avoids amplifying an RQ sign branch;
    // the joint residual then estimates the equivalent detector offsets.
    state0 << circle.radius_mm, sdd0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        rotationToVector(best_rotation), best_translation;
    // SID is fixed to the fitted source-circle radius. The remaining 12 values are free.
    Eigen::VectorXd free0 = state0.tail(12);
    Eigen::VectorXd scales(12);
    scales << 100.0, 2.0, 2.0, 0.02, 0.02, 0.02,
              0.03, 0.03, 0.03, 10.0, 10.0, 10.0;
    JointCandidate best;
    for (const int direction : {1, -1}) {
        auto residual = [&](const Eigen::VectorXd& free) {
            Eigen::VectorXd state = state0; state.tail(12) = free;
            const auto prediction = projectJoint(state, local, views, direction,
                                                 config.pixel, config.tracker.image);
            Eigen::VectorXd r(2 * static_cast<Eigen::Index>(measured.size()));
            for (std::size_t i = 0; i < measured.size(); ++i) {
                const Point2 e = prediction[i] - measured[i];
                r(2 * static_cast<Eigen::Index>(i)) = e.x();
                r(2 * static_cast<Eigen::Index>(i) + 1) = e.y();
            }
            return r;
        };
        OptimizerOptions options; options.max_iterations = config.joint_max_iterations;
        options.robust_scale = config.robust_scale_px;
        const auto optimized = levenbergMarquardt(residual, free0, scales, options);
        Eigen::VectorXd state = state0; state.tail(12) = optimized.parameters;
        const auto prediction = projectJoint(state, local, views, direction,
                                             config.pixel, config.tracker.image);
        double squared = 0.0, maximum = 0.0;
        for (std::size_t i = 0; i < measured.size(); ++i) {
            const double error = (prediction[i] - measured[i]).norm(); squared += error * error; maximum = std::max(maximum, error);
        }
        const double rmse = std::sqrt(squared / measured.size());
        if (rmse < best.rmse) best = {state, direction, rmse, maximum,
                                      optimized.iterations, optimized.converged};
    }
    return best;
}

}  // namespace

CalibrationResult calibrateIndexedPoints(const std::vector<Point2>& indexed_points,
                                         const std::vector<Point3>& local,
                                         int views, const PipelineConfig& config) {
    const int targets = static_cast<int>(local.size());
    if (views < 3 || indexed_points.size() != static_cast<std::size_t>(views * targets))
        throw std::invalid_argument("indexed point array has an invalid size");
    CalibrationResult result;
    result.cameras.reserve(static_cast<std::size_t>(views));
    std::vector<Point3> sources; sources.reserve(static_cast<std::size_t>(views));
    for (int view = 0; view < views; ++view) {
        std::vector<Point2> frame(indexed_points.begin() + view * targets,
                                  indexed_points.begin() + (view + 1) * targets);
        auto camera = calibratePhysicalCamera(local, frame, config.pixel);
        sources.push_back(camera.source_phantom_mm); result.cameras.push_back(std::move(camera));
    }
    result.source_circle = fitSourceCircle(sources);
    const JointCandidate joint = fitJoint(indexed_points, local, result.cameras,
                                          result.source_circle, views, config);
    result.machine.sid_mm = joint.state(0); result.machine.sdd_mm = joint.state(1);
    result.machine.offset_u_mm = joint.state(2); result.machine.offset_v_mm = joint.state(3);
    result.machine.tilt_u_rad = joint.state(4); result.machine.tilt_v_rad = joint.state(5);
    result.machine.tilt_n_rad = joint.state(6);
    result.phantom_rotation_vector_rad = joint.state.segment<3>(7);
    result.phantom_translation_mm = joint.state.segment<3>(10);
    result.rotation_direction = joint.direction; result.joint_rmse_px = joint.rmse;
    result.max_point_error_px = joint.max_error; result.joint_iterations = joint.iterations;
    result.joint_converged = joint.converged;
    return result;
}

CalibrationResult calibrateIndexed(const std::vector<Point2>& indexed_points,
                                   int views, const PipelineConfig& config) {
    return calibrateIndexedPoints(indexed_points,
                                  makeIdentifiableDoubleRing(config.phantom),
                                  views, config);
}

CalibrationResult calibrateRaw(const PipelineConfig& config) {
    RawStackReader raw(config.tracker.raw_path, config.tracker.views,
                       config.tracker.image);
    TrackingResult tracking = trackIdentifiablePhantom(raw, config.tracker,
                                                       config.phantom, config.pixel);
    const auto local = makeIdentifiableDoubleRing(config.phantom);
    for (int view = 0; view < tracking.views; ++view) {
        std::vector<Point3> world;
        std::vector<Point2> image;
        for (int target = 0; target < tracking.targets; ++target) {
            if (!tracking.isObserved(view, target)) continue;
            world.push_back(local[static_cast<std::size_t>(target)]);
            image.push_back(tracking.at(view, target));
        }
        if (world.size() >= 6) {
            const auto projection = normalizedDlt(world, image);
            tracking.dlt_rmse_px[static_cast<std::size_t>(view)] =
                reprojectionRmse(projection, world, image);
        }
    }
    for (int view = 0; view < tracking.views; ++view)
        for (int target = 0; target < tracking.targets; ++target)
            if (!tracking.isObserved(view, target))
                throw std::runtime_error("dense joint fit requires every target in every frame");
    // A synchronized circular ring has two mirror-related numberings with
    // essentially the same single-view DLT residual.  Close both candidates
    // against the shared circular scanner model and retain the globally
    // consistent hand.  External markers fix phase/ring identity, but a
    // single frame alone still does not reliably fix winding in noisy data.
    std::vector<Point2> reversed = tracking.points;
    const int n = config.phantom.beads_per_ring;
    const int lower_mirror_shift = static_cast<int>(std::llround(
        config.phantom.lower_phase_deg * n / 180.0));
    for (int view = 0; view < tracking.views; ++view) {
        for (int bead = 0; bead < n; ++bead) {
            const int upper_source_bead = (n - bead) % n;
            const int lower_source_bead =
                (2 * n - bead - lower_mirror_shift) % n;
            reversed[static_cast<std::size_t>(view * tracking.targets + 2 * bead)] =
                tracking.points[static_cast<std::size_t>(view * tracking.targets +
                                                         2 * upper_source_bead)];
            reversed[static_cast<std::size_t>(view * tracking.targets + 2 * bead + 1)] =
                tracking.points[static_cast<std::size_t>(view * tracking.targets +
                                                         2 * lower_source_bead + 1)];
        }
    }
    CalibrationResult direct = calibrateIndexed(tracking.points, tracking.views, config);
    CalibrationResult mirrored = calibrateIndexed(reversed, tracking.views, config);
    const bool use_mirrored = mirrored.joint_rmse_px < direct.joint_rmse_px;
    CalibrationResult result = !use_mirrored
        ? std::move(direct) : std::move(mirrored);
    if (use_mirrored) tracking.points = std::move(reversed);
    result.tracking = std::move(tracking);
    return result;
}

}  // namespace cbct::calibration
