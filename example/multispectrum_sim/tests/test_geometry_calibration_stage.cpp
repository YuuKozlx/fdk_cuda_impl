#include "calibration/GeometryCalibrationStage.hpp"
#include "cbct_calibration/phantom.hpp"
#include <Eigen/Geometry>
#include <iostream>
using namespace cbct::calibration;
namespace {
constexpr double kPi = 3.14159265358979323846;
std::vector<Point2> synthesize(const std::vector<Point3>& local, int views,
    PixelSize pixel, ImageShape image)
{
    const double sid = 440.0, sdd = 770.0;
    const double ou = 2.085, ov = 4.17;
    const double tu = 1.0 * kPi / 180.0, tv = 2.0 * kPi / 180.0, tn = 3.0 * kPi / 180.0;
    const Eigen::Matrix3d phantom_r = Eigen::AngleAxisd(2.0 * kPi / 180.0, Point3::UnitX()).toRotationMatrix();
    const Point3 phantom_t(10.0, 15.0, 20.0);
    std::vector<Point3> points;
    for (const auto& p : local)
        points.push_back(phantom_r * p + phantom_t);
    const Point2 center((image.cols - 1) * 0.5, (image.rows - 1) * 0.5);
    std::vector<Point2> output(static_cast<std::size_t>(views) * points.size());
    for (int i = 0; i < views; ++i) {
        const double a = 2.0 * kPi * i / views - kPi * 0.5;
        Point3 radial(std::cos(a), std::sin(a), 0.0), u(-std::sin(a), std::cos(a), 0.0);
        Point3 v = Point3::UnitZ(), n = radial;
        v = Eigen::AngleAxisd(tu, u) * v;
        n = Eigen::AngleAxisd(tu, u) * n;
        u = Eigen::AngleAxisd(tv, v) * u;
        n = Eigen::AngleAxisd(tv, v) * n;
        u = Eigen::AngleAxisd(tn, n) * u;
        v = Eigen::AngleAxisd(tn, n) * v;
        const Point3 source = sid * radial;
        const Point3 detector = -(sdd - sid) * radial + ou * u + ov * v;
        const double plane = (detector - source).dot(n);
        for (std::size_t j = 0; j < points.size(); ++j) {
            const Point3 ray = points[j] - source;
            const Point3 hit = source + plane / ray.dot(n) * ray;
            const Point3 relative = hit - detector;
            output[static_cast<std::size_t>(i) * points.size() + j] = Point2(center.x() + relative.dot(u) / pixel.u_mm,
                center.y() + relative.dot(v) / pixel.v_mm);
        }
    }
    return output;
}

}
int main()
{
    try {
        using namespace yk::spectral;
        calibration::Context c;
        c.input = calibration::InputKind::IndexedPoints;
        c.config.tracker.views = 24;
        c.config.joint_max_iterations = 240;
        calibration::Data d;
        d.landmarks = makeIdentifiableDoubleRing(c.config.phantom);
        d.points = synthesize(d.landmarks, 24, c.config.pixel, c.config.tracker.image);
        calibration::GeometryCalibrationStage stage;
        pipeline::validatePluginInfo(stage.info());
        stage.process(d, c);
        auto require = [](bool ok, const char* message) {if(!ok)throw std::runtime_error(message); };
        require(d.result && d.result->joint_rmse_px < .05, "fit residual");
        require(std::abs(d.result->machine.sdd_mm - 770) < 2, "fitted SDD");
        require(d.domain == calibration::Domain::Measurements, "stage must not modify domain");
        d.domain = stage.contract().output_domain;
        c.nominal.kind = GeometryKind::FlatCbct;
        c.nominal.parameters.views = 24;
        c.nominal.parameters.source_offset_y_mm = 10;
        calibration::GeometryCorrectionStage correction;
        correction.process(d, c);
        require(d.corrected && d.corrected->parameters.source_offset_y_mm == 0, "gauge-fixed geometry");
        require(c.nominal.parameters.source_offset_y_mm == 10, "input mutation");
        require(std::abs(d.corrected->parameters.offset_u_mm - 2.085) < .5, "correction offset");
        d.result->joint_converged = false;
        bool rejected = false;
        try {
            correction.process(d, c);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected && !d.corrected, "reject unconverged correction and clear stale result");
        c.full_circle = false;
        rejected = false;
        try {
            stage.process(d, c);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && !d.result, "reject unsupported trajectory and clear stale fit");
        c.full_circle = true;
        c.input = calibration::InputKind::ProjectionRaw;
        c.config.tracker.raw_path = "validation-only.raw";
        using Kind = PipelineConfig::PhantomKind;
        for (auto kind : { Kind::DoubleRing, Kind::Spiral, Kind::SingleRow }) {
            c.config.phantom_kind = kind;
            stage.validateContext(c);
        }
        auto rejectsContext = [&]() {
            try {
                stage.validateContext(c);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        c.config.initial_sdd_mm = c.config.initial_sid_mm;
        require(rejectsContext(), "row accepted invalid initial distances");
        c.config.initial_sdd_mm = 770;
        c.config.phantom_kind = Kind::Spiral;
        c.config.spiral_step_deg = 180;
        require(rejectsContext(), "accepted coplanar spiral");
        c.config.spiral_step_deg = 30;
        c.config.tracker.marker_area_ratio = std::numeric_limits<double>::quiet_NaN();
        require(rejectsContext(), "spiral accepted nonfinite tracking settings");
        std::cout << "geometry calibration/correction stage passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
