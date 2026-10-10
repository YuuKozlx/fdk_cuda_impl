#include "cbct_calibration/dlt.hpp"
#include "cbct_calibration/optimizer.hpp"
#include "cbct_calibration/phantom.hpp"
#include "cbct_calibration/pipeline.hpp"
#include "cbct_calibration/source_circle.hpp"
#include "cbct_calibration/single_row.hpp"
#include "cbct_calibration/tracker.hpp"
#include "cbct_calibration/io.hpp"

#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace cbct::calibration;
namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<Point2> synthesize(const std::vector<Point3>& local, int views,
                               PixelSize pixel, ImageShape image) {
    const double sid = 440.0, sdd = 770.0;
    const double ou = 2.085, ov = 4.17;
    const double tu = 1.0 * kPi / 180.0, tv = 2.0 * kPi / 180.0, tn = 3.0 * kPi / 180.0;
    const Eigen::Matrix3d phantom_r =
        Eigen::AngleAxisd(2.0 * kPi / 180.0, Point3::UnitX()).toRotationMatrix();
    const Point3 phantom_t(10.0, 15.0, 20.0);
    std::vector<Point3> points;
    for (const auto& p : local) points.push_back(phantom_r * p + phantom_t);
    const Point2 center((image.cols - 1) * 0.5, (image.rows - 1) * 0.5);
    std::vector<Point2> output(static_cast<std::size_t>(views) * points.size());
    for (int i = 0; i < views; ++i) {
        const double a = 2.0 * kPi * i / views - kPi * 0.5;
        Point3 radial(std::cos(a), std::sin(a), 0.0), u(-std::sin(a), std::cos(a), 0.0);
        Point3 v = Point3::UnitZ(), n = radial;
        v = Eigen::AngleAxisd(tu, u) * v; n = Eigen::AngleAxisd(tu, u) * n;
        u = Eigen::AngleAxisd(tv, v) * u; n = Eigen::AngleAxisd(tv, v) * n;
        u = Eigen::AngleAxisd(tn, n) * u; v = Eigen::AngleAxisd(tn, n) * v;
        const Point3 source = sid * radial;
        const Point3 detector = -(sdd - sid) * radial + ou * u + ov * v;
        const double plane = (detector - source).dot(n);
        for (std::size_t j = 0; j < points.size(); ++j) {
            const Point3 ray = points[j] - source;
            const Point3 hit = source + plane / ray.dot(n) * ray;
            const Point3 relative = hit - detector;
            output[static_cast<std::size_t>(i) * points.size() + j] =
                Point2(center.x() + relative.dot(u) / pixel.u_mm,
                       center.y() + relative.dot(v) / pixel.v_mm);
        }
    }
    return output;
}

bool testMaskedPhantom(const char* name, const std::vector<Point3>& world) {
    const int views = 24;
    const PixelSize pixel{};
    auto corrupted = synthesize(world, views, pixel, ImageShape{});
    std::vector<unsigned char> usable(corrupted.size(), 1);
    for (int view = 0; view < views; ++view) {
        const std::size_t first = static_cast<std::size_t>(
            view * static_cast<int>(world.size()) + view % static_cast<int>(world.size()));
        const std::size_t second = static_cast<std::size_t>(
            view * static_cast<int>(world.size()) +
            (5 * view + 3) % static_cast<int>(world.size()));
        corrupted[first] += Point2(90.0, -70.0);
        corrupted[second] += Point2(-55.0, 85.0);
        usable[first] = 0;
        usable[second] = 0;
    }

    PipelineConfig config;
    config.tracker.views = views;
    config.tracker.image = ImageShape{};
    config.pixel = pixel;
    config.joint_max_iterations = 240;
    const auto result = calibrateIndexedPoints(
        corrupted, world, usable, views, config);

    bool dlt_ok = result.cameras.size() == static_cast<std::size_t>(views);
    for (const auto& camera : result.cameras)
        dlt_ok = dlt_ok && camera.reprojection_rmse_px < 1.0e-5;
    const bool joint_ok = result.joint_rmse_px < 0.05 &&
        std::abs(result.machine.sid_mm - 440.0) < 2.0 &&
        std::abs(result.machine.sdd_mm - 770.0) < 2.0 &&
        std::abs(result.machine.offset_u_mm - 2.085) < 0.5 &&
        std::abs(result.machine.offset_v_mm - 4.17) < 0.5;
    std::cout << name << " masked regression: DLT="
              << (dlt_ok ? "ok" : "failed")
              << " joint RMSE=" << result.joint_rmse_px << " px"
              << " SDD=" << result.machine.sdd_mm << " mm\n";
    return dlt_ok && joint_ok;
}

bool testMergedTracksRecover() {
    const int views = 72;
    const ImageShape shape{512, 512};
    const PixelSize pixel{0.834, 0.834};
    const PhantomSpec phantom{};
    const auto world = makeIdentifiableDoubleRing(phantom);
    const auto projected = synthesize(world, views, pixel, shape);
    struct TemporaryRaw {
        std::filesystem::path path;
        ~TemporaryRaw() { std::error_code error; std::filesystem::remove(path, error); }
    } temporary{std::filesystem::temp_directory_path() /
        ("cbct-merged-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".raw")};
    {
        std::ofstream output(temporary.path, std::ios::binary);
        for (int view = 0; view < views; ++view) {
            std::vector<float> frame(static_cast<std::size_t>(shape.rows * shape.cols), 0.0F);
            for (std::size_t target = 0; target < world.size(); ++target) {
                const auto center = projected[static_cast<std::size_t>(view) * world.size() + target];
                const double radius = target + 1 == world.size() ? 5.0 : 3.0;
                for (int y = std::max(0, static_cast<int>(center.y() - radius));
                     y <= std::min(shape.rows - 1, static_cast<int>(center.y() + radius)); ++y)
                    for (int x = std::max(0, static_cast<int>(center.x() - radius));
                         x <= std::min(shape.cols - 1, static_cast<int>(center.x() + radius)); ++x)
                        if ((Point2(x, y) - center).squaredNorm() <= radius * radius)
                            frame[static_cast<std::size_t>(y * shape.cols + x)] = 20.0F;
            }
            output.write(reinterpret_cast<const char*>(frame.data()),
                         static_cast<std::streamsize>(frame.size() * sizeof(float)));
        }
        if (!output) throw std::runtime_error("cannot write synthetic RAW test");
    }
    RawStackReader raw(temporary.path, views, shape);
    TrackerConfig config; config.views = views; config.image = shape;
    const auto tracks = trackIdentifiablePhantom(raw, config, phantom, pixel);
    int merged = 0, recovered = 0;
    for (int view = 0; view < views; ++view) for (int target = 0; target < tracks.targets; ++target) {
        const auto index = static_cast<std::size_t>(view * tracks.targets + target);
        if (tracks.isUsable(view, target) != (tracks.isObserved(view, target) && !tracks.merged[index]))
            return false;
        if (!tracks.merged[index]) continue;
        ++merged;
        if (view + 1 < views && tracks.isUsable(view + 1, target)) ++recovered;
    }
    std::cout << "C++ synthetic RAW tracking: merged=" << merged
              << " recovered=" << recovered << "\n";
    return merged > 0 && recovered > 0;
}
}

int main() {
    const auto row = makeSingleRow(5, 10.0);
    assert(row.size() == 5);
    assert(std::abs(row.front().x() + 20.0) < 1.0e-12);
    assert(std::abs(row.back().x() - 20.0) < 1.0e-12);
    assert(makeChoDoubleRing(12).size() == 24);
    assert(makeYangDoubleRing(6).size() == 12);
    PhantomSpec no_marker;
    no_marker.marker_points_mm.clear();
    assert(makeIdentifiableDoubleRing(no_marker).size() == 24);
    PhantomSpec two_markers;
    two_markers.marker_points_mm.push_back({48.296, 12.941, 70.0});
    assert(makeIdentifiableDoubleRing(two_markers).size() == 26);
    const PhantomSpec phantom{};
    const PixelSize pixel{};
    const auto world = makeIdentifiableDoubleRing(phantom);
    std::vector<Point3> camera_world{{0.0, 0.0, 0.0}, {20.0, 0.0, 10.0},
                                     {-10.0, 30.0, 50.0}, {5.0, -20.0, -30.0},
                                     {40.0, 20.0, -10.0}, {-30.0, -25.0, 20.0}};
    ProjectionMatrix known;
    known << 1200.0, 0.0, 512.0, 0.0,
             0.0, 1200.0, 500.0, 0.0,
             0.0, 0.0, 1.0, 1.0;
    std::vector<Point2> image;
    for (const auto& p : camera_world) image.push_back(projectPoint(known, p));
    const auto recovered = normalizedDlt(camera_world, image);
    assert(reprojectionRmse(recovered, camera_world, image) < 1.0e-8);

    const int views = 36;
    const auto indexed = synthesize(world, views, pixel, ImageShape{});
    PipelineConfig config;
    config.tracker.views = views;
    config.tracker.image = ImageShape{};
    config.pixel = pixel;
    config.joint_max_iterations = 240;
    const auto result = calibrateIndexed(indexed, views, config);
    auto corrupted = indexed;
    std::vector<unsigned char> usable(corrupted.size(), 1);
    for (int view = 0; view < views; ++view) {
        const std::size_t index = static_cast<std::size_t>(view) * world.size();
        corrupted[index] += Point2(80.0, -60.0);
        usable[index] = 0;
    }
    const auto masked = calibrateIndexedPoints(corrupted, world, usable, views, config);
    const bool passed =
        result.cameras.size() == static_cast<std::size_t>(views) &&
        result.source_circle.radius_mm > 400.0 && result.source_circle.radius_mm < 480.0 &&
        result.joint_rmse_px < 0.05 &&
        std::abs(result.machine.sdd_mm - 770.0) < 2.0 &&
        std::abs(result.machine.offset_u_mm - 2.085) < 0.5 &&
        std::abs(result.machine.offset_v_mm - 4.17) < 0.5 &&
        masked.joint_rmse_px < 0.05 &&
        std::abs(masked.machine.sdd_mm - result.machine.sdd_mm) < 0.1;
    std::cout << "CBCT calibration C++ pipeline result: RMSE="
              << result.joint_rmse_px << " px; SID=" << result.machine.sid_mm
              << " SDD=" << result.machine.sdd_mm
              << " offsetU=" << result.machine.offset_u_mm
              << " offsetV=" << result.machine.offset_v_mm
              << " circle=" << result.source_circle.radius_mm << "\n";
    const bool cho_masked = testMaskedPhantom("Cho 12+12", makeChoDoubleRing(12));
    const bool yang_masked = testMaskedPhantom("Yang 6+6", makeYangDoubleRing(6));
    if (!passed || !cho_masked || !yang_masked || !testMergedTracksRecover()) {
        std::cerr << "C++ calibration regression failed\n";
        return 1;
    }
    std::cout << "CBCT calibration C++ pipeline test passed\n";
    return 0;
}
