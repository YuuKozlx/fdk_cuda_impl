#include "cbct_calibration/io.hpp"
#include "cbct_calibration/phantom.hpp"
#include "cbct_calibration/pipeline.hpp"
#include "cbct_calibration/single_row.hpp"
#include "cbct_calibration/tracker.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace cbct::calibration;
namespace {
constexpr double pi = 3.141592653589793;
void require(bool ok, const char* what)
{
    if (!ok)
        throw std::runtime_error(what);
}
std::vector<Point2> observations(const std::vector<Point3>& local, const PipelineConfig& c, bool row = false, bool on_axis = false)
{
    const double sid = 440, sdd = 770, ou = 2.085, ov = 4.17, tn = .05235987756;
    Eigen::Matrix3d rotation = Eigen::AngleAxisd(.035, Point3::UnitX()).toRotationMatrix();
    Point3 shift(35, 28, 20);
    if (on_axis) {
        rotation.setIdentity();
        shift = Point3(0, 0, 20);
    }
    std::vector<Point2> points;
    for (int view = 0; view < c.tracker.views; ++view) {
        double a = 2 * pi * view / c.tracker.views - pi / 2;
        Point3 n(std::cos(a), std::sin(a), 0), radial = n, u(-std::sin(a), std::cos(a), 0), v = Point3::UnitZ();
        if (!row) {
            v = Eigen::AngleAxisd(.01745329252, u) * v;
            n = Eigen::AngleAxisd(.01745329252, u) * n;
            u = Eigen::AngleAxisd(.03490658504, v) * u;
            n = Eigen::AngleAxisd(.03490658504, v) * n;
        }
        u = Eigen::AngleAxisd(tn, n) * u;
        v = Eigen::AngleAxisd(tn, n) * v;
        Point3 source = sid * radial, detector = -(sdd - sid) * radial + ou * u + ov * v;
        for (auto p : local) {
            Point3 ray = rotation * p + shift - source;
            Point3 hit = source + ray * ((detector - source).dot(n) / ray.dot(n));
            points.emplace_back((c.tracker.image.cols - 1) * .5 + (hit - detector).dot(u) / c.pixel.u_mm,
                (c.tracker.image.rows - 1) * .5 + (hit - detector).dot(v) / c.pixel.v_mm);
        }
    }
    return points;
}
void rowTest()
{
    PipelineConfig c;
    c.tracker.views = 72;
    c.tracker.image = { 512, 512 };
    c.pixel = { .8, .8 };
    c.row_beads = 12;
    c.row_spacing_mm = 10;
    c.initial_sid_mm = 480;
    c.initial_sdd_mm = 820;
    c.joint_max_iterations = 400;
    std::vector<Point3> local;
    for (int j = 0; j < c.row_beads; ++j)
        local.emplace_back(0, 0, (j - 5.5) * 10);
    auto points = observations(local, c, true);
    std::vector<unsigned char> mask(points.size(), 1);
    auto fit = calibrateSingleRow(points, mask, c);
    std::cout << "single_row SID=" << fit.machine.sid_mm << " SDD=" << fit.machine.sdd_mm << " RMSE=" << fit.joint_rmse_px << std::endl;
    require(fit.joint_rmse_px < 1e-4 && std::abs(fit.machine.sid_mm - 440) < .1 && std::abs(fit.machine.sdd_mm - 770) < .1, "row exact geometry failed");
    for (size_t i = 0; i < points.size(); ++i) {
        points[i] += Point2(.01 * std::sin(i * 1.3), .01 * std::cos(i * .7));
        if (i % 29 == 0)
            mask[i] = 0;
    }
    fit = calibrateSingleRow(points, mask, c);
    require(fit.joint_rmse_px < .03 && std::abs(fit.machine.sid_mm - 440) < 2, "row noisy masked fit failed");
    points = observations(local, c, true, true);
    mask.assign(points.size(), 1);
    bool rejected = false;
    try {
        calibrateSingleRow(points, mask, c);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "on-axis unobservable row accepted");
}
void rawCase(const std::string& name, PipelineConfig c, const std::vector<Point3>& world, int ordinary)
{
    c.tracker.views = 360;
    c.tracker.image = { 512, 512 };
    c.pixel = { .8, .8 };
    c.joint_max_iterations = 240;
    c.tracker.threshold = 5;
    c.tracker.min_pixels = 4;
    const auto points = observations(world, c, c.phantom_kind == PipelineConfig::PhantomKind::SingleRow);
    auto dir = std::filesystem::current_path() / "phantom-family-fixtures";
    std::filesystem::create_directories(dir);
    c.tracker.raw_path = dir / (name + ".raw");
    {
        std::ofstream out(c.tracker.raw_path, std::ios::binary);
        for (int view = 0; view < c.tracker.views; ++view) {
            std::vector<float> frame(size_t(512) * 512, 0);
            for (size_t j = 0; j < world.size(); ++j) {
                auto p = points[view * world.size() + j];
                double sigma = int(j) < ordinary ? 1.7 : 2.9;
                for (int y = std::max(0, int(p.y() - 10)); y < std::min(512, int(p.y() + 11)); ++y)
                    for (int x = std::max(0, int(p.x() - 10)); x < std::min(512, int(p.x() + 11)); ++x) {
                        double r = (Point2(x, y) - p).squaredNorm();
                        frame[y * 512 + x] += float(18 * std::exp(-r / (2 * sigma * sigma)));
                    }
            }
            out.write(reinterpret_cast<char*>(frame.data()), frame.size() * sizeof(float));
        }
    }
    const auto fit = calibrateRaw(c);
    std::cout << name << " SID=" << fit.machine.sid_mm << " SDD=" << fit.machine.sdd_mm << " RMSE=" << fit.joint_rmse_px << std::endl;
    require(fit.joint_converged && fit.joint_rmse_px < 1.0 && std::abs(fit.machine.sid_mm - 440) < 4 && std::abs(fit.machine.sdd_mm - 770) < 6, "RAW family geometry failed");
    const auto report = dir / (name + ".json");
    writeCalibrationJson(report, fit);
    std::ifstream report_input(report);
    const std::string json((std::istreambuf_iterator<char>(report_input)), std::istreambuf_iterator<char>());
    require(json.find(fit.model) != std::string::npos && json.find(": nan") == std::string::npos && json.find("[nan") == std::string::npos, "invalid calibration report");
    if (c.phantom_kind == PipelineConfig::PhantomKind::SingleRow)
        require(json.find("\"source_circle\": null") != std::string::npos, "row report invents DLT source circle");
    report_input.close();
    std::filesystem::remove(report);
    std::filesystem::remove(c.tracker.raw_path);
}
}
int main(int argc, char** argv)
{
    try {
        rowTest();
        if (argc > 1 && std::string(argv[1]) == "row")
            return 0;
        PipelineConfig c;
        rawCase("asymmetric", c, makeIdentifiableDoubleRing(c.phantom), 24);
        c.phantom.beads_per_ring = 6;
        c.phantom.lower_radius_mm = 50;
        c.phantom.lower_phase_deg = 0;
        c.phantom.marker_points_mm = { { 50, 0, 80 }, { 50 * std::cos(pi / 6), 50 * std::sin(pi / 6), 70 } };
        rawCase("sync6", c, makeIdentifiableDoubleRing(c.phantom), 12);
        c.phantom.beads_per_ring = 12;
        c.phantom.marker_points_mm[1] = { 50 * std::cos(pi / 12), 50 * std::sin(pi / 12), 70 };
        rawCase("sync12", c, makeIdentifiableDoubleRing(c.phantom), 24);
        c.phantom.beads_per_ring = 6;
        c.phantom.marker_points_mm = { { 50, 0, 60 }, { 50, 0, -60 } };
        c.phantom.markers_for_numbering_only = true;
        rawCase("cylinder_marker_ids", c, makeIdentifiableDoubleRing(c.phantom), 12);
        c.phantom_kind = PipelineConfig::PhantomKind::Spiral;
        rawCase("spiral", c, makeSpiral(c), 24);
        c.phantom_kind = PipelineConfig::PhantomKind::SingleRow;
        std::vector<Point3> row;
        for (int j = 0; j < c.row_beads; ++j)
            row.emplace_back(0, 0, (j - 5.5) * 10);
        rawCase("single_row", c, row, 12);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
