#include "cbct_calibration/dlt.hpp"
#include "cbct_calibration/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
namespace cbct::calibration {
std::vector<Point3> makeSpiral(const PipelineConfig& c)
{
    if (c.spiral_beads < 6 || !std::isfinite(c.spiral_radius_mm) || c.spiral_radius_mm <= 0 || !std::isfinite(c.spiral_spacing_mm) || c.spiral_spacing_mm <= 0 || !std::isfinite(c.spiral_step_deg) || !c.spiral_marker_mm.allFinite())
        throw std::invalid_argument("Invalid spiral dimensions");
    if (std::abs(std::sin(c.spiral_step_deg * 3.141592653589793 / 180)) < 1e-8)
        throw std::invalid_argument("Spiral angular step must produce non-coplanar bead positions");
    std::vector<Point3> points;
    for (int i = 0; i < c.spiral_beads; ++i) {
        double a = i * c.spiral_step_deg * 3.141592653589793 / 180;
        points.emplace_back(c.spiral_radius_mm * std::cos(a), c.spiral_radius_mm * std::sin(a),
            (i - (c.spiral_beads - 1) * .5) * c.spiral_spacing_mm);
    }
    points.push_back(c.spiral_marker_mm);
    return points;
}
TrackingResult trackSpiral(RawStackReader& raw, const PipelineConfig& c)
{
    const auto world = makeSpiral(c);
    std::vector<float> image;
    // Numbering uses only measured centroids, marker size and known phantom dimensions.
    // Enumerate projected axial orders; no source, detector or phantom-pose truth is used.
    for (int view = 0; view < raw.views(); ++view) {
        raw.readFrame(view, image);
        auto components = detectComponents(image, c.tracker.image, c.tracker.threshold, c.tracker.min_pixels);
        if (components.size() != world.size())
            continue;
        std::vector<int> markers(components.size());
        std::iota(markers.begin(), markers.end(), 0);
        std::sort(markers.begin(), markers.end(), [&](int a, int b) { return components[a].area > components[b].area; });
        if (double(components[markers[0]].area) / std::max(1, components[markers[markers.size() / 2]].area) < c.tracker.marker_area_ratio)
            continue;
        struct Candidate {
            double error;
            std::vector<Point2> points;
        };
        std::vector<Candidate> candidates;
        std::set<std::vector<int>> seen;
        for (int mi = 0; mi < std::min(4, int(markers.size())); ++mi) {
            int marker = markers[mi];
            for (int degree = 0; degree < 180; degree += 2)
                for (int reverse = 0; reverse < 2; ++reverse) {
                    double a = degree * 3.141592653589793 / 180;
                    Point2 axis(std::cos(a), std::sin(a));
                    if (reverse)
                        axis = -axis;
                    std::vector<int> order;
                    for (int i = 0; i < int(components.size()); ++i)
                        if (i != marker)
                            order.push_back(i);
                    std::sort(order.begin(), order.end(), [&](int x, int y) { return components[x].centroid.dot(axis) < components[y].centroid.dot(axis); });
                    order.push_back(marker);
                    if (!seen.insert(order).second)
                        continue;
                    std::vector<Point2> points;
                    for (int id : order)
                        points.push_back(components[id].centroid);
                    try {
                        auto camera = normalizedDlt(world, points);
                        double e = reprojectionRmse(camera, world, points);
                        if (std::isfinite(e))
                            candidates.push_back({ e, std::move(points) });
                    } catch (const std::exception&) {
                    }
                }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.error < b.error; });
        double best = c.tracker.max_dlt_rmse_px;
        std::vector<Point2> anchor;
        for (size_t i = 0; i < std::min<size_t>(48, candidates.size()); ++i) {
            try {
                auto camera = calibratePhysicalCamera(world, candidates[i].points, c.pixel, 300);
                if (camera.reprojection_rmse_px < best) {
                    best = camera.reprojection_rmse_px;
                    anchor = candidates[i].points;
                }
            } catch (const std::exception&) {
            }
        }
        if (!anchor.empty())
            return trackFromAnchor(raw, c.tracker, view, anchor, world);
    }
    throw std::runtime_error("No separated, physically consistent marked-spiral anchor; check threshold, panel coverage and phantom dimensions");
}
}
