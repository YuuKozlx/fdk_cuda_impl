#include "cbct_calibration/single_row.hpp"

#include "cbct_calibration/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cbct::calibration {
namespace {

std::vector<Point2> sortedCentres(const std::vector<Component>& components) {
    std::vector<Point2> points;
    points.reserve(components.size());
    for (const auto& item : components) points.push_back(item.centroid);
    std::sort(points.begin(), points.end(), [](const Point2& a, const Point2& b) {
        return a.y() == b.y() ? a.x() < b.x() : a.y() < b.y();
    });
    return points;
}

}  // namespace

SingleRowTrackingResult trackSingleRow(RawStackReader& raw,
                                       const SingleRowConfig& config) {
    std::vector<float> image;
    raw.readFrame(0, image);
    auto first_components = detectComponents(image, raw.shape(), config.threshold,
                                             config.min_pixels);
    auto first = sortedCentres(first_components);
    if (config.expected_beads > 0) {
        if (static_cast<int>(first.size()) < config.expected_beads)
            throw std::runtime_error("single-row first frame has too few beads");
        first.resize(static_cast<std::size_t>(config.expected_beads));
    }
    if (first.size() < 3) throw std::runtime_error("single-row needs at least three beads");

    SingleRowTrackingResult result;
    result.views = raw.views();
    result.targets = static_cast<int>(first.size());
    const std::size_t total = static_cast<std::size_t>(result.views * result.targets);
    result.points.assign(total, Point2::Constant(std::numeric_limits<double>::quiet_NaN()));
    result.observed.assign(total, 0);
    result.permanently_disabled.assign(static_cast<std::size_t>(result.targets), 0);
    std::vector<Point2> previous = first;
    for (int target = 0; target < result.targets; ++target) {
        result.points[static_cast<std::size_t>(target)] = first[static_cast<std::size_t>(target)];
        result.observed[static_cast<std::size_t>(target)] = 1;
    }

    for (int view = 1; view < result.views; ++view) {
        raw.readFrame(view, image);
        const auto components = detectComponents(image, raw.shape(), config.threshold,
                                                 config.min_pixels);
        std::vector<unsigned char> used(components.size(), 0);
        std::vector<Point2> next = previous;
        std::vector<int> assignment(static_cast<std::size_t>(result.targets), -1);
        std::vector<std::pair<double, std::pair<int, int>>> costs;
        for (int target = 0; target < result.targets; ++target) {
            if (result.permanently_disabled[static_cast<std::size_t>(target)]) continue;
            for (int component = 0; component < static_cast<int>(components.size()); ++component)
                costs.push_back({(components[static_cast<std::size_t>(component)].centroid -
                                  previous[static_cast<std::size_t>(target)]).norm(),
                                 {target, component}});
        }
        std::sort(costs.begin(), costs.end());
        for (const auto& item : costs) {
            const int target = item.second.first, component = item.second.second;
            if (item.first > config.max_jump_px) break;
            if (assignment[static_cast<std::size_t>(target)] >= 0 ||
                used[static_cast<std::size_t>(component)]) continue;
            assignment[static_cast<std::size_t>(target)] = component;
            used[static_cast<std::size_t>(component)] = 1;
        }
        for (int target = 0; target < result.targets; ++target) {
            if (result.permanently_disabled[static_cast<std::size_t>(target)]) continue;
            const int best = assignment[static_cast<std::size_t>(target)];
            const bool in_panel = previous[static_cast<std::size_t>(target)].x() >= 0.0 &&
                                  previous[static_cast<std::size_t>(target)].x() < raw.shape().cols &&
                                  previous[static_cast<std::size_t>(target)].y() >= 0.0 &&
                                  previous[static_cast<std::size_t>(target)].y() < raw.shape().rows;
            if (best < 0 || !in_panel) {
                result.permanently_disabled[static_cast<std::size_t>(target)] = 1;
                continue;
            }
            next[static_cast<std::size_t>(target)] = components[static_cast<std::size_t>(best)].centroid;
            result.points[static_cast<std::size_t>(view * result.targets + target)] =
                next[static_cast<std::size_t>(target)];
            result.observed[static_cast<std::size_t>(view * result.targets + target)] = 1;
        }
        previous = next;
    }
    return result;
}

}  // namespace cbct::calibration
