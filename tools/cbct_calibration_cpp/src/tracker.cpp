#include "cbct_calibration/tracker.hpp"

#include "cbct_calibration/dlt.hpp"
#include "cbct_calibration/phantom.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cbct::calibration {
namespace {

std::vector<Component> connectedComponents(const std::vector<float>& image,
                                           ImageShape shape, float threshold,
                                           int min_pixels) {
    const int count = shape.rows * shape.cols;
    std::vector<unsigned char> mask(static_cast<std::size_t>(count), 0);
    for (int i = 0; i < count; ++i) mask[static_cast<std::size_t>(i)] = image[static_cast<std::size_t>(i)] > threshold;
    std::vector<Component> components;
    std::vector<int> queue;
    queue.reserve(4096);
    for (int y = 0; y < shape.rows; ++y) for (int x = 0; x < shape.cols; ++x) {
        const int seed = y * shape.cols + x;
        if (!mask[static_cast<std::size_t>(seed)]) continue;
        mask[static_cast<std::size_t>(seed)] = 0;
        queue.clear(); queue.push_back(seed);
        Point2 weighted = Point2::Zero(); double weight_sum = 0.0; float peak = 0.0F;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const int id = queue[head]; const int py = id / shape.cols; const int px = id % shape.cols;
            const double w = std::max(1.0e-6, static_cast<double>(image[static_cast<std::size_t>(id)] - threshold));
            weighted += w * Point2(static_cast<double>(px), static_cast<double>(py)); weight_sum += w;
            peak = std::max(peak, image[static_cast<std::size_t>(id)]);
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int nx = px + dx, ny = py + dy;
                if (nx < 0 || nx >= shape.cols || ny < 0 || ny >= shape.rows) continue;
                const int ni = ny * shape.cols + nx;
                if (mask[static_cast<std::size_t>(ni)]) {
                    mask[static_cast<std::size_t>(ni)] = 0; queue.push_back(ni);
                }
            }
        }
        if (static_cast<int>(queue.size()) >= min_pixels)
            components.push_back({weighted / weight_sum, static_cast<int>(queue.size()), peak});
    }
    return components;
}

std::vector<Component> sortByAngle(const std::vector<Component>& input) {
    Point2 center = Point2::Zero(); for (const auto& c : input) center += c.centroid;
    center /= static_cast<double>(input.size());
    auto output = input;
    std::sort(output.begin(), output.end(), [&](const Component& a, const Component& b) {
        return std::atan2(a.centroid.y() - center.y(), a.centroid.x() - center.x()) <
               std::atan2(b.centroid.y() - center.y(), b.centroid.x() - center.x());
    });
    return output;
}

std::vector<Point2> candidateFromComponents(const std::vector<Component>& components,
                                             const PhantomSpec& spec, int swap,
                                             int reverse_a, int reverse_b,
                                             int shift_a, int shift_b,
                                             int marker_swap) {
    const int marker_count = static_cast<int>(markerPoints(spec).size());
    if (static_cast<int>(components.size()) != 2 * spec.beads_per_ring + marker_count) return {};
    std::vector<int> marker_ids(static_cast<std::size_t>(marker_count));
    std::vector<int> order(components.size()); std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + marker_count, order.end(),
                      [&](int a, int b) { return components[a].area > components[b].area; });
    for (int i = 0; i < marker_count; ++i) marker_ids[static_cast<std::size_t>(i)] = order[static_cast<std::size_t>(i)];
    std::vector<Component> ordinary;
    for (int i = 0; i < static_cast<int>(components.size()); ++i)
        if (std::find(marker_ids.begin(), marker_ids.end(), i) == marker_ids.end()) ordinary.push_back(components[i]);
    std::sort(ordinary.begin(), ordinary.end(), [](const Component& a, const Component& b) { return a.centroid.y() < b.centroid.y(); });
    std::vector<Component> a(ordinary.begin(), ordinary.begin() + spec.beads_per_ring);
    std::vector<Component> b(ordinary.begin() + spec.beads_per_ring, ordinary.end());
    if (swap) std::swap(a, b);
    a = sortByAngle(a); b = sortByAngle(b);
    if (reverse_a) std::reverse(a.begin(), a.end()); if (reverse_b) std::reverse(b.begin(), b.end());
    std::rotate(a.begin(), a.begin() + (shift_a % static_cast<int>(a.size())), a.end());
    std::rotate(b.begin(), b.begin() + (shift_b % static_cast<int>(b.size())), b.end());
    std::vector<Point2> result(static_cast<std::size_t>(2 * spec.beads_per_ring + marker_count));
    for (int i = 0; i < spec.beads_per_ring; ++i) { result[2 * i] = a[i].centroid; result[2 * i + 1] = b[i].centroid; }
    if (marker_count >= 1)
        result[2 * spec.beads_per_ring] = components[marker_ids[0]].centroid;
    if (marker_count == 2) {
        result[2 * spec.beads_per_ring + 1] = components[marker_ids[1]].centroid;
        if (marker_swap) std::swap(result[2 * spec.beads_per_ring], result[2 * spec.beads_per_ring + 1]);
    }
    return result;
}

std::vector<Point2> establishAnchor(const std::vector<Component>& components,
                                    const PhantomSpec& spec, PixelSize pixel) {
    const auto world = makeIdentifiableDoubleRing(spec);
    struct Candidate {
        double projective_rmse = std::numeric_limits<double>::infinity();
        std::vector<Point2> points;
    };
    std::vector<Candidate> candidates;
    const int marker_count = static_cast<int>(markerPoints(spec).size());
    for (int swap = 0; swap < 2; ++swap) for (int ra = 0; ra < 2; ++ra) for (int rb = 0; rb < 2; ++rb)
        for (int shift_a = 0; shift_a < spec.beads_per_ring; ++shift_a)
        for (int shift_b = 0; shift_b < spec.beads_per_ring; ++shift_b)
        for (int marker_swap = 0; marker_swap < std::max(1, marker_count); ++marker_swap) {
            auto candidate = candidateFromComponents(components, spec, swap, ra, rb,
                                                     shift_a, shift_b, marker_swap);
            if (candidate.empty()) continue;
            const auto p = normalizedDlt(world, candidate);
            const double error = reprojectionRmse(p, world, candidate);
            if (std::isfinite(error)) candidates.push_back({error, std::move(candidate)});
        }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.projective_rmse < b.projective_rmse;
              });
    // A free projective camera can give a very small DLT residual for a
    // geometrically wrong cyclic map.  Re-score the best projective maps with
    // the physical flat-panel camera (known pixel pitch, zero skew, one SDD),
    // which is the constraint used by the Python reference tracker.
    double best = std::numeric_limits<double>::infinity();
    std::vector<Point2> selected;
    const std::size_t physical_count = std::min<std::size_t>(32, candidates.size());
    for (std::size_t i = 0; i < physical_count; ++i) {
        try {
            const auto camera = calibratePhysicalCamera(world, candidates[i].points,
                                                        pixel, 300);
            if (camera.reprojection_rmse_px < best) {
                best = camera.reprojection_rmse_px;
                selected = candidates[i].points;
            }
        } catch (const std::exception&) {
            // An invalid projective branch is simply not a physical camera.
        }
    }
    if (selected.empty() || best > 5.0)
        throw std::runtime_error("cannot establish physical phantom point IDs");
    return selected;
}

std::vector<Point2> measureFromSeeds(const std::vector<float>& image, ImageShape shape,
                                     const std::vector<Point2>& seeds, float threshold,
                                     double gate, int min_pixels, std::vector<unsigned char>& valid) {
    std::vector<Point2> result(seeds.size(), Point2::Constant(std::numeric_limits<double>::quiet_NaN()));
    valid.assign(seeds.size(), 0); std::vector<Point2> sums(seeds.size(), Point2::Zero()); std::vector<double> weights(seeds.size(), 0.0); std::vector<int> counts(seeds.size(), 0);
    for (int y = 0; y < shape.rows; ++y) for (int x = 0; x < shape.cols; ++x) {
        const float value = image[static_cast<std::size_t>(y * shape.cols + x)]; if (value <= threshold) continue;
        const Point2 p(static_cast<double>(x), static_cast<double>(y)); int owner = -1; double dmin = gate * gate;
        for (std::size_t i = 0; i < seeds.size(); ++i) { const double d = (p - seeds[i]).squaredNorm(); if (d < dmin) { dmin = d; owner = static_cast<int>(i); } }
        if (owner >= 0) { const double w = std::max(1.0e-6, static_cast<double>(value - threshold)); sums[owner] += w * p; weights[owner] += w; ++counts[owner]; }
    }
    for (std::size_t i = 0; i < seeds.size(); ++i) if (counts[i] >= min_pixels) { result[i] = sums[i] / weights[i]; valid[i] = 1; }
    return result;
}

std::vector<int> nearestAssignment(const std::vector<Point2>& predicted,
                                   const std::vector<Component>& components) {
    const int n = static_cast<int>(predicted.size());
    if (static_cast<int>(components.size()) != n) return {};
    // Kuhn-Munkres assignment avoids the ID swaps that a greedy nearest pair
    // can make when two projected beads approach one another.
    std::vector<double> u(static_cast<std::size_t>(n + 1)), v(static_cast<std::size_t>(n + 1));
    std::vector<int> p(static_cast<std::size_t>(n + 1)), way(static_cast<std::size_t>(n + 1));
    for (int i = 1; i <= n; ++i) {
        p[0] = i; int j0 = 0;
        std::vector<double> minv(static_cast<std::size_t>(n + 1), std::numeric_limits<double>::infinity());
        std::vector<unsigned char> used(static_cast<std::size_t>(n + 1), 0);
        do {
            used[static_cast<std::size_t>(j0)] = 1;
            const int i0 = p[static_cast<std::size_t>(j0)]; double delta = std::numeric_limits<double>::infinity(); int j1 = 0;
            for (int j = 1; j <= n; ++j) if (!used[static_cast<std::size_t>(j)]) {
                const double cost = (predicted[static_cast<std::size_t>(i0 - 1)] -
                                     components[static_cast<std::size_t>(j - 1)].centroid).squaredNorm();
                const double cur = cost - u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
                if (cur < minv[static_cast<std::size_t>(j)]) {
                    minv[static_cast<std::size_t>(j)] = cur; way[static_cast<std::size_t>(j)] = j0;
                }
                if (minv[static_cast<std::size_t>(j)] < delta) { delta = minv[static_cast<std::size_t>(j)]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)]) { u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta; v[static_cast<std::size_t>(j)] -= delta; }
                else minv[static_cast<std::size_t>(j)] -= delta;
            }
            j0 = j1;
        } while (p[static_cast<std::size_t>(j0)] != 0);
        do {
            const int j1 = way[static_cast<std::size_t>(j0)];
            p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int> assignment(static_cast<std::size_t>(n), -1);
    for (int j = 1; j <= n; ++j) assignment[static_cast<std::size_t>(p[static_cast<std::size_t>(j)] - 1)] = j - 1;
    return assignment;
}

}  // namespace

std::vector<Component> detectComponents(const std::vector<float>& image, ImageShape shape,
                                        float threshold, int min_pixels) {
    return connectedComponents(image, shape, threshold, min_pixels);
}

TrackingResult trackIdentifiablePhantom(RawStackReader& raw, const TrackerConfig& config,
                                        const PhantomSpec& phantom, PixelSize pixel) {
    std::vector<float> image; int anchor = -1; std::vector<Component> anchor_components;
    for (int view = 0; view < raw.views(); ++view) {
        raw.readFrame(view, image); auto components = detectComponents(image, config.image, config.threshold, config.min_pixels);
        if (static_cast<int>(components.size()) == 2 * phantom.beads_per_ring +
            static_cast<int>(markerPoints(phantom).size())) {
            anchor = view; anchor_components = std::move(components); break;
        }
    }
    if (anchor < 0) throw std::runtime_error("no separated anchor frame in RAW stack");
    const auto anchor_points = establishAnchor(anchor_components, phantom, pixel);
    const int targets = static_cast<int>(anchor_points.size());
    TrackingResult result; result.views = raw.views(); result.targets = targets; result.anchor_frame = anchor;
    result.points.assign(static_cast<std::size_t>(result.views * targets), Point2::Constant(std::numeric_limits<double>::quiet_NaN()));
    result.observed.assign(static_cast<std::size_t>(result.views * targets), 0); result.dlt_rmse_px.assign(result.views, std::numeric_limits<double>::quiet_NaN());
    auto run = [&](int direction) {
        std::vector<float> frame; std::vector<Point2> previous = anchor_points, before; std::vector<unsigned char> active(static_cast<std::size_t>(targets), 1);
        const int steps = (raw.views() + 1) / 2;
        for (int step = 0; step <= steps; ++step) {
            const int view = (anchor + direction * step + raw.views()) % raw.views();
            if (step != 0) {
                raw.readFrame(view, frame); std::vector<Point2> predicted = previous;
                if (!before.empty()) for (int i = 0; i < targets; ++i) predicted[i] = previous[i] + (previous[i] - before[i]);
                const auto components = detectComponents(frame, config.image, config.threshold, config.min_pixels);
                std::vector<Point2> measured;
                std::vector<unsigned char> valid;
                if (static_cast<int>(components.size()) == targets) {
                    measured.assign(static_cast<std::size_t>(targets), Point2::Constant(std::numeric_limits<double>::quiet_NaN()));
                    valid.assign(static_cast<std::size_t>(targets), 0);
                    const auto assignment = nearestAssignment(predicted, components);
                    for (int i = 0; i < targets; ++i) {
                        const int component = assignment[static_cast<std::size_t>(i)];
                        if (component >= 0 && (predicted[static_cast<std::size_t>(i)] - components[static_cast<std::size_t>(component)].centroid).norm() <= config.gate_radius_px) {
                            measured[static_cast<std::size_t>(i)] = components[static_cast<std::size_t>(component)].centroid;
                            valid[static_cast<std::size_t>(i)] = 1;
                        }
                    }
                } else {
                    measured = measureFromSeeds(frame, config.image, predicted, config.threshold, config.gate_radius_px, config.min_pixels, valid);
                }
                for (int i = 0; i < targets; ++i) active[static_cast<std::size_t>(i)] = active[static_cast<std::size_t>(i)] && valid[static_cast<std::size_t>(i)] && predicted[i].x() >= 0.0 && predicted[i].x() < config.image.cols && predicted[i].y() >= 0.0 && predicted[i].y() < config.image.rows;
                before = previous; previous = measured;
            }
            for (int i = 0; i < targets; ++i) if (active[static_cast<std::size_t>(i)]) { result.points[static_cast<std::size_t>(view * targets + i)] = previous[i]; result.observed[static_cast<std::size_t>(view * targets + i)] = 1; }
        }
    };
    run(+1); run(-1);
    return result;
}

}  // namespace cbct::calibration
