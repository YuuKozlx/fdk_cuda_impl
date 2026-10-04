#include "cbct_calibration/phantom.hpp"

#include <cmath>
#include <stdexcept>

namespace cbct::calibration {
namespace { constexpr double kPi = 3.14159265358979323846; }

std::vector<Point3> makeIdentifiableDoubleRing(const PhantomSpec& spec) {
    if (spec.beads_per_ring < 6 || spec.beads_per_ring % 2 != 0) {
        throw std::invalid_argument("beads_per_ring must be an even number >= 6");
    }
    std::vector<Point3> points;
    const auto markers = markerPoints(spec);
    points.reserve(static_cast<std::size_t>(2 * spec.beads_per_ring + markers.size()));
    const double step = 2.0 * kPi / static_cast<double>(spec.beads_per_ring);
    const double phase = spec.lower_phase_deg * kPi / 180.0;
    for (int i = 0; i < spec.beads_per_ring; ++i) {
        const double a = step * i;
        const double b = step * i + phase;
        points.emplace_back(spec.upper_radius_mm * std::cos(a),
                            spec.upper_radius_mm * std::sin(a),
                            spec.ring_half_spacing_mm);
        points.emplace_back(spec.lower_radius_mm * std::cos(b),
                            spec.lower_radius_mm * std::sin(b),
                            -spec.ring_half_spacing_mm);
    }
    points.insert(points.end(), markers.begin(), markers.end());
    return points;
}

std::vector<Point3> markerPoints(const PhantomSpec& spec) {
    return spec.marker_points_mm;
}

std::vector<Point3> makeChoDoubleRing(int beads_per_ring, double radius_mm,
                                      double half_spacing_mm) {
    if (beads_per_ring < 6 || beads_per_ring % 2 != 0)
        throw std::invalid_argument("beads_per_ring must be an even number >= 6");
    std::vector<Point3> points;
    const double step = 2.0 * kPi / beads_per_ring;
    for (int ring = 0; ring < 2; ++ring)
        for (int i = 0; i < beads_per_ring; ++i) {
            const double a = step * i;
            points.emplace_back(radius_mm * std::cos(a), radius_mm * std::sin(a),
                                ring == 0 ? -half_spacing_mm : half_spacing_mm);
        }
    return points;
}

std::vector<Point3> makeYangDoubleRing(int beads_per_ring, double radius_mm,
                                       double half_spacing_mm) {
    // Yang's standard six-plus-six layout has the same 3-D ring coordinates;
    // its PIC equations, not the point factory, define the paper convention.
    return makeChoDoubleRing(beads_per_ring, radius_mm, half_spacing_mm);
}

std::vector<Point3> makeSingleRow(int bead_count, double spacing_mm) {
    if (bead_count < 3 || spacing_mm <= 0.0)
        throw std::invalid_argument("single-row phantom needs >=3 beads and positive spacing");
    std::vector<Point3> points;
    points.reserve(static_cast<std::size_t>(bead_count));
    const double center = 0.5 * (bead_count - 1);
    for (int i = 0; i < bead_count; ++i)
        points.emplace_back((i - center) * spacing_mm, 0.0, 0.0);
    return points;
}

}  // namespace cbct::calibration
