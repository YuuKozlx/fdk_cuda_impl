#pragma once

#include "cbct_calibration/io.hpp"
#include "cbct_calibration/types.hpp"

namespace cbct::calibration {

struct SingleRowConfig {
    float threshold = 5.0F;
    int min_pixels = 4;
    double max_jump_px = 40.0;
    int expected_beads = 0;  // zero means infer the first-frame count
};

struct SingleRowTrackingResult {
    int views = 0;
    int targets = 0;
    std::vector<Point2> points;
    std::vector<unsigned char> observed;
    std::vector<unsigned char> permanently_disabled;

    const Point2& at(int view, int target) const {
        return points.at(static_cast<std::size_t>(view * targets + target));
    }
};

// Detection and ID tracking only. A failed, out-of-panel, or over-gate bead is
// disabled permanently and is never reassigned to a later component.
SingleRowTrackingResult trackSingleRow(RawStackReader& raw,
                                       const SingleRowConfig& config);

}  // namespace cbct::calibration
