#pragma once

#include "cbct_calibration/io.hpp"
#include "cbct_calibration/types.hpp"

namespace cbct::calibration {

std::vector<Component> detectComponents(const std::vector<float>& image,
                                        ImageShape shape,
                                        float threshold,
                                        int min_pixels = 4);

TrackingResult trackIdentifiablePhantom(RawStackReader& raw,
                                        const TrackerConfig& config,
                                        const PhantomSpec& phantom,
                                        PixelSize pixel);

}  // namespace cbct::calibration

