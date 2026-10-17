#pragma once

#include "cbct_calibration/io.hpp"
#include "cbct_calibration/types.hpp"

namespace cbct::calibration {
TrackingResult trackFromAnchor(RawStackReader& raw, const TrackerConfig& config,
                               int anchor, const std::vector<Point2>& anchor_points,
                               const std::vector<Point3>& world);
TrackingResult trackSpiral(RawStackReader& raw, const PipelineConfig& config);
std::vector<Point3> makeSpiral(const PipelineConfig& config);

std::vector<Component> detectComponents(const std::vector<float>& image,
                                        ImageShape shape,
                                        float threshold,
                                        int min_pixels = 4);

TrackingResult trackIdentifiablePhantom(RawStackReader& raw,
                                        const TrackerConfig& config,
                                        const PhantomSpec& phantom,
                                        PixelSize pixel);


}  // namespace cbct::calibration
