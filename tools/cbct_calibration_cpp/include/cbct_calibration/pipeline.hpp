#pragma once

#include "cbct_calibration/types.hpp"

namespace cbct::calibration {

CalibrationResult calibrateIndexed(const std::vector<Point2>& indexed_points,
                                   int views,
                                   const PipelineConfig& config);

// Generic geometry core.  `world_points` must use the same canonical frame as
// the indexed image points and have one entry per target, in ID order.
CalibrationResult calibrateIndexedPoints(const std::vector<Point2>& indexed_points,
                                         const std::vector<Point3>& world_points,
                                         int views,
                                         const PipelineConfig& config);

CalibrationResult calibrateRaw(const PipelineConfig& config);

}  // namespace cbct::calibration
