#pragma once

#include "cbct_calibration/types.hpp"

namespace cbct::calibration {

SourceCircle fitSourceCircle(const std::vector<Point3>& sources);
void scannerAlignment(const SourceCircle& circle,
                      Eigen::Matrix3d& rotation,
                      Point3& translation);

}  // namespace cbct::calibration

