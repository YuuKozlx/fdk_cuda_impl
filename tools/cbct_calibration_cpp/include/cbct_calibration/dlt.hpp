#pragma once

#include "cbct_calibration/types.hpp"

namespace cbct::calibration {

ProjectionMatrix normalizedDlt(const std::vector<Point3>& world,
                               const std::vector<Point2>& image);
Point2 projectPoint(const ProjectionMatrix& projection, const Point3& point);
double reprojectionRmse(const ProjectionMatrix& projection,
                       const std::vector<Point3>& world,
                       const std::vector<Point2>& image);
DltCamera calibratePhysicalCamera(const std::vector<Point3>& world,
                                  const std::vector<Point2>& image,
                                  PixelSize pixel,
                                  int max_iterations = 50);

}  // namespace cbct::calibration

