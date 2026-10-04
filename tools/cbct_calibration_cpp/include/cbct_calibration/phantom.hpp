#pragma once

#include "cbct_calibration/types.hpp"

namespace cbct::calibration {

std::vector<Point3> makeIdentifiableDoubleRing(const PhantomSpec& spec);
std::vector<Point3> markerPoints(const PhantomSpec& spec);

// Point factories used by the independent C++ calibration layer.  They only
// define 3-D IDs; image tracking remains a phantom-specific business layer.
std::vector<Point3> makeChoDoubleRing(int beads_per_ring = 12,
                                      double radius_mm = 50.0,
                                      double half_spacing_mm = 50.0);
std::vector<Point3> makeYangDoubleRing(int beads_per_ring = 6,
                                       double radius_mm = 50.0,
                                       double half_spacing_mm = 50.0);
std::vector<Point3> makeSingleRow(int bead_count, double spacing_mm);

}  // namespace cbct::calibration
