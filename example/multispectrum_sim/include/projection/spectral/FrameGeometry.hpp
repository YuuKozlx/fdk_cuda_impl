#pragma once
#include <array>
#include <vector>
namespace yk::spectral {
using Point = std::array<double, 3>;
using Pose = std::array<double, 16>; // Row-major, local point -> world point.
inline Pose identityPose() { return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; }
struct FrameGeometry {
    int view = 0;
    Point source {};
    // Two edge rows, U increasing. Covers pixel boundaries, not pixel centers.
    std::vector<Point> detector;
    Point detector_center {};
    std::array<Point, 3> detector_axes {};
    Pose object_pose = identityPose();
};
}
