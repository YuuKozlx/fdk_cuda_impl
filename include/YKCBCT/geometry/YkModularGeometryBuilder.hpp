#pragma once

#include "YKCBCT/geometry/YkDetectorGeometryBuilders.hpp"
#include "YKCBCT/geometry/YkTrajectoryGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkPlanarGeometryBuilder.hpp"

namespace YK {

template <typename Trajectory>
inline bool buildProjectionGeometry(const Trajectory& trajectory,
    const SFlatDetectorSpec& detector,
    std::vector<SConeProjGeomVec>& geometry)
{
    std::vector<SScannerViewFrame> frames;
    return buildScannerViewFrames(trajectory, frames) &&
        buildFlatProjectionGeometry(frames, detector, geometry);
}

inline bool buildCalibratedCircularProjectionGeometry(
    const SCircularTrajectorySpec& trajectory,
    const SFlatDetectorSpec& detector,
    const std::vector<SViewGeometryCalibration>& calibration,
    std::vector<SConeProjGeomVec>& geometry)
{
    if (calibration.size() != trajectory.angles_rad.size()) return false;
    geometry.clear();
    geometry.reserve(calibration.size());
    for (size_t i = 0; i < calibration.size(); ++i) {
        SCircularTrajectorySpec view_trajectory = trajectory;
        view_trajectory.angles_rad = {trajectory.angles_rad[i]};
        view_trajectory.source_offset_mm = calibration[i].source_offset_mm;
        SFlatDetectorSpec view_detector = detector;
        view_detector.pose = calibration[i].detector_pose;
        std::vector<SConeProjGeomVec> view;
        if (!buildProjectionGeometry(view_trajectory, view_detector, view)) {
            geometry.clear();
            return false;
        }
        geometry.push_back(view.front());
    }
    return true;
}

template <typename Trajectory>
inline bool buildProjectionGeometry(const Trajectory& trajectory,
    const SCylDetectorSpec& detector,
    std::vector<SCylConeProjGeomVec>& geometry)
{
    std::vector<SScannerViewFrame> frames;
    return buildScannerViewFrames(trajectory, frames) &&
        buildCylProjectionGeometry(frames, detector, geometry);
}

} // namespace YK
