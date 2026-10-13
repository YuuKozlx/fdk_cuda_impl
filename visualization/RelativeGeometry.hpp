#pragma once
#include "LiveView.hpp"
#include <cmath>
#include <stdexcept>

namespace yk::visualization {
inline Point posePoint(const Pose& m, const Point& p, bool direction = false)
{
    Point out {};
    for (int r = 0; r < 3; ++r) {
        out[r] = direction ? 0 : m[r * 4 + 3];
        for (int c = 0; c < 3; ++c)
            out[r] += m[r * 4 + c] * p[c];
    }
    return out;
}
inline Pose composePose(const Pose& a, const Pose& b)
{
    Pose out {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k)
                out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    return out;
}
// Orthonormalize the measured detector basis to remove float geometry roundoff.
inline Pose detectorRigidPose(const FrameGeometry& f)
{
    auto normalize = [](Point p) {double n=0;for(double v:p)n+=v*v;
        if(!std::isfinite(n)||n<1e-20)throw std::invalid_argument("invalid detector basis");
        for(auto& v:p)v/=std::sqrt(n);return p; };
    auto u = normalize(f.detector_axes[0]);
    auto v = f.detector_axes[1];
    double dot = 0;
    for (int i = 0; i < 3; ++i)
        dot += u[i] * v[i];
    for (int i = 0; i < 3; ++i)
        v[i] -= dot * u[i];
    v = normalize(v);
    Point n { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
    Pose out = identityPose();
    for (int r = 0; r < 3; ++r) {
        out[r * 4] = u[r];
        out[r * 4 + 1] = v[r];
        out[r * 4 + 2] = n[r];
        out[r * 4 + 3] = f.detector_center[r];
    }
    return out;
}
inline Pose relativeWorldPose(const FrameGeometry& reference, const FrameGeometry& current)
{
    const auto pose = detectorRigidPose(current);
    auto inverse = identityPose();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            inverse[r * 4 + c] = pose[c * 4 + r];
        inverse[r * 4 + 3] = 0;
        for (int c = 0; c < 3; ++c)
            inverse[r * 4 + 3] -= inverse[r * 4 + c] * pose[c * 4 + 3];
    }
    return composePose(detectorRigidPose(reference), inverse);
}
inline FrameGeometry relativeGeometry(const FrameGeometry& reference, const FrameGeometry& current)
{
    const auto transform = relativeWorldPose(reference, current);
    auto out = current;
    out.source = posePoint(transform, current.source);
    out.detector_center = posePoint(transform, current.detector_center);
    for (auto& point : out.detector)
        point = posePoint(transform, point);
    for (auto& axis : out.detector_axes)
        axis = posePoint(transform, axis, true);
    out.object_pose = composePose(transform, current.object_pose);
    return out;
}
inline bool rigidGantryMatches(const FrameGeometry& reference, const FrameGeometry& relative, double tolerance = .02)
{
    auto close = [&](const Point& a, const Point& b) {double d=0;for(int i=0;i<3;++i)d+=(a[i]-b[i])*(a[i]-b[i]);return d<=tolerance*tolerance; };
    if (!close(reference.source, relative.source) || reference.detector.size() != relative.detector.size())
        return false;
    for (size_t i = 0; i < reference.detector.size(); ++i)
        if (!close(reference.detector[i], relative.detector[i]))
            return false;
    return true;
}
} // namespace yk::visualization
