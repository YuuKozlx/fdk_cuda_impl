#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "YKCBCT/geometry/YkSystemGeometry.hpp"
#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtGeometry.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomography.hpp"

namespace Viewer {
using Point = std::array<double, 3>;
inline Point point(float4 p) { return {p.x, p.y, p.z}; }
inline Point point(float3 p) { return {p.x, p.y, p.z}; }
inline Point subtract(Point a, Point b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline double dot(Point a, Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline Point cross(Point a, Point b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
inline Point normalized(Point v) {
    const double length = std::sqrt(dot(v, v));
    if (!std::isfinite(length) || length < 1e-10)
        throw std::runtime_error("Cannot derive local coordinate frame");
    return {v[0]/length, v[1]/length, v[2]/length};
}
struct LocalFrame {
    Point origin;
    std::array<Point, 3> basis;
};
struct DisplayTransform {
    std::array<Point, 3> rows{{{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}}};
    Point translation{0.,0.,0.};
    Point vector(Point p) const { return {dot(rows[0],p), dot(rows[1],p), dot(rows[2],p)}; }
    Point position(Point p) const {
        auto result = vector(p);
        for (int j = 0; j < 3; ++j) result[j] += translation[j];
        return result;
    }
    LocalFrame frame(const LocalFrame& f) const {
        return {position(f.origin), {vector(f.basis[0]), vector(f.basis[1]), vector(f.basis[2])}};
    }
};

struct Options {
    std::string geometry = "cyl-helical";
    int views = 180, views_per_turn = 90, frame = 0;
    int channels = 160, rows = 96;
    double sid = 120., sdd = 240., du = 1.5, dv = 1.5;
    double radius = 240., pitch = 36., tilt = 30.;
    std::vector<int> size{64, 64, 48};
    std::vector<double> spacing{1., 1., 1.};
    std::vector<double> center{0., 0., 0.};
    std::string raw, screenshot, export_dir;
    bool box_only = false, offscreen = false, self_check = false, gantry_fixed = false;
};

class Scene {
public:
    explicit Scene(const Options& o) : options(o) {
        if (o.gantry_fixed && !supportsRelative())
            throw std::runtime_error("SourceOrbit motion is non-rigid: source and detector cannot both be fixed");
        if (o.views < 2 || o.views > 100000 || o.views_per_turn < 2 ||
            o.channels < 2 || o.rows < 2 || o.frame < 0 || o.frame >= o.views ||
            !std::isfinite(o.sid) || !std::isfinite(o.sdd) || !(o.sdd > o.sid) ||
            !(o.sid > 0.) || !std::isfinite(o.du) || !std::isfinite(o.dv) ||
            !(o.du > 0.) || !(o.dv > 0.) || !std::isfinite(o.radius) || !(o.radius > 0.) ||
            !std::isfinite(o.pitch) || !std::isfinite(o.tilt) ||
            o.size.size() != 3 || o.spacing.size() != 3 || o.center.size() != 3)
            throw std::runtime_error("Invalid geometry, frame, or volume parameters");
        for (int i = 0; i < 3; ++i)
            if (o.size[i] < 2 || o.size[i] > 512 || !std::isfinite(o.spacing[i]) ||
                !(o.spacing[i] > 0.) || !std::isfinite(o.center[i]))
                throw std::runtime_error("Volume size must be 2..512 with finite positive spacing");
        const size_t elements = size_t(o.size[0])*o.size[1]*o.size[2];
        if (elements > 128u*1024*1024)
            throw std::runtime_error("Prototype volume limit is 128M float elements");
        YK::SSystemConfig system{};
        system.volume = {o.size[0], o.size[1], o.size[2], float(o.spacing[0]),
            float(o.spacing[1]), float(o.spacing[2]),
            make_float3(float(o.center[0]), float(o.center[1]), float(o.center[2]))};
        system.detector = o.geometry.rfind("cyl-", 0) == 0
            ? YK::EDetectorKind::Cylindrical : YK::EDetectorKind::Flat;
        system.trajectory = o.geometry.find("helical") != std::string::npos
            ? YK::ETrajectoryKind::Helical : YK::ETrajectoryKind::Circular;
        system.circular.total_views = o.views;
        system.circular.views_per_turn = o.views_per_turn;
        system.circular.sid_mm = float(o.sid); system.circular.sdd_mm = float(o.sdd);
        system.helical.total_views = o.views;
        system.helical.views_per_turn = o.views_per_turn;
        system.helical.sid_mm = float(o.sid); system.helical.sdd_mm = float(o.sdd);
        system.helical.pitch_mm_per_turn = float(o.pitch);
        system.helical.start_z_mm = float(-0.5*o.pitch*(o.views-1)/o.views_per_turn);
        system.flat_detector = {o.channels, o.rows, float(o.du), float(o.dv), {}};
        system.cylindrical_detector = {o.channels, o.rows, float(o.du), float(o.dv),
            float(o.radius), {}};
        if (o.geometry == "source-orbit" || o.geometry == "nikon-tilted") {
            if (!YK::buildVolumeGeometry(system.volume, volume))
                throw std::runtime_error("Invalid volume");
            constexpr float two_pi = 6.2831853071795864769f;
            std::vector<float> angles(o.views);
            for (int i = 0; i < o.views; ++i) angles[i] = two_pi*i/o.views_per_turn;
            bool ok = false;
            if (o.geometry == "source-orbit") {
                YK::NonStandardFpBp::SourceOrbitGeometryConfig g{};
                g.detector_u = o.channels; g.detector_v = o.rows;
                g.detector_pixel_u_mm = float(o.du); g.detector_pixel_v_mm = float(o.dv);
                g.sid_mm = float(o.sid); g.detector_y_mm = float(o.sdd-o.sid);
                g.source_orbit_radius_mm = float(o.sid*0.15);
                ok = YK::NonStandardFpBp::buildSourceOrbitGeometry(g, angles, flat);
                // Match the public backend's source-orbit volume-center translation.
                for (auto& v : flat) {
                    v.src.x += volume.center.x; v.src.y += volume.center.y; v.src.z += volume.center.z;
                    v.detS.x += volume.center.x; v.detS.y += volume.center.y; v.detS.z += volume.center.z;
                }
            } else {
                YK::NonStandardFpBp::NikonTiltedCt::GeometryConfig g{};
                g.detector_u = o.channels; g.detector_v = o.rows;
                g.detector_pixel_u_mm = float(o.du); g.detector_pixel_v_mm = float(o.dv);
                g.sid_mm = float(o.sid); g.sdd_mm = float(o.sdd);
                g.source = make_float3(0.f, -float(o.sid), 0.f);
                g.tilt_rad = float(o.tilt*two_pi/360.);
                std::vector<YK::NonStandardFpBp::NonStandardViewPose> poses;
                ok = YK::NonStandardFpBp::NikonTiltedCt::buildTiltedAxisGeometry(g, angles, poses, flat);
            }
            if (!ok) throw std::runtime_error("Non-standard geometry builder rejected configuration");
        } else if (!YK::buildSystemGeometry(system, flat, cyl, volume)) {
            throw std::runtime_error("System geometry builder rejected configuration");
        }
    }

    Point source(int i) const { return cyl.empty() ? point(flat.at(i).src) : point(cyl.at(i).source); }
    bool supportsRelative() const { return options.geometry != "source-orbit"; }
    DisplayTransform gantryFixedTransform(int i) const {
        if (!supportsRelative()) throw std::runtime_error("SourceOrbit has no rigid gantry pose");
        const auto reference = detectorFrame(0);
        const auto current = detectorFrame(i);
        DisplayTransform t;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                t.rows[row][col] = 0.;
                for (int axis = 0; axis < 3; ++axis)
                    t.rows[row][col] += reference.basis[axis][row]*current.basis[axis][col];
            }
        t.translation = subtract(reference.origin, t.vector(current.origin));
        return t;
    }
    LocalFrame detectorFrame(int i) const {
        const Point origin = detectorPoint(i, (options.channels-1)*0.5, (options.rows-1)*0.5);
        const Point u = normalized(cyl.empty() ? point(flat.at(i).detU) : point(cyl.at(i).detectorU));
        const Point v = normalized(cyl.empty() ? point(flat.at(i).detV) : point(cyl.at(i).detectorV));
        // U and V follow pixel indices; U x V is the inward surface normal.
        return {origin, {u, v, normalized(cross(u, v))}};
    }
    LocalFrame sourceFrame(int i) const {
        const auto detector = detectorFrame(i);
        const Point origin = source(i);
        const Point ray = normalized(subtract(detector.origin, origin));
        const double projection = dot(detector.basis[0], ray);
        Point u = detector.basis[0];
        for (int j = 0; j < 3; ++j) u[j] -= projection*ray[j];
        u = normalized(u);
        // Right-handed U/R/V gauge: R points source -> detector, V=U x R.
        // This is derived from rays, not an independent physical tube pose.
        return {origin, {u, ray, normalized(cross(u, ray))}};
    }
    Point detectorPoint(int i, double u, double v) const {
        if (cyl.empty()) {
            const auto& g = flat.at(i);
            return {g.detS.x+u*g.detU.x+v*g.detV.x,
                g.detS.y+u*g.detU.y+v*g.detV.y, g.detS.z+u*g.detU.z+v*g.detV.z};
        }
        YK::SCylProjectionFrame f{};
        if (!YK::deriveCylProjectionFrame(cyl.at(i), options.channels, options.rows, f))
            throw std::runtime_error("Cannot derive cylindrical frame");
        const double phi = (u-f.principalU)*f.channelStepRad;
        const double h = (v-f.principalV)*f.rowStepMm;
        return {f.cylinderCenter.x+f.radius_mm*(f.radialUnit.x*std::cos(phi)+f.tangentUnit.x*std::sin(phi))+h*f.axisUnit.x,
            f.cylinderCenter.y+f.radius_mm*(f.radialUnit.y*std::cos(phi)+f.tangentUnit.y*std::sin(phi))+h*f.axisUnit.y,
            f.cylinderCenter.z+f.radius_mm*(f.radialUnit.z*std::cos(phi)+f.tangentUnit.z*std::sin(phi))+h*f.axisUnit.z};
    }
    Options options;
    YK::SVolGeom volume{};
    std::vector<YK::SConeProjGeomVec> flat;
    std::vector<YK::SCylConeProjGeomVec> cyl;
};
} // namespace Viewer
