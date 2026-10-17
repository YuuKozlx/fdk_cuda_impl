#include "calibration/GeometryCalibrationStage.hpp"
#include "cbct_calibration/tracker.hpp"
#include <cmath>
namespace yk::spectral::calibration {
GeometryCalibrationStage::GeometryCalibrationStage()
{
    info_.id = "calibration.geometry";
    info_.name = "Circular CBCT geometry calibration";
    info_.description = "Double rings, marked spiral or single-row negative-log RAW. Spatial phantoms use DLT and joint fitting; single rows use a restricted five-parameter circular model.";
    info_.version = "1.1";
    info_.kind = pipeline::PluginKind::CalibrationStage;
    info_.category = "geometry";
    info_.tags = { "cpu", "full_circle", "flat_detector" };
    info_.implementation_status = pipeline::ImplementationStatus::Experimental;
}
void GeometryCalibrationStage::validateContext(const Context& c) const
{
    if (c.input != InputKind::ProjectionRaw && c.input != InputKind::IndexedPoints)
        throw std::invalid_argument("Unknown calibration input kind");
    const auto& t = c.config.tracker;
    const auto& p = c.config.pixel;
    if (!c.full_circle || t.views < 3 || t.image.rows <= 0 || t.image.cols <= 0)
        throw std::invalid_argument("Geometry calibration requires a full circular scan and valid dimensions");
    if (!std::isfinite(p.u_mm) || !std::isfinite(p.v_mm) || p.u_mm <= 0 || p.v_mm <= 0 || c.config.joint_max_iterations <= 0 || !std::isfinite(c.config.robust_scale_px) || c.config.robust_scale_px <= 0)
        throw std::invalid_argument("Invalid calibration pixel size or fitting parameters");
    if (c.input == InputKind::ProjectionRaw) {
        if (t.raw_path.empty() || !std::isfinite(t.threshold) || t.threshold <= 0 || t.min_pixels <= 0 || !std::isfinite(t.gate_radius_px) || t.gate_radius_px <= 0)
            throw std::invalid_argument("Invalid RAW detector/tracking settings");
        using Kind = cbct::calibration::PipelineConfig::PhantomKind;
        if (!std::isfinite(t.marker_area_ratio) || t.marker_area_ratio <= 1 || !std::isfinite(t.max_dlt_rmse_px) || t.max_dlt_rmse_px <= 0)
            throw std::invalid_argument("Invalid marker tracking thresholds");
        if (c.config.phantom_kind == Kind::Spiral) {
            cbct::calibration::makeSpiral(c.config);
            return;
        }
        if (c.config.phantom_kind == Kind::SingleRow) {
            if (c.config.row_beads < 3 || !std::isfinite(c.config.row_spacing_mm) || c.config.row_spacing_mm <= 0 || t.views < 12 || !std::isfinite(c.config.initial_sid_mm) || !std::isfinite(c.config.initial_sdd_mm) || c.config.initial_sid_mm <= 0 || c.config.initial_sdd_mm <= c.config.initial_sid_mm)
                throw std::invalid_argument("Single-row calibration requires >=3 beads, positive spacing and >=12 views");
            return;
        }
        if (c.config.phantom_kind != Kind::DoubleRing)
            throw std::invalid_argument("Unknown calibration phantom");
        const auto& f = c.config.phantom;
        for (double value : { f.upper_radius_mm, f.lower_radius_mm, f.ring_half_spacing_mm, f.lower_phase_deg })
            if (!std::isfinite(value))
                throw std::invalid_argument("Non-finite phantom dimensions");
        for (const auto& point : f.marker_points_mm)
            if (!point.allFinite())
                throw std::invalid_argument("Non-finite marker coordinates");
        if (f.beads_per_ring < 6 || f.beads_per_ring % 2 != 0 || f.upper_radius_mm <= 0 || f.lower_radius_mm <= 0 || f.ring_half_spacing_mm <= 0 || f.marker_points_mm.size() > 2)
            throw std::invalid_argument("Double rings require even bead counts >=6, positive dimensions and at most two markers");
    }
}
void GeometryCalibrationStage::process(Data& d, const Context& c)
{
    d.result.reset();
    d.corrected.reset();
    validateContext(c);
    if (d.domain != Domain::Measurements || d.location != pipeline::Location::Host)
        throw std::invalid_argument("Invalid calibration input domain/location");
    if (c.progress)
        c.progress({ "Geometry calibration", 0, 1 });
    cbct::calibration::CalibrationResult result;
    if (c.input == InputKind::ProjectionRaw)
        result = cbct::calibration::calibrateRaw(c.config);
    else {
        if (d.landmarks.size() < 6 || d.points.size() != size_t(c.config.tracker.views) * d.landmarks.size())
            throw std::invalid_argument("Indexed calibration needs >=6 non-coplanar landmarks and view-major image points");
        for (const auto& p : d.landmarks)
            if (!p.allFinite())
                throw std::invalid_argument("Non-finite landmark");
        if (!d.usable.empty() && d.usable.size() != d.points.size())
            throw std::invalid_argument("Invalid usability mask");
        for (size_t i = 0; i < d.points.size(); ++i)
            if ((d.usable.empty() || d.usable[i]) && !d.points[i].allFinite())
                throw std::invalid_argument("Non-finite measurement");
        auto usable = d.usable.empty() ? std::vector<unsigned char>(d.points.size(), 1) : d.usable;
        result = cbct::calibration::calibrateIndexedPoints(d.points, d.landmarks, usable, c.config.tracker.views, c.config);
    }
    if (!std::isfinite(result.joint_rmse_px) || !std::isfinite(result.machine.sid_mm) || !std::isfinite(result.machine.sdd_mm))
        throw std::runtime_error("Calibration produced non-finite geometry");
    d.result = std::move(result);
    if (c.progress)
        c.progress({ "Geometry calibration", 1, 1 });
}
}
