#include "calibration/GeometryCalibrationStage.hpp"
#include <cmath>
namespace yk::spectral::calibration {
GeometrySpec nominalGeometry(GeometrySpec geometry)
{
    auto& g = geometry.parameters;
    g.offset_u_mm = g.offset_v_mm = g.offset_n_mm = 0;
    g.tilt_u_rad = g.tilt_v_rad = g.tilt_n_rad = 0;
    g.source_offset_x_mm = g.source_offset_y_mm = g.source_offset_z_mm = 0;
    return geometry;
}
GeometryCorrectionStage::GeometryCorrectionStage()
{
    info_.id = "calibration.geometry_correction";
    info_.name = "Apply calibrated circular geometry";
    info_.description = "Build reconstruction geometry in the solver's gauge-fixed scanner frame; leaves simulation truth and projection pixels unchanged.";
    info_.version = "1.0";
    info_.kind = pipeline::PluginKind::CalibrationStage;
    info_.category = "geometry";
    info_.implementation_status = pipeline::ImplementationStatus::Experimental;
}
void GeometryCorrectionStage::process(Data& d, const Context& c)
{
    d.corrected.reset();
    if (d.domain != Domain::Geometry || d.location != pipeline::Location::Host || !d.result)
        throw std::invalid_argument("Geometry correction requires a calibration result");
    const auto& r = *d.result;
    const auto& m = r.machine;
    if (!r.joint_converged || !std::isfinite(r.joint_rmse_px) || !std::isfinite(c.maximum_rmse_px) || c.maximum_rmse_px <= 0 || r.joint_rmse_px > c.maximum_rmse_px)
        throw std::runtime_error("Fit did not converge or exceeds the accepted pixel error; correction not applied");
    for (double v : { m.sid_mm, m.sdd_mm, m.offset_u_mm, m.offset_v_mm, m.tilt_u_rad, m.tilt_v_rad, m.tilt_n_rad })
        if (!std::isfinite(v))
            throw std::invalid_argument("Non-finite calibrated geometry");
    if (m.sid_mm <= 0 || m.sdd_mm <= m.sid_mm || (r.rotation_direction != 1 && r.rotation_direction != -1))
        throw std::invalid_argument("Invalid calibrated distances or rotation direction");
    if (c.nominal.kind != GeometryKind::FlatCbct || !c.full_circle)
        throw std::invalid_argument("Correction currently supports full circular flat CBCT");
    auto corrected = nominalGeometry(c.nominal);
    auto& g = corrected.parameters;
    g.sid_mm = m.sid_mm;
    g.sdd_mm = m.sdd_mm;
    g.offset_u_mm = m.offset_u_mm;
    g.offset_v_mm = m.offset_v_mm;
    g.tilt_u_rad = m.tilt_u_rad;
    g.tilt_v_rad = m.tilt_v_rad;
    g.tilt_n_rad = m.tilt_n_rad;
    g.start_angle_rad = 0;
    g.start_z_mm = 0;
    g.rotation_direction = r.rotation_direction;
    g.views_per_turn = g.views;
    // Calibrated frame differs by a rigid transform from the simulation frame.
    // Object pose is estimated separately in CalibrationResult, never copied as truth.
    g.phantom_offset_x_mm = g.phantom_offset_y_mm = g.phantom_offset_z_mm = 0;
    g.phantom_rotation_x_rad = g.phantom_rotation_y_rad = g.phantom_rotation_z_rad = 0;
    d.corrected = corrected;
}
}
