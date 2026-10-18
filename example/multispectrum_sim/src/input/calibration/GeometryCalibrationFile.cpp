#include "input/config/ReconstructionGeometry.hpp"
#include "recon/geometry/GeometryCalibrationStage.hpp"
#include <cmath>
#include <fstream>
#include <toml++/toml.hpp>
namespace yk::spectral::calibration {
namespace {
    void valid(const GeometrySpec& s, double rmse)
    {
        const auto& g = s.parameters;
        if (s.kind != GeometryKind::FlatCbct || g.views < 3 || g.views_per_turn != g.views || g.detector_u <= 0 || g.detector_v <= 0 || !std::isfinite(g.pixel_u_mm) || !std::isfinite(g.pixel_v_mm) || g.pixel_u_mm <= 0 || g.pixel_v_mm <= 0 || !std::isfinite(rmse) || rmse < 0 || (g.rotation_direction != 1 && g.rotation_direction != -1))
            throw std::invalid_argument("Invalid circular geometry calibration");
        SimulationConfig c;
        c.geometry = s;
        c.reconstruction.geometry_override.enabled = true;
        c.reconstruction.geometry_override.parameters = g;
        resolveReconstructionGeometry(c);
        if (g.start_angle_rad != 0 || g.start_z_mm != 0)
            throw std::invalid_argument("Geometry correction must use the calibrated scanner frame");
    }
}
void saveGeometryCorrection(const std::filesystem::path& path, const GeometrySpec& s, double rmse)
{
    valid(s, rmse);
    const auto& g = s.parameters;
    toml::table t { { "format", "circular_geometry_correction" }, { "version", 1 }, { "length_unit", "mm" }, { "angle_unit", "rad" },
        { "coordinate_frame", "calibrated_scanner" }, { "rmse_px", rmse }, { "views", g.views }, { "detector_u", g.detector_u },
        { "detector_v", g.detector_v }, { "pixel_u_mm", g.pixel_u_mm }, { "pixel_v_mm", g.pixel_v_mm }, { "rotation_direction", g.rotation_direction } };
    for (const auto& field : reconstructionGeometryFields())
        t.insert(field.key, g.*field.member);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f << t;
    f.close();
}
GeometrySpec loadGeometryCorrection(const std::filesystem::path& path, const GeometrySpec& acquisition, double max_rmse)
{
    auto t = toml::parse_file(path.string());
    if (t["format"].value_or(std::string{}) != "circular_geometry_correction" || t["version"].value_or(0) != 1 || t["length_unit"].value_or(std::string{}) != "mm" || t["angle_unit"].value_or(std::string{}) != "rad" || t["coordinate_frame"].value_or(std::string{}) != "calibrated_scanner")
        throw std::runtime_error("Unsupported geometry correction format/frame/units");
    auto number = [&](const char* key) {auto v=t[key].value<double>();if(!v || !std::isfinite(*v))throw std::runtime_error(std::string("Invalid geometry correction field: ")+key);return *v; };
    if(acquisition.kind!=GeometryKind::FlatCbct || acquisition.parameters.views!=acquisition.parameters.views_per_turn)
        throw std::runtime_error("Geometry correction requires a complete circular acquisition");
    auto s = nominalGeometry(acquisition);
    auto& g = s.parameters;
    const double rmse = number("rmse_px");
    if (!std::isfinite(max_rmse) || max_rmse <= 0 || rmse > max_rmse)
        throw std::runtime_error("Geometry calibration exceeds accepted RMSE");
    if (number("views") != g.views || number("detector_u") != g.detector_u || number("detector_v") != g.detector_v || std::abs(number("pixel_u_mm") - g.pixel_u_mm) > 1e-9 || std::abs(number("pixel_v_mm") - g.pixel_v_mm) > 1e-9)
        throw std::runtime_error("Geometry calibration does not match acquisition dimensions/pixel size");
    for (const auto& field : reconstructionGeometryFields())
        g.*field.member = number(field.key);
    const auto direction = t["rotation_direction"].value<int>();
    if (!direction)
        throw std::runtime_error("Missing calibration rotation direction");
    g.rotation_direction = *direction;
    g.views_per_turn = g.views;
    g.start_angle_rad = g.start_z_mm = 0;
    g.phantom_offset_x_mm = g.phantom_offset_y_mm = g.phantom_offset_z_mm = 0;
    g.phantom_rotation_x_rad = g.phantom_rotation_y_rad = g.phantom_rotation_z_rad = 0;
    valid(s, rmse);
    return s;
}
}
