#include "cbct_calibration/io.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cbct::calibration {

RawStackReader::RawStackReader(const std::filesystem::path& path, int views,
                               ImageShape shape)
    : stream_(path, std::ios::binary), views_(views), shape_(shape) {
    if (!stream_) throw std::runtime_error("cannot open RAW file: " + path.string());
    if (views <= 0 || shape.rows <= 0 || shape.cols <= 0)
        throw std::invalid_argument("invalid RAW stack dimensions");
    stream_.seekg(0, std::ios::end);
    const auto bytes = stream_.tellg();
    const std::uintmax_t expected = static_cast<std::uintmax_t>(views) *
        static_cast<std::uintmax_t>(shape.rows) * static_cast<std::uintmax_t>(shape.cols) * 4U;
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) != expected)
        throw std::runtime_error("RAW size does not match configured dimensions");
    stream_.seekg(0, std::ios::beg);
}

void RawStackReader::readFrame(int view, std::vector<float>& destination) {
    if (view < 0 || view >= views_) throw std::out_of_range("RAW frame index");
    const std::size_t count = static_cast<std::size_t>(shape_.rows) * shape_.cols;
    destination.resize(count);
    stream_.seekg(static_cast<std::streamoff>(view * count * sizeof(float)), std::ios::beg);
    stream_.read(reinterpret_cast<char*>(destination.data()),
                 static_cast<std::streamsize>(count * sizeof(float)));
    if (!stream_) throw std::runtime_error("cannot read RAW frame");
}

void writeCalibrationJson(const std::filesystem::path& path,
                          const CalibrationResult& result) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create calibration report: " + path.string());
    out << std::setprecision(12) << "{\n"
        << "  \"machine\": {\n"
        << "    \"sid_mm\": " << result.machine.sid_mm << ",\n"
        << "    \"sdd_mm\": " << result.machine.sdd_mm << ",\n"
        << "    \"offset_u_mm\": " << result.machine.offset_u_mm << ",\n"
        << "    \"offset_v_mm\": " << result.machine.offset_v_mm << ",\n"
        << "    \"tilt_u_rad\": " << result.machine.tilt_u_rad << ",\n"
        << "    \"tilt_v_rad\": " << result.machine.tilt_v_rad << ",\n"
        << "    \"tilt_n_rad\": " << result.machine.tilt_n_rad << ",\n"
        << "    \"source_offset_mm\": [0, 0, 0],\n"
        << "    \"offset_n_mm\": 0\n"
        << "  },\n"
        << "  \"phantom_rotation_vector_rad\": ["
        << result.phantom_rotation_vector_rad.x() << ", "
        << result.phantom_rotation_vector_rad.y() << ", "
        << result.phantom_rotation_vector_rad.z() << "],\n"
        << "  \"phantom_translation_mm\": ["
        << result.phantom_translation_mm.x() << ", "
        << result.phantom_translation_mm.y() << ", "
        << result.phantom_translation_mm.z() << "],\n"
        << "  \"source_circle\": {\n"
        << "    \"center_phantom_mm\": [" << result.source_circle.center_phantom_mm.x() << ", "
        << result.source_circle.center_phantom_mm.y() << ", "
        << result.source_circle.center_phantom_mm.z() << "],\n"
        << "    \"axis_phantom\": [" << result.source_circle.axis_phantom.x() << ", "
        << result.source_circle.axis_phantom.y() << ", "
        << result.source_circle.axis_phantom.z() << "],\n"
        << "    \"radius_mm\": " << result.source_circle.radius_mm << ",\n"
        << "    \"radial_rms_mm\": " << result.source_circle.radial_rms_mm << ",\n"
        << "    \"axial_rms_mm\": " << result.source_circle.axial_rms_mm << "\n"
        << "  },\n"
        << "  \"joint_fit\": {\n"
        << "    \"rmse_px\": " << result.joint_rmse_px << ",\n"
        << "    \"max_point_error_px\": " << result.max_point_error_px << ",\n"
        << "    \"rotation_direction\": " << result.rotation_direction << ",\n"
        << "    \"iterations\": " << result.joint_iterations << ",\n"
        << "    \"converged\": " << (result.joint_converged ? "true" : "false") << "\n"
        << "  },\n"
        << "  \"per_view\": {\n"
        << "    \"dlt_rmse_px\": [";
    for (std::size_t i = 0; i < result.tracking.dlt_rmse_px.size(); ++i) {
        if (i) out << ", ";
        out << result.tracking.dlt_rmse_px[i];
    }
    out << "],\n    \"source_phantom_mm\": [";
    for (std::size_t i = 0; i < result.cameras.size(); ++i) {
        if (i) out << ", ";
        const auto& source = result.cameras[i].source_phantom_mm;
        out << "[" << source.x() << ", " << source.y() << ", " << source.z() << "]";
    }
    out << "],\n    \"sdd_mm\": [";
    for (std::size_t i = 0; i < result.cameras.size(); ++i) {
        if (i) out << ", ";
        out << result.cameras[i].sdd_mm;
    }
    out << "],\n    \"principal_point_px\": [";
    for (std::size_t i = 0; i < result.cameras.size(); ++i) {
        if (i) out << ", ";
        const auto& principal = result.cameras[i].principal_point_px;
        out << "[" << principal.x() << ", " << principal.y() << "]";
    }
    out << "]\n  }\n}\n";
}

}  // namespace cbct::calibration
