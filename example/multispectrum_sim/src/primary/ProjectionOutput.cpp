#include "ProjectionOutput.hpp"
#include "DetectorSignal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yk::spectral::primary_detail {
namespace {
void openOutput(std::ofstream& stream, const std::filesystem::path& path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    stream.open(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open projection output: " + path.string());
}
}
void EnergyProjectionOutput::prepare(const SimulationConfig& c,
    const std::filesystem::path& output) {
    reset();
    config_ = c.projection; output_ = output;
    width_ = c.geometry.parameters.detector_u; height_ = c.geometry.parameters.detector_v;
    views_ = c.geometry.parameters.views;
    attenuation_buffer_.resize(static_cast<size_t>(width_) * height_);
    openOutput(attenuation_, output_);
    if (!config_.energy_output_file.empty()) openOutput(energy_, config_.energy_output_file);
    if (!config_.air_output_file.empty()) openOutput(air_, config_.air_output_file);
}
void EnergyProjectionOutput::writeBuffer(std::ofstream& stream,
    const std::vector<float>& buffer) {
    stream.write(reinterpret_cast<const char*>(buffer.data()),
        static_cast<std::streamsize>(buffer.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("projection output write failed");
}
void EnergyProjectionOutput::write(const ProjectionFrame& frame) {
    if (frame.domain != ProjectionSignalDomain::EnergyKeV ||
        frame.location != ProjectionMemoryLocation::Host ||
        frame.width != width_ || frame.height != height_ || frame.view != written_ ||
        frame.signal_energy.size() != attenuation_buffer_.size() ||
        frame.reference_energy.size() != attenuation_buffer_.size())
        throw std::runtime_error("invalid energy projection output frame");
    for (size_t p = 0; p < attenuation_buffer_.size(); ++p)
        attenuation_buffer_[p] = -std::log(std::max(frame.signal_energy[p] /
            std::max(frame.reference_energy[p], 1e-30f), 1e-30f));
    if (air_.is_open()) writeBuffer(air_, frame.reference_energy);
    writeBuffer(attenuation_, attenuation_buffer_);
    if (energy_.is_open()) writeBuffer(energy_, frame.signal_energy);
    ++written_;
}
void EnergyProjectionOutput::finish() {
    if (written_ != views_) throw std::runtime_error("incomplete projection output");
    auto close = [](std::ofstream& stream) {
        if (!stream.is_open()) return;
        stream.close();
        if (stream.fail()) throw std::runtime_error("projection output close failed");
    };
    close(attenuation_); close(energy_); close(air_);
    if (config_.save_metadata)
        writeMetadata(output_, width_, height_, views_, "air_corrected_negative_log_attenuation");
    if (!config_.energy_output_file.empty())
        writeMetadata(config_.energy_output_file, width_, height_, views_,
            "detected_energy_keV_per_pixel_per_view");
    if (!config_.air_output_file.empty())
        writeMetadata(config_.air_output_file, width_, height_, views_,
            "processed_air_energy_keV_per_pixel_per_view");
}
void EnergyProjectionOutput::reset() noexcept {
    if (attenuation_.is_open()) attenuation_.close();
    if (energy_.is_open()) energy_.close();
    if (air_.is_open()) air_.close();
    attenuation_.clear(); energy_.clear(); air_.clear(); written_ = 0;
}
} // namespace yk::spectral::primary_detail
