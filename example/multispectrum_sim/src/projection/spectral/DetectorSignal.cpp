#include "DetectorSignal.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace yk::spectral::primary_detail {

void writeMetadata(const std::filesystem::path& file, int u, int v, int views, const char* signal) {
    std::ofstream f(file.string()+".json");
    if (!f) throw std::runtime_error("cannot create projection metadata");
    f << "{\n  \"columns\": " << u << ",\n  \"rows\": " << v
      << ",\n  \"frames\": " << views
      << ",\n  \"data_type\": \"float32\",\n  \"byte_order\": \"little_endian\",\n"
      << "  \"layout\": \"frame_row_column\",\n  \"signal\": \"" << signal << "\"\n}\n";
}
std::vector<float> loadLayerThickness(const DetectorLayerConfig& layer, int pixels) {
    if (pixels <= 0)
        throw std::runtime_error("detector thickness map requires positive pixel count");
    if (layer.thickness_map_file.empty() &&
        (!std::isfinite(layer.thickness_mm) || layer.thickness_mm < 0.0 ||
         layer.thickness_mm > std::numeric_limits<float>::max()))
        throw std::runtime_error("uniform detector thickness must be finite and non-negative");
    std::vector<float> thickness(static_cast<size_t>(pixels), static_cast<float>(layer.thickness_mm));
    if (layer.thickness_map_file.empty()) return thickness;
    std::ifstream input(layer.thickness_map_file, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open detector thickness map: " + layer.thickness_map_file.string());
    input.read(reinterpret_cast<char*>(thickness.data()),
        static_cast<std::streamsize>(thickness.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(thickness.size() * sizeof(float)) ||
        input.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("detector thickness map must contain exactly detector_u * detector_v float32 values");
    for (float value : thickness)
        if (!std::isfinite(value) || value < 0.f)
            throw std::runtime_error("detector thickness map contains a negative or non-finite value");
    return thickness;
}

void convolveSeparable(std::vector<float>& image, int width, int height,
                       const std::vector<double>& kernel_u,
                       const std::vector<double>& kernel_v) {
    std::vector<float> temporary(image.size());
    const int radius_u = static_cast<int>(kernel_u.size() / 2);
    for (int v=0; v<height; ++v) for (int u=0; u<width; ++u) {
        double value = 0.0, weight = 0.0;
        for (int k=-radius_u; k<=radius_u; ++k) {
            const int source_u = u + k;
            if (source_u < 0 || source_u >= width) continue;
            const double w = kernel_u[static_cast<size_t>(k + radius_u)];
            value += w * image[static_cast<size_t>(v*width + source_u)];
            weight += w;
        }
        temporary[static_cast<size_t>(v*width + u)] = static_cast<float>(value / std::max(weight, 1e-30));
    }
    const int radius_v = static_cast<int>(kernel_v.size() / 2);
    for (int v=0; v<height; ++v) for (int u=0; u<width; ++u) {
        double value = 0.0, weight = 0.0;
        for (int k=-radius_v; k<=radius_v; ++k) {
            const int source_v = v + k;
            if (source_v < 0 || source_v >= height) continue;
            const double w = kernel_v[static_cast<size_t>(k + radius_v)];
            value += w * temporary[static_cast<size_t>(source_v*width + u)];
            weight += w;
        }
        image[static_cast<size_t>(v*width + u)] = static_cast<float>(value / std::max(weight, 1e-30));
}
}
void applyCrosstalkSignal(std::vector<float>& signal, int width, int height,
                          const CrosstalkConfig& config) {
    if (!config.enabled) return;
    convolveSeparable(signal, width, height, config.kernel_u, config.kernel_v);
}

void applyAfterglowSignal(std::vector<float>& signal,
                          const DetectorPostprocessConfig::AfterglowConfig& config,
                          std::vector<std::vector<float>>& state) {
    if (!config.enabled) return;
    const bool initialize_steady_state = state.empty() &&
        config.initialization == "steady_state";
    if (state.empty())
        state.assign(config.weights.size(), std::vector<float>(signal.size(), 0.f));
    if (initialize_steady_state) {
        const float exposure = static_cast<float>(config.exposure_time_ms);
        const float readout = static_cast<float>(config.frame_time_ms - config.exposure_time_ms);
        for (size_t component=0; component<config.weights.size(); ++component) {
            const float tau = static_cast<float>(config.time_constants_ms[component]);
            const float de = std::exp(-exposure / tau);
            const float dr = std::exp(-readout / tau);
            const float cycle_decay = de * dr;
            const float input_scale = static_cast<float>(config.p) *
                static_cast<float>(config.weights[component]) * tau * (1.f - de) / exposure;
            const float steady_scale = dr * input_scale / std::max(1.f - cycle_decay, 1e-6f);
            for (size_t p=0; p<signal.size(); ++p)
                state[component][p] = steady_scale * signal[p];
        }
    }
    std::vector<float> output(signal.size(), static_cast<float>(0.0));
    const float prompt = static_cast<float>(1.0 - config.p);
    for (size_t p=0; p<signal.size(); ++p) output[p] = prompt * signal[p];
    for (size_t component=0; component<config.weights.size(); ++component) {
        const float exposure = static_cast<float>(config.exposure_time_ms);
        const float readout = static_cast<float>(
            config.frame_time_ms - config.exposure_time_ms);
        const float weight = static_cast<float>(config.weights[component]);
        const float tau = static_cast<float>(config.time_constants_ms[component]);
        const float exposure_decay = std::exp(-exposure / tau);
        const float readout_decay = std::exp(-readout / tau);
        const float released_fraction = 1.f - exposure_decay;
        for (size_t p=0; p<signal.size(); ++p) {
            const float input_trapped = static_cast<float>(config.p) * weight * signal[p];
            output[p] += state[component][p] * released_fraction
                + input_trapped * (1.f - tau * released_fraction / exposure);
            const float state_after_exposure = state[component][p] * exposure_decay
                + input_trapped * tau * released_fraction / exposure;
            state[component][p] = state_after_exposure * readout_decay;
        }
    }
    signal.swap(output);
}

void applyDasSignal(std::vector<float>& signal,
                    const DetectorPostprocessConfig::DasConfig& config,
                    std::mt19937& random) {
    if (!config.enabled) return;
    std::normal_distribution<double> noise(0.0, config.electronic_noise_std_electrons);
    const double gain = config.gain_electrons_per_keV;
    auto digitize = [&](double electrons) {
        if (config.saturation_electrons > 0)
            electrons = std::min(electrons, config.saturation_electrons);
        if (config.adc_lsb_electrons > 0)
            electrons = std::round(electrons / config.adc_lsb_electrons) * config.adc_lsb_electrons;
        return electrons;
    };
    for (size_t p=0; p<signal.size(); ++p) {
        const double electrons = digitize(
            signal[p] * gain + config.offset_electrons + noise(random));
        signal[p] = static_cast<float>(electrons / gain);
    }
}

} // namespace yk::spectral::primary_detail
