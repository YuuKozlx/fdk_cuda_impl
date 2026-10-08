#include "FasksRuntimeModel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace yk::spectral {
namespace {

double finiteNumber(const nlohmann::json& value) {
    const double result = value.get<double>();
    if (!std::isfinite(result))
        throw std::runtime_error("non-finite fASKS model value");
    return result;
}

double endpointSlope(double h0, double h1, double d0, double d1) {
    double slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (slope * d0 <= 0.0) return 0.0;
    if (d0 * d1 < 0.0 && std::abs(slope) > 3.0 * std::abs(d0))
        return 3.0 * d0;
    return slope;
}

void requireStrictlyIncreasing(const std::vector<double>& values,
                               const char* message) {
    if (values.size() < 2)
        throw std::runtime_error(message);
    for (size_t i = 1; i < values.size(); ++i)
        if (!(values[i] > values[i - 1]))
            throw std::runtime_error(message);
}

}

std::vector<double> FasksRuntimeModel::pchipSlopes(
    const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2)
        throw std::runtime_error("invalid fASKS PCHIP nodes");
    std::vector<double> h(x.size() - 1), d(h.size()), slopes(x.size());
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = x[i + 1] - x[i];
        if (!(h[i] > 0.0))
            throw std::runtime_error("fASKS PCHIP nodes must increase");
        d[i] = (y[i + 1] - y[i]) / h[i];
    }
    if (x.size() == 2) {
        slopes[0] = slopes[1] = d[0];
        return slopes;
    }
    slopes.front() = endpointSlope(h[0], h[1], d[0], d[1]);
    const size_t last = h.size() - 1;
    slopes.back() = endpointSlope(h[last], h[last - 1], d[last], d[last - 1]);
    for (size_t i = 1; i + 1 < x.size(); ++i) {
        if (d[i - 1] * d[i] <= 0.0) {
            slopes[i] = 0.0;
        } else {
            const double w1 = 2.0 * h[i] + h[i - 1];
            const double w2 = h[i] + 2.0 * h[i - 1];
            slopes[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
        }
    }
    return slopes;
}

double FasksRuntimeModel::evaluatePchip(
    const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& slopes, double value) {
    value = std::clamp(value, x.front(), x.back());
    const auto upper = std::upper_bound(x.begin(), x.end(), value);
    const size_t i = std::min(
        static_cast<size_t>(std::max<std::ptrdiff_t>(upper - x.begin() - 1, 0)),
        x.size() - 2);
    const double h = x[i + 1] - x[i];
    const double t = (value - x[i]) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
         + (t3 - 2.0 * t2 + t) * h * slopes[i]
         + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
         + (t3 - t2) * h * slopes[i + 1];
}

FasksRuntimeModel::FasksRuntimeModel(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open fASKS runtime model: " + path.string());
    const auto model = nlohmann::json::parse(stream);
    if (model.at("format") != "fasks-runtime-model/v4" ||
        model.at("schema_version").get<int>() != 4)
        throw std::runtime_error("scatter requires fasks-runtime-model/v4");
    if (model.at("signal_domain") != "energy_deposition_per_pixel")
        throw std::runtime_error("fASKS model signal domain must be energy deposition per pixel");

    const auto& validity = model.at("validity");
    transmission_min_ = finiteNumber(validity.at("transmission_min"));
    transmission_max_ = finiteNumber(validity.at("transmission_max"));
    if (!(transmission_min_ > 0.0 && transmission_min_ < transmission_max_ &&
          transmission_max_ < 1.0))
        throw std::runtime_error("invalid fASKS transmission range");

    const auto& sampling = model.at("kernel_sampling");
    if (sampling.at("representation") != "discrete_radial_table")
        throw std::runtime_error("unsupported fASKS kernel representation");
    support_u_mm_ = finiteNumber(sampling.at("support_u_mm"));
    support_v_mm_ = finiteNumber(sampling.at("support_v_mm"));
    if (!(support_u_mm_ > 0.0 && support_v_mm_ > 0.0))
        throw std::runtime_error("fASKS kernel support must be positive");

    const auto& amplitude = model.at("center_kernel").at("amplitude");
    if (amplitude.at("interpolation") != "linear-log-attenuation")
        throw std::runtime_error("unsupported fASKS amplitude interpolation");
    for (const auto& node : amplitude.at("log_attenuation_nodes"))
        discrete_attenuation_nodes_.push_back(finiteNumber(node));
    for (const auto& node : amplitude.at("values"))
        discrete_amplitude_nodes_.push_back(finiteNumber(node));
    requireStrictlyIncreasing(discrete_attenuation_nodes_,
                              "fASKS discrete attenuation nodes must increase");
    if (discrete_amplitude_nodes_.size() != discrete_attenuation_nodes_.size())
        throw std::runtime_error("fASKS amplitude node count mismatch");

    const auto& shape = model.at("center_kernel").at("shape");
    if (shape.at("family") != "discrete-radial-kernel")
        throw std::runtime_error("unsupported fASKS kernel family");
    for (const auto& node : shape.at("radial_positions_mm"))
        discrete_radius_mm_.push_back(finiteNumber(node));
    if (discrete_radius_mm_.size() != 256 || discrete_radius_mm_.front() != 0.0)
        throw std::runtime_error("fASKS discrete radial grid must contain 256 samples");
    requireStrictlyIncreasing(discrete_radius_mm_,
                              "fASKS radial positions must increase");
    for (const auto& row : shape.at("radial_density_nodes_mm2")) {
        std::vector<double> density;
        for (const auto& value : row) {
            const double sample = finiteNumber(value);
            if (sample < 0.0) throw std::runtime_error("negative fASKS radial density");
            density.push_back(sample);
        }
        if (density.size() != discrete_radius_mm_.size())
            throw std::runtime_error("fASKS radial density row size mismatch");
        discrete_density_nodes_.push_back(std::move(density));
    }
    if (discrete_density_nodes_.size() != discrete_attenuation_nodes_.size())
        throw std::runtime_error("fASKS radial density node count mismatch");

    const auto& wet = model.at("water_equivalent_thickness");
    if (wet.at("interpolation") != "pchip")
        throw std::runtime_error("fASKS WET interpolation must be pchip");
    for (const auto& node : wet.at("log_attenuation_nodes"))
        wet_attenuation_nodes_.push_back(finiteNumber(node));
    for (const auto& node : wet.at("thickness_mm_nodes"))
        wet_thickness_nodes_.push_back(finiteNumber(node));
    requireStrictlyIncreasing(wet_attenuation_nodes_,
                              "fASKS WET attenuation nodes must increase");
    if (wet_attenuation_nodes_.size() != wet_thickness_nodes_.size())
        throw std::runtime_error("fASKS WET node count mismatch");
    wet_slopes_ = pchipSlopes(wet_attenuation_nodes_, wet_thickness_nodes_);

    const auto& fasks = model.at("fasks");
    gamma_per_mm_ = finiteNumber(fasks.at("gamma_per_mm"));
    wet_smooth_sigma_mm_ = finiteNumber(fasks.at("wet_smooth_sigma_mm"));
    if (gamma_per_mm_ < 0.0 || wet_smooth_sigma_mm_ < 0.0)
        throw std::runtime_error("fASKS gamma and WET smoothing must be nonnegative");

    const auto& edge = model.at("edge_amplitude_window");
    if (edge.at("model") != "paper-linear-distance-to-boundary")
        throw std::runtime_error("unsupported fASKS edge window");
    edge_boundary_scale_ = finiteNumber(edge.at("boundary_scale"));
    edge_ramp_mm_ = finiteNumber(edge.at("ramp_mm"));
    if (!(edge_boundary_scale_ >= 0.0 && edge_boundary_scale_ <= 1.0 &&
          edge_ramp_mm_ > 0.0))
        throw std::runtime_error("invalid fASKS edge window");

    const auto& response = model.at("detector_response");
    if (response.at("model") != "air-pencil-beam-scatter-per-primary")
        throw std::runtime_error("unsupported fASKS detector response");
    const auto& rows = response.at("scatter_kernel_per_primary");
    detector_kernel_height_ = static_cast<int>(rows.size());
    if (detector_kernel_height_ <= 0 || detector_kernel_height_ % 2 == 0)
        throw std::runtime_error("fASKS detector kernel must be odd-sized");
    detector_kernel_width_ = static_cast<int>(rows.at(0).size());
    if (detector_kernel_width_ <= 0 || detector_kernel_width_ % 2 == 0)
        throw std::runtime_error("fASKS detector kernel must be odd-sized");
    detector_kernel_.reserve(
        static_cast<size_t>(detector_kernel_width_) * detector_kernel_height_);
    for (const auto& row : rows) {
        if (static_cast<int>(row.size()) != detector_kernel_width_)
            throw std::runtime_error("fASKS detector kernel must be rectangular");
        for (const auto& item : row) {
            const double value = finiteNumber(item);
            if (value < 0.0)
                throw std::runtime_error("fASKS detector kernel must be nonnegative");
            detector_kernel_.push_back(static_cast<float>(value));
        }
    }
}

double FasksRuntimeModel::amplitude(double transmission) const {
    if (!std::isfinite(transmission))
        throw std::runtime_error("invalid fASKS transmission");
    const double physical_t = std::clamp(
        transmission, std::numeric_limits<double>::min(), 1.0);
    if (physical_t >= 1.0) return 0.0;
    const double x = std::log(1.0 / physical_t);
    const auto upper = std::upper_bound(discrete_attenuation_nodes_.begin(),
                                        discrete_attenuation_nodes_.end(),
                                        std::clamp(x, discrete_attenuation_nodes_.front(),
                                                    discrete_attenuation_nodes_.back()));
    const size_t i = std::min(static_cast<size_t>(
        std::max<std::ptrdiff_t>(upper - discrete_attenuation_nodes_.begin() - 1, 0)),
        discrete_attenuation_nodes_.size() - 2);
    const double fraction = (std::clamp(x, discrete_attenuation_nodes_.front(),
                                        discrete_attenuation_nodes_.back())
                             - discrete_attenuation_nodes_[i])
                            / (discrete_attenuation_nodes_[i + 1]
                               - discrete_attenuation_nodes_[i]);
    double result = discrete_amplitude_nodes_[i]
                  + fraction * (discrete_amplitude_nodes_[i + 1]
                                - discrete_amplitude_nodes_[i]);
    if (x < discrete_attenuation_nodes_.front()) {
        const double first_x = discrete_attenuation_nodes_.front();
        const double thin_yield = transmission_max_ * discrete_amplitude_nodes_.front()
            * x / first_x;
        result = thin_yield / physical_t;
    }
    if (!std::isfinite(result) || result < 0.0)
        throw std::runtime_error("invalid evaluated fASKS discrete amplitude");
    return result;
}

std::vector<float> FasksRuntimeModel::makeDiscreteKernel(
    double transmission, double pixel_u_mm, double pixel_v_mm,
    double support_u_mm, double support_v_mm) const {
    const int radius_u = std::max(1, static_cast<int>(std::ceil(support_u_mm / pixel_u_mm)));
    const int radius_v = std::max(1, static_cast<int>(std::ceil(support_v_mm / pixel_v_mm)));
    const int width = 2 * radius_u + 1;
    const int height = 2 * radius_v + 1;
    const double x = std::clamp(std::log(1.0 / std::clamp(
        transmission, std::numeric_limits<double>::min(), 1.0)),
        discrete_attenuation_nodes_.front(), discrete_attenuation_nodes_.back());
    const auto upper = std::upper_bound(discrete_attenuation_nodes_.begin(),
                                        discrete_attenuation_nodes_.end(), x);
    const size_t i = std::min(static_cast<size_t>(
        std::max<std::ptrdiff_t>(upper - discrete_attenuation_nodes_.begin() - 1, 0)),
        discrete_attenuation_nodes_.size() - 2);
    const double fraction = (x - discrete_attenuation_nodes_[i])
        / (discrete_attenuation_nodes_[i + 1] - discrete_attenuation_nodes_[i]);
    std::vector<float> kernel(static_cast<size_t>(width) * height, 0.0f);
    for (int y = -radius_v; y <= radius_v; ++y) {
        for (int x_pixel = -radius_u; x_pixel <= radius_u; ++x_pixel) {
            const double radius = std::hypot(y * pixel_v_mm, x_pixel * pixel_u_mm);
            const auto radial = std::lower_bound(discrete_radius_mm_.begin(),
                                                 discrete_radius_mm_.end(), radius);
            double value = 0.0;
            if (radial == discrete_radius_mm_.begin()) {
                value = (1.0 - fraction) * discrete_density_nodes_[i].front()
                      + fraction * discrete_density_nodes_[i + 1].front();
            } else if (radial != discrete_radius_mm_.end()) {
                const size_t r = static_cast<size_t>(radial - discrete_radius_mm_.begin());
                const double f = (radius - discrete_radius_mm_[r - 1])
                    / (discrete_radius_mm_[r] - discrete_radius_mm_[r - 1]);
                const double d0 = (1.0 - f) * discrete_density_nodes_[i][r - 1]
                                + f * discrete_density_nodes_[i][r];
                const double d1 = (1.0 - f) * discrete_density_nodes_[i + 1][r - 1]
                                + f * discrete_density_nodes_[i + 1][r];
                value = (1.0 - fraction) * d0 + fraction * d1;
            }
            kernel[static_cast<size_t>(y + radius_v) * width + x_pixel + radius_u]
                = static_cast<float>(value * pixel_u_mm * pixel_v_mm);
        }
    }
    const double mass = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    if (!(std::isfinite(mass) && mass > 0.0))
        throw std::runtime_error("invalid fASKS discrete kernel mass");
    // Keep the calibrated mass.  The image support is a runtime crop, so
    // renormalizing here would turn lost out-of-window scatter into extra
    // in-window scatter and make small detector windows inconsistent with
    // the saved calibration kernel.
    return kernel;
}

double FasksRuntimeModel::wetFromLogAttenuation(double log_attenuation) const {
    if (!std::isfinite(log_attenuation))
        throw std::runtime_error("invalid fASKS log attenuation");
    return evaluatePchip(wet_attenuation_nodes_, wet_thickness_nodes_,
                         wet_slopes_, log_attenuation);
}

}
