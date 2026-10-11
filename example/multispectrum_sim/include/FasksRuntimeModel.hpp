#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace yk::spectral {

class FasksRuntimeModel {
public:
    explicit FasksRuntimeModel(const std::filesystem::path& path);

    std::vector<float> makeDiscreteKernel(double transmission,
                                          double pixel_u_mm,
                                          double pixel_v_mm,
                                          double support_u_mm,
                                          double support_v_mm) const;
    int discreteKernelCount() const {
        return static_cast<int>(discrete_attenuation_nodes_.size());
    }
    double discreteAttenuationNode(int index) const {
        return discrete_attenuation_nodes_.at(static_cast<size_t>(index));
    }
    double amplitude(double transmission) const;
    const std::vector<double>& amplitudeAttenuationNodes() const { return discrete_attenuation_nodes_; }
    const std::vector<double>& amplitudeValues() const { return discrete_amplitude_nodes_; }
    double wetFromLogAttenuation(double log_attenuation) const;

    double transmissionMin() const { return transmission_min_; }
    double transmissionMax() const { return transmission_max_; }
    double supportUMm() const { return support_u_mm_; }
    double supportVMm() const { return support_v_mm_; }
    double gammaPerMm() const { return gamma_per_mm_; }
    double wetSmoothSigmaMm() const { return wet_smooth_sigma_mm_; }
    double edgeBoundaryScale() const { return edge_boundary_scale_; }
    double edgeRampMm() const { return edge_ramp_mm_; }
    const std::vector<double>& wetAttenuationNodes() const {
        return wet_attenuation_nodes_;
    }
    const std::vector<double>& wetThicknessNodes() const {
        return wet_thickness_nodes_;
    }
    const std::vector<double>& wetSlopes() const { return wet_slopes_; }
    int detectorKernelWidth() const { return detector_kernel_width_; }
    int detectorKernelHeight() const { return detector_kernel_height_; }
    const std::vector<float>& detectorKernel() const { return detector_kernel_; }

private:
    static std::vector<double> pchipSlopes(const std::vector<double>& x,
                                           const std::vector<double>& y);
    static double evaluatePchip(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const std::vector<double>& slopes,
                                double value);
    double transmission_min_ = 0.0;
    double transmission_max_ = 1.0;
    double support_u_mm_ = 0.0;
    double support_v_mm_ = 0.0;
    double gamma_per_mm_ = 0.0;
    double wet_smooth_sigma_mm_ = 0.0;
    double edge_boundary_scale_ = 1.0;
    double edge_ramp_mm_ = 1.0;
    std::vector<double> discrete_attenuation_nodes_;
    std::vector<double> discrete_radius_mm_;
    std::vector<std::vector<double>> discrete_density_nodes_;
    std::vector<double> discrete_amplitude_nodes_;
    std::vector<double> wet_attenuation_nodes_;
    std::vector<double> wet_thickness_nodes_;
    std::vector<double> wet_slopes_;
    int detector_kernel_width_ = 0;
    int detector_kernel_height_ = 0;
    std::vector<float> detector_kernel_;
};

}
