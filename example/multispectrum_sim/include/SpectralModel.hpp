#pragma once
#include "AttenuationProvider.hpp"
#include <unordered_map>
namespace yk::spectral {
struct SpectralResult { double incident_intensity=0.0; double transmitted_intensity=0.0; double line_integral=0.0; };
class SpectralTransmissionModel {
public:
    SpectralTransmissionModel(const std::vector<MaterialSpec>&, const std::vector<SpectrumPoint>&,
                              const IMassAttenuationProvider&, const DetectorEffectsConfig&);
    SpectralResult simulate(const std::unordered_map<std::uint8_t,double>& material_path_cm) const;
    SpectralResult simulate(const std::vector<double>& material_path_cm) const;
    const std::vector<MaterialSpec>& materials() const { return materials_; }
    const std::vector<SpectrumPoint>& spectrum() const { return spectrum_; }
    const std::vector<double>& massAttenuation(std::uint8_t label) const {
        return mass_attenuation_.at(label);
    }
private:
    std::vector<MaterialSpec> materials_; std::vector<SpectrumPoint> spectrum_;
    const IMassAttenuationProvider& attenuation_; DetectorEffectsConfig effects_;
    std::unordered_map<std::uint8_t,std::vector<double>> mass_attenuation_;
};
}
