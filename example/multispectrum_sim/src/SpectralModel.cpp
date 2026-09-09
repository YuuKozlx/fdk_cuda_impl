#include "SpectralModel.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace yk::spectral {
SpectralTransmissionModel::SpectralTransmissionModel(const std::vector<MaterialSpec>& m,const std::vector<SpectrumPoint>& s,const IMassAttenuationProvider& a,const DetectorEffectsConfig& e):materials_(m),spectrum_(s),attenuation_(a),effects_(e){std::vector<double> en;for(auto& p:s)en.push_back(p.energy_keV);for(auto& x:m){std::vector<double> v;std::string err;if(!a.query(x,en,v,err))throw std::runtime_error(x.name+": "+err);mass_attenuation_.emplace(x.label,std::move(v));}}
SpectralResult SpectralTransmissionModel::simulate(const std::unordered_map<std::uint8_t,double>& paths) const {SpectralResult r;for(auto& p:spectrum_)r.incident_intensity+=p.relative_photons;if(r.incident_intensity<=0)throw std::runtime_error("能谱总权重必须为正");for(size_t e=0;e<spectrum_.size();++e){double exponent=0;for(auto& m:materials_){auto i=paths.find(m.label);if(i!=paths.end())exponent+=i->second*m.density_g_cm3*mass_attenuation_.at(m.label)[e];}r.transmitted_intensity+=spectrum_[e].relative_photons*std::exp(-exponent);}if(effects_.efficiency_enabled){r.incident_intensity*=effects_.efficiency;r.transmitted_intensity*=effects_.efficiency;}r.line_integral=-std::log(std::max(r.transmitted_intensity,1e-30)/r.incident_intensity);return r;}
SpectralResult SpectralTransmissionModel::simulate(const std::vector<double>& paths) const {
    if (paths.size() != materials_.size()) throw std::runtime_error("材料路径数组长度不匹配");
    std::unordered_map<std::uint8_t, double> mapped;
    for (std::size_t i = 0; i < materials_.size(); ++i) mapped.emplace(materials_[i].label, paths[i]);
    return simulate(mapped);
}
}
