#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include <cmath>
#include <stdexcept>
namespace yk::spectral {
WaterReference resolveWaterReference(const SimulationConfig& config,const BeamHardeningConfig& bh){
    double energy=bh.reference_energy_keV;
    if(!std::isfinite(energy)||energy<0)throw std::runtime_error("invalid BHC reference energy");
    if(energy==0){
        auto projection=config.projection;projection.spectrum.export_file.clear();
        const auto spectrum=bh.spectrum_file.empty()?loadProjectionSpectrum(projection):loadSpectrum(bh.spectrum_file);
        double photons=0,weighted=0;
        for(const auto& p:spectrum){const double weight=p.fluence_per_keV_cm2_mAs_at_1m*p.bin_width_keV;photons+=weight;weighted+=weight*p.energy_keV;}
        if(!std::isfinite(photons)||photons<=0||!std::isfinite(weighted)||weighted<=0)throw std::runtime_error("invalid reference spectrum");energy=weighted/photons;
    }
    MaterialSpec water;water.label=1;water.name="water";water.formula="H2O";water.density_g_cm3=1;
    XcomAttenuationProvider attenuation({});std::vector<double> mass;std::string error;
    if(!attenuation.query(water,{energy},mass,error)||mass.empty()||!std::isfinite(mass[0])||mass[0]<=0)
        throw std::runtime_error("reference water attenuation query failed: "+error);
    // XCOM cm2/g * water 1 g/cm3 -> cm^-1; scanner/reconstruction geometry uses mm.
    return {energy,mass[0]/10.0};
}
}
