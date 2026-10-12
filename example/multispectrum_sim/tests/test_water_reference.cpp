#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include "../../../visualization/FloatVolume.hpp"
#include <cmath>
#include <iostream>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    yk::spectral::SimulationConfig c;
    auto& bh=c.reconstruction.preprocessing.beam_hardening;bh.reference_energy_keV=70;
    const auto reference=yk::spectral::resolveWaterReference(c);
    yk::spectral::XcomAttenuationProvider provider({});yk::spectral::MaterialSpec water{1,"water","H2O","",1};
    std::vector<double> mu;std::string error;require(provider.query(water,{70},mu,error),"water query failed");
    require(std::abs(reference.mu_water_mm_inv-mu[0]/10)<1e-12,"cm-to-mm reference conversion");
    require(std::abs(yk::visualization::attenuationToHu(reference.mu_water_mm_inv,reference.mu_water_mm_inv))<1e-12,"water is not 0 HU");
    require(yk::visualization::attenuationToHu(0,reference.mu_water_mm_inv)==-1000,"zero attenuation is not -1000 HU");
    require(yk::visualization::attenuationToHu(2*reference.mu_water_mm_inv,reference.mu_water_mm_inv)==1000,"HU ratio mismatch");
    bh.reference_energy_keV=0;c.projection.spectrum.source="generated";c.projection.spectrum.filters={{"Al",2}};
    const auto mean=yk::spectral::resolveWaterReference(c);
    require(std::abs(mean.energy_keV-54.001898)<.03,"BHC photon mean energy mismatch");
    bool rejected=false;try{yk::visualization::attenuationToHu(1,0);}catch(const std::exception&){rejected=true;}require(rejected,"invalid water reference accepted");
    std::cout<<"BHC reference, unit conversion and HU anchors passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
