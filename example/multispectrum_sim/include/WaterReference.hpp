#pragma once
#include "config/SimConfig.hpp"
namespace yk::spectral {
struct WaterReference {double energy_keV=0;double mu_water_mm_inv=0;};
WaterReference resolveWaterReference(const SimulationConfig&,const BeamHardeningConfig&);
inline WaterReference resolveWaterReference(const SimulationConfig& c){return resolveWaterReference(c,c.reconstruction.preprocessing.beam_hardening);}
}
