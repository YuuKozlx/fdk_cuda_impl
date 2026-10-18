#include "spek_spectrum/spectrum.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
  auto require_close = [](double actual, double expected, double relative, const char* name) {
    if (std::abs(actual - expected) / expected >= relative) {
      std::cerr << name << " mismatch: " << actual << " vs " << expected << '\n';
      std::exit(1);
    }
  };
  spek_spectrum::Config config;
  config.kvp = 120.0;
  config.model = spek_spectrum::Model::spekpy_v1;
  config.filters = {{"Al", 2.0}};
  if (argc > 1) config.data_file = argv[1];
  const auto result = spek_spectrum::generate(config);
  assert(result.energy_keV.size() == 119);
  assert(result.fluence_per_keV_cm2.size() == result.energy_keV.size());
  require_close(result.mean_energy_keV, 54.001898, 5.0e-4, "spekpy-v1 mean");
  require_close(result.air_kerma_uGy, 187.384325, 5.0e-4, "spekpy-v1 kerma");
  require_close(result.hvl1_mm_al, 3.773724, 5.0e-4, "spekpy-v1 hvl");
  std::cout << std::setprecision(12)
            << "spekpy-v1 mean=" << result.mean_energy_keV << " kerma=" << result.air_kerma_uGy << " hvl=" << result.hvl1_mm_al << '\n';

  config.model = spek_spectrum::Model::spekcalc;
  const auto legacy = spek_spectrum::generate(config);
  require_close(legacy.mean_energy_keV, 53.334190, 5.0e-4, "spekcalc mean");
  require_close(legacy.air_kerma_uGy, 149.397554, 5.0e-4, "spekcalc kerma");
  require_close(legacy.hvl1_mm_al, 3.626526, 5.0e-4, "spekcalc hvl");
  std::cout << "spekcalc mean=" << legacy.mean_energy_keV << " kerma=" << legacy.air_kerma_uGy << " hvl=" << legacy.hvl1_mm_al << '\n';
}
