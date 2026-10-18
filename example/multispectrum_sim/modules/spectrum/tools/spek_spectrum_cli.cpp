#include "spek_spectrum/spectrum.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: spek_spectrum_cli <spekcalc|spekpy-v1> <output.spec>\n";
    return 2;
  }
  spek_spectrum::Config config;
  config.model = std::string(argv[1]) == "spekcalc"
                     ? spek_spectrum::Model::spekcalc
                     : spek_spectrum::Model::spekpy_v1;
  config.filters = {{"Al", 2.0}};
  const auto result = spek_spectrum::generate(config);
  spek_spectrum::write_mcgpu_spec(result, config, argv[2]);
  std::cout << spek_spectrum::model_name(config.model)
            << ": mean=" << result.mean_energy_keV
            << " keV, kerma=" << result.air_kerma_uGy
            << " uGy/mAs, Al HVL=" << result.hvl1_mm_al << " mm\n";
}
