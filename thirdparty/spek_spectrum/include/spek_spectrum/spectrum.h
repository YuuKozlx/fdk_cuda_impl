#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace spek_spectrum {

enum class Model {
  spekcalc,
  spekpy_v1,
};

struct Filter {
  std::string material;
  double thickness_mm = 0.0;
  // Set both fields to use any elemental filter in the embedded Z=1..92
  // attenuation table. Named common filters leave them at zero.
  int atomic_number = 0;
  double density_g_cm3 = 0.0;
};

struct Config {
  Model model = Model::spekpy_v1;
  double kvp = 120.0;
  double anode_angle_deg = 12.0;
  double bin_width_keV = 1.0;
  double bin_shift = 0.0;
  double mas = 1.0;
  double distance_cm = 100.0;
  std::vector<Filter> filters;

  // Empty selects the table embedded in the library. A non-empty value is an
  // explicit override for research or custom table sets.
  std::filesystem::path data_file;
};

struct SpectrumResult {
  std::vector<double> energy_keV;
  std::vector<double> fluence_per_keV_cm2;
  std::vector<double> bremsstrahlung_per_keV_cm2;
  std::vector<double> characteristic_per_keV_cm2;
  double mean_energy_keV = 0.0;
  double air_kerma_uGy = 0.0;
  double hvl1_mm_al = 0.0;
};

// Generates a tungsten reflection-target spectrum. The differential fluence
// is evaluated at Config::distance_cm and includes Config::mas.
SpectrumResult generate(const Config& config);

// Writes MCGPU-SPEC 1.0. Values are normalized per mAs and geometrically
// rescaled to 100 cm, matching the format's declared units.
void write_mcgpu_spec(const SpectrumResult& spectrum,
                      const Config& config,
                      const std::filesystem::path& output);

const char* model_name(Model model) noexcept;

}  // namespace spek_spectrum
