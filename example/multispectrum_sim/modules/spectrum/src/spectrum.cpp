#include "spek_spectrum/spectrum.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>

namespace spek_spectrum::embedded {
extern const std::uint8_t kData[];
extern const std::size_t kDataSize;
}  // namespace spek_spectrum::embedded

namespace spek_spectrum {
namespace {

constexpr double kElectronMass = 511.0;
constexpr double kFineStructure = 1.0 / 137.0;
constexpr double kElectronRadiusCm = 2.81794e-13;
constexpr double kAvogadro = 6.02e23;
constexpr double kElectronCharge = 1.602e-19;
constexpr double kDensityTungsten = 19.3;
constexpr double kAtomicWeightTungsten = 183.8;
constexpr double kDensityAir = 1.205e-3;
constexpr double kGeometryFactor = 1.0 / (4.0 * 3.14159265358979323846);
constexpr double kKeVToJ = 1.6022e-16;
constexpr double kGyToUGy = 1.0e6;

struct Section {
  std::uint64_t offset = 0;
  std::uint64_t count = 0;
};

struct Tables {
  std::ifstream stream;
  const std::uint8_t* memory;
  std::size_t memory_size;
  std::map<std::string, Section> sections;
  std::map<std::string, std::vector<double>> cache;
  std::uint32_t version_ = 0;

  explicit Tables(const std::filesystem::path& file)
      : stream(file, std::ios::binary), memory(nullptr), memory_size(0) {
    if (!stream) throw std::runtime_error("cannot open SpekPy table file: " + file.string());
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || std::string_view(magic.data(), magic.size()) != "SPEKDATA")
      throw std::runtime_error("invalid SpekPy table magic");
    std::uint32_t version = 0, count = 0;
    stream.read(reinterpret_cast<char*>(&version), sizeof(version));
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream) throw std::runtime_error("truncated SpekPy table header");
    parse_sections(version, count, nullptr, 0);
  }

  Tables(const std::uint8_t* data, std::size_t size) : memory(data), memory_size(size) {
    parse_header(memory, memory_size);
  }

  void parse_header(const std::uint8_t* data, std::size_t size) {
    if (size < 16 || std::string_view(reinterpret_cast<const char*>(data), 8) != "SPEKDATA")
      throw std::runtime_error("invalid SpekPy table magic");
    std::uint32_t version = 0, count = 0;
    if (memory != nullptr) {
      std::memcpy(&version, data + 8, sizeof(version));
      std::memcpy(&count, data + 12, sizeof(count));
      parse_sections(version, count, data + 16, size - 16);
    } else {
      stream.read(reinterpret_cast<char*>(&version), sizeof(version));
      stream.read(reinterpret_cast<char*>(&count), sizeof(count));
      if (!stream) throw std::runtime_error("truncated SpekPy table header");
      parse_sections(version, count, nullptr, 0);
    }
  }

  void parse_sections(std::uint32_t version, std::uint32_t count,
                      const std::uint8_t* data, std::size_t size) {
    if ((version < 1 || version > 3) || count > 10000)
      throw std::runtime_error("unsupported SpekPy table version");
    version_ = version;
    constexpr std::size_t kSectionSize = 40;
    if (data != nullptr && size < static_cast<std::size_t>(count) * kSectionSize)
      throw std::runtime_error("truncated SpekPy table header");
    for (std::uint32_t i = 0; i < count; ++i) {
      char name[24]{};
      Section section;
      if (data != nullptr) {
        const auto* entry = data + static_cast<std::size_t>(i) * kSectionSize;
        std::copy(entry, entry + sizeof(name), reinterpret_cast<std::uint8_t*>(name));
        std::memcpy(&section.offset, entry + 24, sizeof(section.offset));
        std::memcpy(&section.count, entry + 32, sizeof(section.count));
      } else {
        stream.read(name, sizeof(name));
        stream.read(reinterpret_cast<char*>(&section.offset), sizeof(section.offset));
        stream.read(reinterpret_cast<char*>(&section.count), sizeof(section.count));
        if (!stream) throw std::runtime_error("truncated SpekPy table header");
      }
      sections.emplace(std::string(name, name + std::char_traits<char>::length(name)), section);
    }
  }

  void read_bytes(std::uint64_t offset, void* output, std::size_t bytes) {
    if (memory != nullptr) {
      if (offset > memory_size || bytes > memory_size - static_cast<std::size_t>(offset))
        throw std::runtime_error("truncated SpekPy table section");
      std::memcpy(output, memory + static_cast<std::size_t>(offset), bytes);
    } else {
      stream.seekg(static_cast<std::streamoff>(offset));
      stream.read(reinterpret_cast<char*>(output), static_cast<std::streamsize>(bytes));
      if (!stream) throw std::runtime_error("truncated SpekPy table section");
    }
  }

  const std::vector<double>& get(const std::string& name) {
    if (const auto cached = cache.find(name); cached != cache.end()) return cached->second;
    const auto it = sections.find(name);
    if (it == sections.end()) throw std::runtime_error("missing SpekPy table section: " + name);
    std::vector<double> values(it->second.count);
    const auto bytes = values.size() * (version_ == 2 ? sizeof(double) : sizeof(float));
    std::vector<float> legacy;
    if (version_ == 1 || version_ == 3) legacy.resize(values.size());
    void* destination = legacy.empty() ? static_cast<void*>(values.data()) : static_cast<void*>(legacy.data());
    read_bytes(it->second.offset, destination, bytes);
    if (!legacy.empty()) std::copy(legacy.begin(), legacy.end(), values.begin());
    auto [inserted, _] = cache.emplace(name, std::move(values));
    return inserted->second;
  }
};

double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

double linear(const std::vector<double>& x, const std::vector<double>& y, double q) {
  if (x.size() != y.size() || x.empty()) throw std::runtime_error("invalid interpolation table");
  if (q <= x.front()) return y.front();
  if (q >= x.back()) return y.back();
  const auto it = std::upper_bound(x.begin(), x.end(), q);
  const std::size_t i = static_cast<std::size_t>(it - x.begin() - 1);
  const double f = (q - x[i]) / (x[i + 1] - x[i]);
  return y[i] + f * (y[i + 1] - y[i]);
}

struct SplineBasis {
  std::size_t first = 0;
  double value[4]{};
};

SplineBasis cubic_bspline_basis(const std::vector<double>& knots, double q) {
  constexpr std::size_t degree = 3;
  const std::size_t coefficients = knots.size() - degree - 1;
  q = clamp(q, knots[degree], knots[coefficients]);
  std::size_t span = degree;
  if (q >= knots[coefficients]) {
    span = coefficients - 1;
  } else {
    const auto upper = std::upper_bound(knots.begin() + degree,
                                        knots.begin() + coefficients + 1, q);
    span = static_cast<std::size_t>(upper - knots.begin() - 1);
  }
  SplineBasis result;
  result.first = span - degree;
  result.value[0] = 1.0;
  double left[4]{}, right[4]{};
  for (std::size_t j = 1; j <= degree; ++j) {
    left[j] = q - knots[span + 1 - j];
    right[j] = knots[span + j] - q;
    double saved = 0.0;
    for (std::size_t r = 0; r < j; ++r) {
      const double denominator = right[r + 1] + left[j - r];
      const double term = denominator == 0.0 ? 0.0 : result.value[r] / denominator;
      result.value[r] = saved + right[r + 1] * term;
      saved = left[j - r] * term;
    }
    result.value[j] = saved;
  }
  return result;
}

double log_linear(const std::vector<double>& x, const std::vector<double>& y, double q) {
  std::vector<double> lx, ly;
  lx.reserve(x.size()); ly.reserve(y.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i] > 0.0 && y[i] > 0.0) { lx.push_back(std::log(x[i])); ly.push_back(std::log(y[i])); }
  }
  if (lx.empty() || q <= 0.0) return 0.0;
  return std::exp(linear(lx, ly, std::log(clamp(q, std::exp(lx.front()), std::exp(lx.back())))));
}

double simpson(const std::vector<double>& values, double dx) {
  if (values.size() < 2) return 0.0;
  if (values.size() == 2) return 0.5 * dx * (values[0] + values[1]);
  const std::size_t used = values.size() % 2 == 0 ? values.size() - 1 : values.size();
  double sum = values[0] + values[used - 1];
  for (std::size_t i = 1; i + 1 < used; ++i) sum += (i % 2 ? 4.0 : 2.0) * values[i];
  double result = sum * dx / 3.0;
  if (used != values.size()) {
    const std::size_t n = values.size();
    result += dx * (5.0 * values[n - 1] / 12.0 + 2.0 * values[n - 2] / 3.0 - values[n - 3] / 12.0);
  }
  return result;
}

double range_csda(Tables& tables, double energy_keV) {
  const auto& e_raw = tables.get("range_e");
  const auto& r = tables.get("range_r");
  std::vector<double> e = e_raw;
  for (double& value : e) value *= 1000.0;
  return linear(e, r, clamp(energy_keV, e.front(), e.back()));
}

double nearest_grid(double q, double step, std::size_t n) {
  return clamp(std::round(q / step), 0.0, static_cast<double>(n - 1));
}

double pe_value(Tables& tables, double e0, double t, double u) {
  const auto& pe = tables.get("pe");
  const auto iE = static_cast<std::size_t>(nearest_grid(e0, 10.0, 101));
  const auto it = static_cast<std::size_t>(nearest_grid(t, 0.01, 101));
  const auto iu = static_cast<std::size_t>(nearest_grid(u, 0.01, 101));
  return pe[(iE * 101 + it) * 101 + iu];
}

double ne_value(Tables& tables, double e0, double t) {
  const auto& ne = tables.get("ne");
  const auto iE = static_cast<std::size_t>(nearest_grid(e0, 10.0, 101));
  const auto it = static_cast<std::size_t>(nearest_grid(t, 0.01, 101));
  return ne[iE * 101 + it];
}

double mebh(double ei, double k) {
  const double ef = ei - k;
  if (ei <= 0.0 || ef < 0.0 || k <= 0.0) return 0.0;
  const double phi0 = 74.0 * 74.0 * kElectronRadiusCm * kElectronRadiusCm * kFineStructure;
  if (ef < 1.0)
    return phi0 * (2.0 / 3.0) * (1.0 + 8.0 * kElectronMass / k) / ei;
  const double ti = ei + kElectronMass;
  const double tf = ef + kElectronMass;
  const double qi = std::sqrt(std::max(0.0, ti * ti - kElectronMass * kElectronMass));
  const double qf = std::sqrt(std::max(0.0, tf * tf - kElectronMass * kElectronMass));
  if (qf <= 0.0) return 0.0;
  const double elwert = qi / qf;
  const double arg = (ti * tf + qi * qf - kElectronMass * kElectronMass) /
                     (kElectronMass * k);
  if (arg <= 0.0 || qi <= 0.0) return 0.0;
  const double l = 2.0 * std::log(arg);
  const double phi_value = (2.0 / 3.0) * (ei / (k * qi * qi)) *
                           (4.0 * ti * tf * l - 7.0 * qi * qf) * elwert;
  return phi0 * phi_value / ei;
}

double nist_brem(Tables& tables, double ei, double k) {
  const auto& tx = tables.get("brem_tx");
  const auto& ty = tables.get("brem_ty");
  const auto& coefficients = tables.get("brem_coeff");
  if (ei <= 0.0 || k <= 0.0 || k > ei) return 0.0;
  const auto bx = cubic_bspline_basis(tx, ei);
  const auto by = cubic_bspline_basis(ty, k / ei);
  const std::size_t ny = ty.size() - 4;
  double v = 0.0;
  for (std::size_t a = 0; a < 4; ++a) {
    for (std::size_t b = 0; b < 4; ++b) {
      v += bx.value[a] * by.value[b] *
           coefficients[(bx.first + a) * ny + by.first + b];
    }
  }
  const double total = ei + kElectronMass;
  const double beta2 = 1.0 - (kElectronMass * kElectronMass) / (total * total);
  return 1.0e-27 * (74.0 * 74.0 / (k * std::max(beta2, 1e-12))) * v;
}

struct MaterialData { int z; double density; };

MaterialData material(const Filter& filter) {
  if (filter.atomic_number != 0 || filter.density_g_cm3 != 0.0) {
    if (filter.atomic_number < 1 || filter.atomic_number > 92 ||
        !(filter.density_g_cm3 > 0.0))
      throw std::invalid_argument(
          "elemental Spek filter requires atomic_number=1..92 and positive density_g_cm3");
    return {filter.atomic_number, filter.density_g_cm3};
  }
  const std::string_view name = filter.material;
  if (name == "Be") return {4, 1.848};
  if (name == "Ti") return {22, 4.506};
  if (name == "Al") return {13, 2.699};
  if (name == "Cu") return {29, 8.96};
  if (name == "Mo") return {42, 10.28};
  if (name == "Rh") return {45, 12.41};
  if (name == "Ag") return {47, 10.49};
  if (name == "Sn") return {50, 7.31};
  if (name == "W") return {74, 19.3};
  if (name == "Au") return {79, 19.32};
  if (name == "Pb") return {82, 11.34};
  throw std::invalid_argument("unsupported Spek spectrum filter: " + std::string(name));
}

double mu_over_rho(Tables& tables, int z, double energy_keV) {
  const auto& e = tables.get("mu" + std::to_string(z) + "_e");
  const auto& v = tables.get("mu" + std::to_string(z) + "_v");
  std::vector<double> energies = e;
  for (double& value : energies) value *= 1000.0;
  return log_linear(energies, v, energy_keV);
}

double muen_air(Tables& tables, double energy_keV) {
  const auto& e_raw = tables.get("muen_e");
  const auto& v = tables.get("muen_v");
  std::vector<double> e = e_raw;
  for (double& value : e) value *= 1000.0;
  return log_linear(e, v, energy_keV);
}

struct LineSet { double edge; std::vector<double> lines, weights; };

LineSet lines(Tables& tables, const char* shell) {
  return {tables.get(std::string(shell) + "_edge").front(),
          tables.get(std::string(shell) + "_lines"),
          tables.get(std::string(shell) + "_weights")};
}

double br_value(Tables& tables, Model model, double e0, double k) {
  return model == Model::spekcalc ? mebh(e0, k) : nist_brem(tables, e0, k);
}

}  // namespace

const char* model_name(Model model) noexcept {
  return model == Model::spekcalc ? "spekcalc" : "spekpy-v1";
}

SpectrumResult generate(const Config& config) {
  if (!(config.kvp > 1.0) || !(config.bin_width_keV > 0.0) || !(config.mas > 0.0) || !(config.distance_cm > 0.0))
    throw std::invalid_argument("invalid Spek spectrum configuration");
  Tables tables = config.data_file.empty()
      ? Tables(embedded::kData, embedded::kDataSize)
      : Tables(config.data_file);
  const double dk = config.bin_width_keV;
  const int count = static_cast<int>(((config.kvp - dk + config.bin_shift * dk) - 1.0 + 1e-6) / dk) + 1;
  SpectrumResult result;
  result.energy_keV.resize(count);
  result.fluence_per_keV_cm2.assign(count, 0.0);
  result.bremsstrahlung_per_keV_cm2.assign(count, 0.0);
  result.characteristic_per_keV_cm2.assign(count, 0.0);
  for (int i = 0; i < count; ++i)
    result.energy_keV[i] = config.kvp - dk * (count - i - 0.5) + config.bin_shift * dk;

  const bool legacy = config.model == Model::spekcalc;
  const double nbr = legacy ? 0.7046 : 0.9279;
  const double nL1 = legacy ? 0.0 : 0.5832;
  const double nL2 = legacy ? 0.0 : 1.1107;
  const double nL3 = legacy ? 0.0 : 2.8486;
  const double nK = legacy ? 0.4921 : 0.3734;
  const double ntar = kAvogadro * kDensityTungsten / kAtomicWeightTungsten;
  const double constant = ntar * kGeometryFactor * 2.0;
  const double csda_cm = range_csda(tables, config.kvp) / kDensityTungsten;
  const double tw_cm = 0.0119 * std::pow(config.kvp, 1.513) / 1000.0 / kDensityTungsten;
  const double e0_ref = legacy ? 100.0 : config.kvp;
  const double scale = legacy ? ((0.0119 * std::pow(config.kvp, 1.513) / 1000.0) /
                                (0.0119 * std::pow(e0_ref, 1.513) / 1000.0)) /
                               (range_csda(tables, config.kvp) / range_csda(tables, e0_ref)) : 1.0;
  const double max_depth = legacy ? tw_cm : csda_cm * 0.5;
  const int nt = 100;
  const double dt = max_depth / (nt - 1);
  std::vector<double> t(nt);
  for (int j = 0; j < nt; ++j) t[j] = max_depth * j / (nt - 1);
  std::vector<double> t_scaled(nt);
  for (int j = 0; j < nt; ++j) t_scaled[j] = t[j] / (csda_cm * scale);
  const double self_ft = std::sqrt(1.0 + std::tan((3.14159265358979323846 / 2.0) - config.anode_angle_deg * 3.14159265358979323846 / 180.0) *
                         std::tan((3.14159265358979323846 / 2.0) - config.anode_angle_deg * 3.14159265358979323846 / 180.0));

  std::vector<double> self_mu(count);
  for (int i = 0; i < count; ++i) self_mu[i] = mu_over_rho(tables, 74, result.energy_keV[i]) * kDensityTungsten;
  auto one_brem = [&](double k, double depth_index) {
    const int n = std::max(3, static_cast<int>((1.0 - k / config.kvp) * 200.0));
    std::vector<double> values(n);
    const double du = (1.0 - k / config.kvp) / (n - 1);
    for (int q = 0; q < n; ++q) {
      const double u = k / config.kvp + q * du;
      const double ei = u * config.kvp;
      values[q] = br_value(tables, config.model, ei, k) * pe_value(tables, e0_ref, t_scaled[static_cast<int>(depth_index)], u) *
                  ne_value(tables, e0_ref, t_scaled[static_cast<int>(depth_index)]);
    }
    const double norm = [&] {
      const int nn = 200;
      std::vector<double> p(nn);
      for (int q = 0; q < nn; ++q) p[q] = pe_value(tables, e0_ref, t_scaled[static_cast<int>(depth_index)], q / static_cast<double>(nn - 1));
      return simpson(p, 1.0 / (nn - 1));
    }();
    return simpson(values, du) / std::max(norm, 1e-30);
  };
  for (int i = 0; i < count; ++i) {
    std::vector<double> b(nt);
    for (int j = 0; j < nt; ++j) b[j] = one_brem(result.energy_keV[i], j) * std::exp(-self_mu[i] * t[j] * self_ft);
    result.bremsstrahlung_per_keV_cm2[i] = nbr * constant * simpson(b, dt);
  }

  const LineSet l1 = lines(tables, "L_1"), l2 = lines(tables, "L_2"), l3 = lines(tables, "L_3"), kk = lines(tables, "K");
  auto add_characteristic = [&](const LineSet& set, double production) {
    if (config.kvp <= set.edge || production == 0.0) return;
    const int n = std::max(2, static_cast<int>((config.kvp - set.edge) / dk));
    const double dx = (config.kvp - set.edge) / (n - 1);
    std::vector<double> integrated(nt, 0.0);
    for (int j = 0; j < nt; ++j) {
      std::vector<double> b(n);
      for (int q = 0; q < n; ++q) b[q] = one_brem(set.edge + q * dx, j);
      integrated[j] = simpson(b, dx);
    }
    const double emitted = nbr * production * constant * simpson(integrated, dt) / dk;
    const double lower_edge = result.energy_keV.front() - 0.5 * dk;
    for (std::size_t q = 0; q < set.lines.size(); ++q) {
      const int bin = static_cast<int>(std::floor((set.lines[q] - lower_edge) / dk));
      if (bin >= 0 && bin < count) result.characteristic_per_keV_cm2[bin] += emitted * set.weights[q];
    }
  };
  add_characteristic(l1, nL1);
  add_characteristic(l2, nL2);
  add_characteristic(l3, nL3);
  add_characteristic(kk, nK);

  const double geometric = 1.0 / (config.distance_cm * config.distance_cm);
  const double electrons_per_mAs = 1.0e-3 / kElectronCharge;
  for (int i = 0; i < count; ++i) {
    double attenuation = 1.0;
    for (const auto& filter : config.filters) {
      const auto m = material(filter);
      attenuation *= std::exp(-mu_over_rho(tables, m.z, result.energy_keV[i]) * m.density * filter.thickness_mm * 0.1);
    }
    result.bremsstrahlung_per_keV_cm2[i] *= electrons_per_mAs * config.mas * geometric * attenuation;
    result.characteristic_per_keV_cm2[i] *= electrons_per_mAs * config.mas * geometric * attenuation;
    result.fluence_per_keV_cm2[i] = result.bremsstrahlung_per_keV_cm2[i] + result.characteristic_per_keV_cm2[i];
  }
  double fluence_sum = 0.0, energy_sum = 0.0;
  for (int i = 0; i < count; ++i) { fluence_sum += result.fluence_per_keV_cm2[i] * dk; energy_sum += result.energy_keV[i] * result.fluence_per_keV_cm2[i] * dk; }
  result.mean_energy_keV = fluence_sum > 0.0 ? energy_sum / fluence_sum : 0.0;
  for (int i = 0; i < count; ++i) result.air_kerma_uGy += result.energy_keV[i] * result.fluence_per_keV_cm2[i] * muen_air(tables, result.energy_keV[i]) * 1000.0 * kKeVToJ * kGyToUGy * dk;
  double baseline_hvl_weight = 0.0;
  for (int i = 0; i < count; ++i) baseline_hvl_weight += result.energy_keV[i] * result.fluence_per_keV_cm2[i] * muen_air(tables, result.energy_keV[i]) * dk;
  const double target = 0.5;
  double lo = 0.0, hi = 100.0;
  for (int iteration = 0; iteration < 50; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    double k = 0.0;
    for (int i = 0; i < count; ++i) k += result.energy_keV[i] * result.fluence_per_keV_cm2[i] * std::exp(-mu_over_rho(tables, 13, result.energy_keV[i]) * 2.699 * mid * 0.1) * muen_air(tables, result.energy_keV[i]) * dk;
    if (k > target * baseline_hvl_weight) lo = mid; else hi = mid;
  }
  result.hvl1_mm_al = 0.5 * (lo + hi);
  return result;
}

void write_mcgpu_spec(const SpectrumResult& spectrum, const Config& config, const std::filesystem::path& output) {
  if (spectrum.energy_keV.empty() || spectrum.energy_keV.size() != spectrum.fluence_per_keV_cm2.size()) throw std::invalid_argument("empty spectrum");
  const double dk = spectrum.energy_keV.size() > 1 ? spectrum.energy_keV[1] - spectrum.energy_keV[0] : config.bin_width_keV;
  const double scale = (config.distance_cm / 100.0) * (config.distance_cm / 100.0) / config.mas;
  std::ofstream stream(output);
  if (!stream) throw std::runtime_error("cannot create spectrum output: " + output.string());
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.setf(std::ios::fmtflags(0), std::ios::floatfield);
  stream << "# Format: MCGPU-SPEC 1.0\n# Generated by spek_spectrum C++\n"
         << "# Model: " << model_name(config.model) << "\n"
         << "# Tube voltage: " << config.kvp << " kVp; target angle: " << config.anode_angle_deg << " deg\n"
         << "# Bin width: " << config.bin_width_keV << " keV; bin shift: " << config.bin_shift << "\n";
  if (config.filters.empty()) {
    stream << "# Filters: none\n";
  } else {
    for (const auto& filter : config.filters) {
      stream << "# Filter: ";
      if (!filter.material.empty()) {
        stream << filter.material;
      } else {
        stream << "Z=" << filter.atomic_number << "; density="
               << filter.density_g_cm3 << " g/cm3";
      }
      stream << "; thickness=" << filter.thickness_mm << " mm\n";
    }
  }
  stream << "# Reference: 1 mAs at 100 cm\n"
         << "# Mean energy: " << spectrum.mean_energy_keV << " keV\n"
         << "# Air kerma: " << spectrum.air_kerma_uGy * scale << " uGy/mAs at 100 cm\n"
         << "# Energy[keV]  N[keV cm^2 mAs]^-1 @ 1 meter\n";
  stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (std::size_t i = 0; i < spectrum.energy_keV.size(); ++i) stream << spectrum.energy_keV[i] - 0.5 * dk << ' ' << spectrum.fluence_per_keV_cm2[i] * scale << '\n';
  stream << spectrum.energy_keV.back() + 0.5 * dk << " 0\n";
  stream.close();
}

}  // namespace spek_spectrum
