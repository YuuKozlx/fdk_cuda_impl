#include "recon/preprocess/calibration/EmpiricalTwoMaterialFit.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
using namespace yk::spectral::preprocess;
int main(int argc, char** argv)
{
    try {
        if (argc != 3)
            throw std::runtime_error("Usage: multispectrum_fit_material_phantom samples.csv output-directory");
        std::ifstream in(argv[1]);
        if (!in)
            throw std::runtime_error("Cannot read samples");
        std::string line;
        std::getline(in, line);
        std::vector<ThicknessMeasurement> train, holdout;
        std::map<std::pair<int, int>, int> bins;
        double xmax = 0, ymax = 0;
        while (std::getline(in, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            int view;
            ThicknessMeasurement r {};
            if (!(row >> view >> r.first_mm >> r.second_mm >> r.log_projection))
                throw std::runtime_error("Malformed sample row");
            if (!std::isfinite(r.first_mm) || !std::isfinite(r.second_mm) || !std::isfinite(r.log_projection) || r.first_mm < 0 || r.second_mm < 0 || r.log_projection < 0)
                throw std::runtime_error("Invalid sample row");
            if (r.first_mm + r.second_mm < .5)
                continue;
            // Entire views held out, before balancing; never random per-pixel leakage.
            if (view % 5 == 0) {
                holdout.push_back(r);
                continue;
            }
            auto& n = bins[{ int(r.first_mm / 5), int(r.second_mm / 2) }];
            if (n++ < 12)
                train.push_back(r);
            xmax = std::max(xmax, r.first_mm);
            ymax = std::max(ymax, r.second_mm);
        }
        if (holdout.empty())
            throw std::runtime_error("No held-out views");
        const auto fit = fitEmpiricalTwoMaterial(train);
        double squared = 0, peak = 0;
        for (const auto& r : holdout) {
            const double e = fit.evaluate(r.first_mm, r.second_mm) - r.log_projection;
            squared += e * e;
            peak = std::max(peak, std::abs(e));
        }
        const std::filesystem::path out = argv[2];
        std::filesystem::create_directories(out);
        if (std::filesystem::exists(out / "calibration.json"))
            throw std::runtime_error("Output calibration already exists");
        nlohmann::json metadata;
        std::ifstream sidecar(std::string(argv[1]) + ".json");
        if (sidecar)
            sidecar >> metadata;
        auto model = fit.table(xmax, ymax, "PMMA/Al phantom empirical fit; pure Al, not AL6082; provided phantom label paths; zero-thickness tangent output, not 70 keV; samples=" + std::filesystem::absolute(argv[1]).generic_string());
        model.save(out / "calibration.json");
        nlohmann::json report = { { "coefficients_a_b_c_d_f", fit.coefficients }, { "training_samples", train.size() }, { "heldout_samples", holdout.size() },
            { "occupied_thickness_bins", bins.size() }, { "training_rmse", fit.rmse }, { "heldout_rmse", std::sqrt(squared / holdout.size()) }, { "heldout_max_error", peak },
            { "first_material", "PMMA" }, { "second_material", "pure aluminum" }, { "first_max_mm", xmax }, { "second_max_mm", ymax },
            { "target", "zero-thickness tangent; no assigned energy" }, { "first_target_mu_mm", model.target_water_mu_mm }, { "second_target_mu_mm", model.target_bone_mu_mm },
            { "spectrum_input_to_fitter", false }, { "thickness_source", "provided phantom label volume" }, { "sample_metadata", metadata } };
        std::ofstream(out / "fit.json") << report.dump(2);
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
