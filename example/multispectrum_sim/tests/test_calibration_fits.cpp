#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include "recon/preprocess/BeamHardeningFit.hpp"
#include "recon/preprocess/calibration/EmpiricalBhcFit.hpp"
#include "recon/preprocess/calibration/EmpiricalTwoMaterialFit.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
using namespace yk::spectral;
using namespace yk::spectral::preprocess;
void check(bool b, const char* m)
{
    if (!b)
        throw std::runtime_error(m);
}
template <class F>
void reject(F f)
{
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, "invalid fit input accepted");
}
double evaluate(const std::vector<double>& c, double x)
{
    double y = 0;
    for (auto i = c.rbegin(); i != c.rend(); ++i)
        y = y * x + *i;
    return y;
}
#include "NumericalFitSamples.hpp"
int main(int argc, char** argv)
{
    try {
        check(argc == 3, "case and scratch path required");
        const std::string test = argv[1];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (test == "water_matrix" || test == "material_matrix" || test == "empirical_matrix") {
            fitSampleMatrix(test, std::filesystem::path(argv[2]) / test);
        } else if (test.rfind("water_", 0) == 0) {
            const std::filesystem::path dir = std::filesystem::path(argv[2]) / test;
            std::filesystem::create_directories(dir);
            const auto csv = dir / "spectrum.csv";
            {
                std::ofstream f(csv);
                f << "Energy[keV]  N[keV cm^2 mAs]^-1 @ 1 meter\n";
                if (test == "water_eid")
                    f << "40 1\n80 1\n";
                else
                    f << "70 1\n";
            }
            SimulationConfig config;
            config.projection.spectrum_file = csv;
            BeamHardeningConfig bh;
            bh.order = 3;
            bh.water_plate_samples = 81;
            bh.water_plate_max_mm = 400;
            bh.reference_energy_keV = 70;
            if (test == "water_invalid") {
                for (int k = 0; k < 8; ++k) {
                    auto bad = bh;
                    if (k == 0)
                        bad.order = -1;
                    if (k == 1)
                        bad.order = 17;
                    if (k == 2)
                        bad.water_plate_samples = 1;
                    if (k == 3)
                        bad.water_plate_samples = bad.order + 1;
                    if (k == 4)
                        bad.water_plate_max_mm = 0;
                    if (k == 5)
                        bad.water_plate_max_mm = nan;
                    if (k == 6)
                        bad.reference_energy_keV = -1;
                    if (k == 7)
                        bad.reference_energy_keV = nan;
                    reject([&] { fitWaterPlate(config, bad, {}); });
                }
                reject([&] { fitWaterPlate(config, bh, [](TaskProgress) { throw std::runtime_error("cancel fit"); }); });
                check(fitWaterPlate(config, bh, {}).coefficients.size() == 4, "fit not reusable after cancellation");
            } else if (test == "water_mono") {
                for (int order = 1; order <= 16; ++order) {
                    bh.order = order;
                    int notifications = 0;
                    auto fitted = fitWaterPlate(config, bh, [&](TaskProgress p) { check(p.completed == notifications++ && p.total == 81, "fit progress order"); });
                    check(notifications == 82, "fit final progress missing");
                    for (double p : { 0., .01, .37, 1.9, 4.2, 7.7 })
                        check(std::abs(evaluate(fitted.coefficients, p) - p) < 2e-7, "mono spectrum fit must be identity");
                }
            } else if (test == "water_eid") {
                bh.reference_energy_keV = 0;
                MaterialSpec water;
                water.formula = "H2O";
                water.density_g_cm3 = 1.;
                XcomAttenuationProvider provider({});
                std::vector<double> mu;
                std::string error;
                check(provider.query(water, { 40, 80, 60 }, mu, error), "XCOM oracle query failed");
                check(std::abs(resolveWaterReference(config, bh).energy_keV - 60) < 1e-12, "reference energy must be photon mean");
                for (int order = 3; order <= 16; ++order) {
                    bh.order = order;
                    auto fitted = fitWaterPlate(config, bh, {});
                    for (double mm : { .7, 13.2, 89.1, 181.7, 299.3, 398.1 }) {
                        double p = -std::log((40 * std::exp(-mu[0] * mm / 10) + 80 * std::exp(-mu[1] * mm / 10)) / 120);
                        check(std::abs(evaluate(fitted.coefficients, p) - mu[2] * mm / 10) < .01, "EID held-out polynomial error");
                    }
                }
            } else
                throw std::runtime_error("unknown water test");
        } else if (test.rfind("empirical_", 0) == 0) {
            std::vector<EmpiricalBhcDifference> samples;
            for (int i = 0; i < 300; ++i) {
                double a = std::sin(i * .17), b = std::cos(i * .31);
                samples.push_back({ -.7 * a + .2 * b, a, b });
            }
            if (test == "empirical_recovery") {
                auto fit = fitEmpiricalBhc(samples);
                check(std::abs(fit.square - .7) < 1e-4 && std::abs(fit.cross + .2) < 1e-4, "coefficient recovery");
                check(fit.after < 1e-4 && fit.before > .1, "fit error reporting");
                std::reverse(samples.begin(), samples.end());
                auto reverse = fitEmpiricalBhc(samples);
                check(std::abs(fit.square - reverse.square) < 1e-10, "sample-order dependence");
                for (int i = 0; i < 8; ++i)
                    samples[i].baseline += 1000;
                fit = fitEmpiricalBhc(samples, .1);
                check(std::abs(fit.square - .7) < .02 && std::abs(fit.cross + .2) < .02, "Huber outlier resistance");
            } else if (test == "empirical_invalid") {
                reject([&] { fitEmpiricalBhc({}, 1); });
                for (double h : { 0., -1., nan })
                    reject([&] { fitEmpiricalBhc(samples, h); });
                for (int k = 0; k < 3; ++k) {
                    auto s = samples;
                    if (k == 0)
                        s[0].baseline = nan;
                    if (k == 1)
                        s[0].square = nan;
                    if (k == 2)
                        s[0].cross = nan;
                    reject([&] { fitEmpiricalBhc(s); });
                }
                for (auto& s : samples)
                    s.cross = 2 * s.square;
                reject([&] { fitEmpiricalBhc(samples); });
            } else
                throw std::runtime_error("unknown empirical test");
        } else if (test.rfind("material_", 0) == 0) {
            auto truth = [](double x, double y) { return -std::log(.35 * std::exp(-.035 * x - .09 * y) + .65 * std::exp(-.018 * x - .04 * y)); };
            std::vector<ThicknessMeasurement> samples;
            for (int x = 0; x <= 160; x += 10)
                for (int y = 0; y <= 40; y += 4)
                    samples.push_back({ double(x), double(y), truth(x, y) });
            if (test == "material_holdout") {
                auto fit = fitEmpiricalTwoMaterial(samples);
                check(fit.rmse < 1e-5, "two-material fit failed");
                for (double x : { .3, 17.2, 93.7, 159.1 })
                    for (double y : { .1, 3.7, 13.3, 39.9 })
                        check(std::abs(fit.evaluate(x, y) - truth(x, y)) < 3e-5, "two-material independent holdout");
                auto lut = fit.table(160, 40, "test");
                check(lut.reference_energy_keV == 0, "tangent incorrectly assigned mono energy");
                check(std::abs(lut.target_water_mu_mm - (.35 * .035 + .65 * .018)) < 1e-5, "first material slope");
                check(std::abs(lut.target_bone_mu_mm - (.35 * .09 + .65 * .04)) < 1e-5, "second material slope");
                for (double x : { 0., -1., 1001., nan })
                    reject([&] { fit.table(x, 40, "bad"); });
            } else if (test == "material_invalid") {
                reject([&] { fitEmpiricalTwoMaterial({}); });
                for (int k = 0; k < 6; ++k) {
                    auto s = samples;
                    if (k == 0)
                        s[0].first_mm = -1;
                    if (k == 1)
                        s[0].second_mm = -1;
                    if (k == 2)
                        s[0].log_projection = -1;
                    if (k == 3)
                        s[0].first_mm = nan;
                    if (k == 4)
                        s[0].second_mm = nan;
                    if (k == 5)
                        s[0].log_projection = nan;
                    reject([&] { fitEmpiricalTwoMaterial(s); });
                }
                for (auto& s : samples)
                    s.second_mm = 2 * s.first_mm;
                reject([&] { fitEmpiricalTwoMaterial(samples); });
            } else
                throw std::runtime_error("unknown material test");
        } else
            throw std::runtime_error("unknown test case");
        std::cout << test << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
