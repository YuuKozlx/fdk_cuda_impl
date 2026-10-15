// Fixed-seed numerical families. Accuracy thresholds refer to held-out responses,
// not uniquely identifiable nonlinear coefficients.
#include <random>
void fitSampleMatrix(const std::string& kind, const std::filesystem::path& root)
{
    std::mt19937 rng(20261015);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);
    int cases = 0;
    if (kind == "material_matrix") {
        for (double mixture : { .2, .5, .8 })
            for (double scale : { .7, 1.3 })
                for (double noise : { 0., .001 }) {
                    auto response = [&](double x, double y) { return -std::log(mixture * std::exp(-.035 * scale * x - .09 * scale * y) + (1 - mixture) * std::exp(-.018 * scale * x - .04 * scale * y)); };
                    std::vector<ThicknessMeasurement> rows;
                    for (int x = 0; x <= 180; x += 12)
                        for (int y = 0; y <= 48; y += 4) {
                            if (x && y && uniform(rng) < .2)
                                continue;
                            rows.push_back({ double(x), double(y), std::max(0., response(x, y) + noise * normal(rng)) });
                        }
                    auto fit = fitEmpiricalTwoMaterial(rows);
                    double squared = 0, peak = 0;
                    for (int k = 0; k < 150; ++k) {
                        double x = uniform(rng) * 180, y = uniform(rng) * 48, d = fit.evaluate(x, y) - response(x, y);
                        squared += d * d;
                        peak = std::max(peak, std::abs(d));
                    }
                    const double rmse = std::sqrt(squared / 150);
                    if (rmse > (noise ? .0015 : 2e-5) || peak > (noise ? .005 : 1e-4))
                        throw std::runtime_error("material matrix holdout failed case " + std::to_string(cases) + " rmse=" + std::to_string(rmse));
                    ++cases;
                }
    } else if (kind == "empirical_matrix") {
        for (double a : { -.6, 0., 1.4 })
            for (double scale : { .1, 1., 10. })
                for (double rho : { 0., .8 })
                    for (double fraction : { 0., .05, .15 }) {
                        const double b = -.25, noise = .002 * scale;
                        std::vector<EmpiricalBhcDifference> rows;
                        for (int k = 0; k < 800; ++k) {
                            double x = scale * normal(rng), y = rho * x + scale * std::sqrt(1 - rho * rho) * normal(rng), base = -a * x - b * y + noise * normal(rng);
                            if (uniform(rng) < fraction)
                                base += scale * 30 * normal(rng);
                            rows.push_back({ base, x, y });
                        }
                        auto fit = fitEmpiricalBhc(rows, .02 * scale);
                        if (std::abs(fit.square - a) > .025 || std::abs(fit.cross - b) > .025)
                            throw std::runtime_error("robust matrix failed case " + std::to_string(cases));
                        ++cases;
                    }
    } else if (kind == "water_matrix") {
        std::filesystem::create_directories(root);
        const auto file = root / "matrix-spectrum.csv";
        SimulationConfig config;
        config.projection.spectrum_file = file;
        for (double energy : { 35., 70., 120. })
            for (double thickness : { 10., 100., 500. })
                for (int order : { 1, 3, 6, 10, 16 })
                    for (int samples : { 33, 101 }) {
                        {
                            std::ofstream out(file);
                            out << "Energy[keV]  N[keV cm^2 mAs]^-1 @ 1 meter\n"
                                << energy << " 1\n";
                        }
                        BeamHardeningConfig bh;
                        bh.order = order;
                        bh.water_plate_max_mm = thickness;
                        bh.water_plate_samples = samples;
                        bh.reference_energy_keV = energy;
                        auto fit = fitWaterPlate(config, bh, {});
                        const double maxp = resolveWaterReference(config, bh).mu_water_mm_inv * thickness;
                        for (int k = 0; k < 101; ++k) {
                            double p = maxp * (k + .37) / 101;
                            double d = std::abs(evaluate(fit.coefficients, p) - p);
                            if (d > 1e-6 * std::max(1., maxp))
                                throw std::runtime_error("water matrix identity failed case " + std::to_string(cases));
                        }
                        ++cases;
                    }
        // Numerically degenerate thickness range must fail, never return NaN coefficients.
        BeamHardeningConfig bh;
        bh.order = 6;
        bh.water_plate_max_mm = 1e-12;
        bh.water_plate_samples = 30;
        bh.reference_energy_keV = 70;
        reject([&] { fitWaterPlate(config, bh, {}); });
    } else
        throw std::runtime_error("unknown sample matrix");
    std::cout << kind << " numerical cases=" << cases << '\n';
}
