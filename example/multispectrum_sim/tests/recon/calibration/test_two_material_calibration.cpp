#include "recon/preprocess/calibration/TwoMaterialCalibration.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
using namespace yk::spectral::preprocess;
void check(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}
template <class F>
void rejects(F action)
{
    bool caught = false;
    try {
        action();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, "invalid calibration accepted");
}
int main(int argc, char** argv)
{
    try {
        check(argc == 2, "output path required");
        TwoMaterialSamples s;
        s.bone_rows = 3;
        s.water_columns = 3;
        s.bone_step_mm = 2;
        s.water_step_mm = 10;
        s.reference_energy_keV = 70;
        s.target_water_mu_mm = .02;
        s.target_bone_mu_mm = .04;
        s.provenance = "linear oracle";
        // Independent analytic oracle: p=.03*w+.07*b; target=.02*w+.04*b.
        for (int b = 0; b < 3; ++b)
            for (int w = 0; w < 3; ++w)
                s.log_values.push_back(.03 * w * 10 + .07 * b * 2);
        auto m = calibrateTwoMaterial(s, 17);
        for (double b : { 0., .5, 2., 3.9, 4. })
            for (double w : { -2., 0., 3.7, 10., 20., 25. })
                check(std::abs(m.correct(float(.03 * w + .07 * b), float(b)) - (.02 * w + .04 * b)) < 2e-6, "linear interpolation/extrapolation oracle");
        const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
        for (double b : { -1., 4.01, nan, inf })
            rejects([&] { m.correct(1, float(b)); });
        for (double p : { nan, inf, -inf })
            rejects([&] { m.correct(float(p), 1); });
        for (int which = 0; which < 9; ++which) {
            auto bad = m;
            switch (which) {
            case 0:
                bad.bone_rows = 1;
                break;
            case 1:
                bad.projection_columns = 1;
                break;
            case 2:
                bad.bone_step_mm = 0;
                break;
            case 3:
                bad.reference_energy_keV = -1;
                break;
            case 4:
                bad.minimum_log.pop_back();
                break;
            case 5:
                bad.log_step[0] = 0;
                break;
            case 6:
                bad.target_log[1] = bad.target_log[0];
                break;
            case 7:
                bad.target_log[0] = float(nan);
                break;
            case 8:
                bad.target_water_mu_mm = inf;
                break;
            }
            rejects([&] { bad.validate(); });
        }
        for (int which = 0; which < 5; ++which) {
            auto bad = s;
            if (which == 0)
                bad.log_values.pop_back();
            if (which == 1)
                bad.log_values[1] = bad.log_values[0];
            if (which == 2)
                bad.log_values[2] = nan;
            if (which == 3)
                bad.water_step_mm = inf;
            if (which == 4)
                bad.target_bone_mu_mm = 0;
            rejects([&] { calibrateTwoMaterial(bad); });
        }
        rejects([&] { calibrateTwoMaterial(s, 1); });
        const std::filesystem::path file = argv[1];
        std::filesystem::create_directories(file.parent_path());
        m.save(file);
        auto loaded = TwoMaterialCalibration::load(file);
        check(loaded.target_log == m.target_log && loaded.minimum_log == m.minimum_log && loaded.log_step == m.log_step && loaded.provenance == m.provenance, "serialization changed table");
        nlohmann::json saved;
        {
            std::ifstream in(file);
            in >> saved;
        }
        for (int which = 0; which < 4; ++which) {
            auto j = saved;
            if (which == 0)
                j["schema"] = "unknown";
            if (which == 1)
                j["input_domain"] = "water-corrected-log";
            if (which == 2)
                j.erase("target_log");
            if (which == 3)
                j["bone_rows"] = 999;
            {
                std::ofstream out(file);
                out << j;
            }
            rejects([&] { TwoMaterialCalibration::load(file); });
        }
        {
            std::ofstream out(file);
            out << "{broken";
        }
        rejects([&] { TwoMaterialCalibration::load(file); });
        m.reference_energy_keV = 0;
        m.save(file);
        check(TwoMaterialCalibration::load(file).reference_energy_keV == 0, "empirical tangent energy changed");
        std::cout << "Calibration CPU analytic oracle, invalid inputs and serialization tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
