#pragma once
#include "TwoMaterialCalibration.hpp"
#include <array>
namespace yk::spectral::preprocess {
// Independent of spectrum, detector model and material attenuation database.
struct ThicknessMeasurement {
    double first_mm, second_mm, log_projection;
};
struct EmpiricalTwoMaterialFit {
    // a,b,c,d,f: -log(a*exp(-b*x-c*y)+(1-a)*exp(-d*x-f*y)).
    std::array<double, 5> coefficients {};
    double rmse = 0;
    double evaluate(double first_mm, double second_mm) const;
    TwoMaterialCalibration table(double first_max_mm, double second_max_mm,
        const std::string& provenance) const;
};
EmpiricalTwoMaterialFit fitEmpiricalTwoMaterial(const std::vector<ThicknessMeasurement>&);
}
