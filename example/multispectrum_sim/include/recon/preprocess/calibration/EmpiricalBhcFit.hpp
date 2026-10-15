#pragma once
#include <vector>
namespace yk::spectral::preprocess {
// Differences across valid tissue samples, all in the same image units.
struct EmpiricalBhcDifference {
    double baseline, square, cross;
};
struct EmpiricalBhcFit {
    double square = 0, cross = 0, correlation = 0, before = 0, after = 0;
};
// Robust flatness fit, not a reproduction of the original paper's exact objective.
EmpiricalBhcFit fitEmpiricalBhc(const std::vector<EmpiricalBhcDifference>&, double huber = 10);
}
