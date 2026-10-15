#include "recon/preprocess/calibration/EmpiricalTwoMaterialFit.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::preprocess {
namespace {
    using A = std::array<double, 5>;
    A decode(const A& q) { return { 1 / (1 + std::exp(-q[0])), std::exp(q[1]), std::exp(q[2]), std::exp(q[3]), std::exp(q[4]) }; }
    double value(const A& p, double x, double y)
    {
        const double u = std::log(p[0]) - p[1] * x - p[2] * y;
        const double v = std::log1p(-p[0]) - p[3] * x - p[4] * y;
        const double m = std::max(u, v);
        return -(m + std::log(std::exp(u - m) + std::exp(v - m)));
    }
    double cost(const A& q, const std::vector<ThicknessMeasurement>& rows)
    {
        double sum = 0;
        const auto p = decode(q);
        for (const auto& r : rows) {
            const double e = value(p, r.first_mm, r.second_mm) - r.log_projection;
            sum += e * e;
        }
        return sum / rows.size();
    }
    bool solve(double h[5][6], A& out)
    {
        for (int i = 0; i < 5; ++i) {
            int pivot = i;
            for (int j = i + 1; j < 5; ++j)
                if (std::abs(h[j][i]) > std::abs(h[pivot][i]))
                    pivot = j;
            if (std::abs(h[pivot][i]) < 1e-20)
                return false;
            for (int k = i; k < 6; ++k)
                std::swap(h[i][k], h[pivot][k]);
            const double d = h[i][i];
            for (int k = i; k < 6; ++k)
                h[i][k] /= d;
            for (int j = 0; j < 5; ++j)
                if (j != i) {
                    const double a = h[j][i];
                    for (int k = i; k < 6; ++k)
                        h[j][k] -= a * h[i][k];
                }
        }
        for (int i = 0; i < 5; ++i)
            out[i] = h[i][5];
        return true;
    }
}
EmpiricalTwoMaterialFit fitEmpiricalTwoMaterial(const std::vector<ThicknessMeasurement>& rows)
{
    if (rows.size() < 20)
        throw std::invalid_argument("Too few calibration measurements");
    double xx = 0, yy = 0, xy = 0;
    for (const auto& r : rows) {
        if (!std::isfinite(r.first_mm) || !std::isfinite(r.second_mm) || !std::isfinite(r.log_projection) || r.first_mm < 0 || r.second_mm < 0 || r.log_projection < 0)
            throw std::invalid_argument("Invalid calibration measurement");
        xx += r.first_mm * r.first_mm;
        yy += r.second_mm * r.second_mm;
        xy += r.first_mm * r.second_mm;
    }
    if (!(xx > 0 && yy > 0 && xx * yy - xy * xy > 1e-5 * xx * yy))
        throw std::invalid_argument("Calibration does not span two independent thicknesses");
    double best = std::numeric_limits<double>::infinity();
    A bestq {};
    for (double mixture : { .2, .5, .8 }) {
        A q = { std::log(mixture / (1 - mixture)), std::log(.04), std::log(.18), std::log(.018), std::log(.05) };
        double current = cost(q, rows), lambda = 1e-3;
        for (int iteration = 0; iteration < 160; ++iteration) {
            double h[5][6] = {};
            const auto p = decode(q);
            std::array<A, 5> plus, minus;
            for (int j = 0; j < 5; ++j) {
                auto a = q, b = q;
                a[j] += 1e-4;
                b[j] -= 1e-4;
                plus[j] = decode(a);
                minus[j] = decode(b);
            }
            for (const auto& r : rows) {
                const double residual = value(p, r.first_mm, r.second_mm) - r.log_projection;
                A jac {};
                for (int j = 0; j < 5; ++j)
                    jac[j] = (value(plus[j], r.first_mm, r.second_mm) - value(minus[j], r.first_mm, r.second_mm)) / 2e-4;
                for (int j = 0; j < 5; ++j) {
                    h[j][5] -= jac[j] * residual;
                    for (int k = 0; k < 5; ++k)
                        h[j][k] += jac[j] * jac[k];
                }
            }
            for (int j = 0; j < 5; ++j)
                h[j][j] += lambda * (h[j][j] + 1e-12);
            A step {}, candidate = q;
            if (!solve(h, step))
                break;
            for (int j = 0; j < 5; ++j)
                candidate[j] = std::clamp(q[j] + step[j], -14., j == 0 ? 14. : 1.);
            const double next = cost(candidate, rows);
            if (next < current) {
                const double improvement = current - next;
                q = candidate;
                current = next;
                lambda = std::max(lambda * .3, 1e-12);
                if (improvement < 1e-16)
                    break;
            } else {
                lambda *= 10;
                if (lambda > 1e12)
                    break;
            }
        }
        if (current < best) {
            best = current;
            bestq = q;
        }
    }
    if (!std::isfinite(best))
        throw std::runtime_error("Empirical fit failed");
    return { decode(bestq), std::sqrt(best) };
}
double EmpiricalTwoMaterialFit::evaluate(double x, double y) const { return value(coefficients, x, y); }
TwoMaterialCalibration EmpiricalTwoMaterialFit::table(double xmax, double ymax, const std::string& provenance) const
{
    if (!(std::isfinite(xmax) && std::isfinite(ymax) && xmax > 0 && ymax > 0 && xmax <= 1000 && ymax <= 1000))
        throw std::invalid_argument("Invalid table extent");
    TwoMaterialSamples s;
    s.water_columns = int(std::ceil(xmax / .5)) + 1;
    s.water_step_mm = xmax / (s.water_columns - 1);
    s.bone_rows = int(std::ceil(ymax / .1)) + 1;
    s.bone_step_mm = ymax / (s.bone_rows - 1);
    const auto& p = coefficients;
    s.target_water_mu_mm = p[0] * p[1] + (1 - p[0]) * p[3];
    s.target_bone_mu_mm = p[0] * p[2] + (1 - p[0]) * p[4];
    s.reference_energy_keV = 0; // No monoenergetic interpretation without separate calibration.
    s.provenance = provenance;
    for (int b = 0; b < s.bone_rows; ++b)
        for (int w = 0; w < s.water_columns; ++w)
            s.log_values.push_back(evaluate(w * s.water_step_mm, b * s.bone_step_mm));
    return calibrateTwoMaterial(s);
}
}
