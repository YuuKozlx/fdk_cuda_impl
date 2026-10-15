#include "recon/preprocess/calibration/EmpiricalBhcFit.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace yk::spectral::preprocess {
EmpiricalBhcFit fitEmpiricalBhc(const std::vector<EmpiricalBhcDifference>& samples, double huber)
{
    if (samples.size() < 20 || !std::isfinite(huber) || huber <= 0)
        throw std::invalid_argument("Insufficient EBHC samples / invalid Huber scale");
    double aa = 0, ab = 0, bb = 0;
    for (const auto& s : samples) {
        if (!std::isfinite(s.baseline) || !std::isfinite(s.square) || !std::isfinite(s.cross))
            throw std::invalid_argument("Nonfinite EBHC difference");
        aa += s.square * s.square;
        ab += s.square * s.cross;
        bb += s.cross * s.cross;
    }
    if (aa <= 1e-16 || bb <= 1e-16 || aa * bb - ab * ab < 1e-8 * aa * bb)
        throw std::invalid_argument("Degenerate EBHC basis");
    EmpiricalBhcFit out;
    out.correlation = ab / std::sqrt(aa * bb);
    for (int iteration = 0; iteration < 30; ++iteration) {
        double a = aa * 1e-5, b = 0, c = bb * 1e-5, d = 0, e = 0;
        for (const auto& s : samples) {
            const double residual = s.baseline + out.square * s.square + out.cross * s.cross;
            const double w = std::min(1., huber / std::max(std::abs(residual), 1e-20));
            a += w * s.square * s.square;
            b += w * s.square * s.cross;
            c += w * s.cross * s.cross;
            d -= w * s.square * s.baseline;
            e -= w * s.cross * s.baseline;
        }
        const double det = a * c - b * b, x = (d * c - b * e) / det, y = (a * e - b * d) / det;
        if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 100 || std::abs(y) > 100)
            throw std::runtime_error("Unstable EBHC fit");
        const double change = std::abs(x - out.square) + std::abs(y - out.cross);
        out.square = x;
        out.cross = y;
        if (change < 1e-8)
            break;
    }
    for (const auto& s : samples) {
        out.before += s.baseline * s.baseline;
        const double r = s.baseline + out.square * s.square + out.cross * s.cross;
        out.after += r * r;
    }
    out.before = std::sqrt(out.before / samples.size());
    out.after = std::sqrt(out.after / samples.size());
    return out;
}
}
