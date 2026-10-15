#include "recon/preprocess/CudaPixelOps.hpp"
#include <algorithm>
#include <cmath>
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
    check(caught, "invalid CUDA pixel request accepted");
}
int main(int argc, char** argv)
{
    try {
        check(argc == 2, "case required");
        const std::string name = argv[1];
        const size_t n = 513;
        CudaPixelOps ops(n);
        std::vector<float> p(n), air(n), out;
        for (size_t i = 0; i < n; ++i) {
            p[i] = float(.1 + (i % 31) * .03);
            air[i] = float(100 + (i % 17) * 7);
        }
        if (name == "polynomial") {
            for (int degree = 0; degree <= 16; ++degree) {
                BeamHardeningConfig c;
                c.clamp_min = .005;
                c.clamp_max = .025;
                c.coefficients.clear();
                for (int j = 0; j <= degree; ++j)
                    c.coefficients.push_back((j % 2 ? -.02 : .03) / (j + 1));

                ops.configureBhc(c);
                ops.upload(p, nullptr);
                ops.finalize(false, true);
                ops.download(out);
                for (size_t i = 0; i < n; ++i) {
                    double expected = 0;
                    for (int j = 0; j <= degree; ++j)
                        expected += c.coefficients[j] * std::pow(double(p[i]), j);
                    expected = std::clamp(expected, c.clamp_min, c.clamp_max);
                    check(std::abs(out[i] - expected) < 2e-7, "CUDA polynomial/clamp differs from power-sum oracle");
                }
            }
        } else if (name == "roundtrip") {
            ops.upload(p, &air);
            ops.restoreEnergy();
            ops.download(out);
            for (size_t i = 0; i < n; ++i)
                check(std::abs(out[i] - air[i] * std::exp(-p[i])) < 5e-5, "log-to-energy mismatch");
            ops.finalize(true, false);
            ops.download(out);
            for (size_t i = 0; i < n; ++i)
                check(std::abs(out[i] - p[i]) < 2e-6, "energy/log round trip");
        } else if (name == "residual") {
            ops.upload(p, &air);
            ops.restoreEnergy();
            std::vector<float> prediction(n);
            for (size_t i = 0; i < n; ++i)
                prediction[i] = i % 3 == 0 ? -10 : i % 3 == 1 ? 20
                                                              : 10000;
            ops.uploadTo(ops.prediction(), prediction);
            ops.guidedResidual(.5);
            ops.download(out);
            for (size_t i = 0; i < n; ++i) {
                const float expected = std::max(air[i] * 1e-8f, air[i] * std::exp(-p[i]) - .5f * std::max(0.f, prediction[i]));
                check(std::abs(out[i] - expected) < 5e-5, "guided residual floor or negative prediction");
            }
        } else if (name == "mask") {
            ops.upload(p, &air);
            ops.restoreEnergy();
            auto mask = ops.mask(ops.data(), ops.air(), .515);
            check(mask.size() == n, "mask size");
            for (size_t i = 0; i < n; ++i)
                check(bool(mask[i]) == (p[i] > .515), "mask threshold or tail indexing");
        } else if (name == "invalid") {
            reject([] { CudaPixelOps empty(0); });
            reject([&] { ops.upload(std::vector<float>(n - 1), nullptr); });
            for (int k = 0; k < 4; ++k) {
                BeamHardeningConfig c;
                c.coefficients = { 0, 1 };
                if (k == 0)
                    c.coefficients.clear();
                if (k == 1)
                    c.coefficients.resize(18);
                if (k == 2)
                    c.coefficients[0] = NAN;
                if (k == 3) {
                    c.clamp_min = 2;
                    c.clamp_max = 1;
                }
                reject([&] { ops.configureBhc(c); });
            }
            auto bad = p;
            bad[n - 1] = NAN;
            ops.upload(bad, nullptr);
            ops.finalize(false, false);
            reject([&] { ops.download(out); });
            ops.upload(p, nullptr);
            ops.finalize(false, false);
            ops.download(out);
            check(out == p, "invalid flag leaked into next frame");
            air[n - 1] = 0;
            ops.upload(p, &air);
            ops.restoreEnergy();
            reject([&] { ops.download(out); });
        } else
            throw std::runtime_error("unknown case");
        std::cout << name << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
