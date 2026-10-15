#include "recon/preprocess/calibration/EmpiricalBhcFit.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;
namespace fs = std::filesystem;
using namespace yk::spectral::preprocess;
template <class T>
std::vector<T> read(const fs::path& path, size_t n)
{
    if (fs::file_size(path) != n * sizeof(T))
        throw std::runtime_error("Wrong volume size: " + path.string());
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ios::badbit | std::ios::failbit);
    std::vector<T> v(n);
    in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    return v;
}
int main(int argc, char** argv)
{
    try {
        const auto start = std::chrono::steady_clock::now();
        if (argc != 2)
            throw std::runtime_error("Usage: fit-ebhc experiment.json");
        std::ifstream in(argv[1]);
        Json job;
        in >> job;
        auto dims = job.at("dimensions_xyz").get<std::array<int, 3>>();
        const int nx = dims[0], ny = dims[1], nz = dims[2], s = 4;
        if (nx < 16 || ny < 16 || nz < 16 || nx % s || ny % s || nz % s)
            throw std::runtime_error("Experiment requires dimensions divisible by four");
        const double mu = job.at("water_mu_mm_inv");
        if (!std::isfinite(mu) || mu <= 0)
            throw std::runtime_error("Invalid water attenuation");
        const size_t n = size_t(nx) * ny * nz;
        const int dx = nx / s, dy = ny / s, dz = nz / s;
        const size_t dn = size_t(dx) * dy * dz;
        const fs::path output = job.at("output_directory").get<std::string>();
        fs::create_directories(output);
        if (fs::exists(output / "reconstruction.raw") || fs::exists(output / "coefficients.json"))
            throw std::runtime_error("Fit outputs exist");
        auto labels = read<unsigned char>(job.at("segmentation").get<std::string>(), n);
        std::vector<unsigned char> mask(dn, 1);
        std::array<std::vector<float>, 3> full, low;
        const std::array<std::string, 3> keys = { "baseline", "square", "cross" };
        for (int k = 0; k < 3; ++k) {
            full[k] = read<float>(job.at(keys[k]).get<std::string>(), n);
            low[k].resize(dn);
            for (int z = 0; z < dz; ++z)
                for (int y = 0; y < dy; ++y)
                    for (int x = 0; x < dx; ++x) {
                        const size_t j = (size_t(z) * dy + y) * dx + x;
                        double sum = 0;
                        for (int iz = 0; iz < s; ++iz)
                            for (int iy = 0; iy < s; ++iy)
                                for (int ix = 0; ix < s; ++ix) {
                                    const size_t i = (size_t(z * s + iz) * ny + y * s + iy) * nx + x * s + ix;
                                    if (!std::isfinite(full[k][i]))
                                        throw std::runtime_error("Nonfinite input volume");
                                    sum += full[k][i];
                                    if (labels[i] != 1)
                                        mask[j] = 0;
                                }
                        low[k][j] = float(sum / (s * s * s) * 1000 / mu);
                    }
        }
        // Erode coarse tissue mask in all directions: keep bone/air boundaries out of fitting.
        auto valid = mask;
        for (int z = 1; z < dz - 1; ++z)
            for (int y = 1; y < dy - 1; ++y)
                for (int x = 1; x < dx - 1; ++x) {
                    const size_t j = (size_t(z) * dy + y) * dx + x;
                    for (int iz = -1; iz <= 1; ++iz)
                        for (int iy = -1; iy <= 1; ++iy)
                            for (int ix = -1; ix <= 1; ++ix)
                                if (!mask[(size_t(z + iz) * dy + y + iy) * dx + x + ix])
                                    valid[j] = 0;
                    const double hu = low[0][j] - 1000;
                    if (hu < -100 || hu > 180)
                        valid[j] = 0;
                }
        std::vector<EmpiricalBhcDifference> train, heldout;
        for (int z = 2; z < dz - 2; ++z)
            for (int y = 2; y < dy - 2; ++y)
                for (int x = 2; x < dx - 2; ++x) {
                    const size_t i = (size_t(z) * dy + y) * dx + x;
                    if (!valid[i])
                        continue;
                    for (size_t offset : { size_t(2), size_t(2 * dx), size_t(2 * dx * dy) }) {
                        const size_t j = i + offset;
                        if (!valid[j])
                            continue;
                        EmpiricalBhcDifference d { low[0][j] - low[0][i], low[1][j] - low[1][i], low[2][j] - low[2][i] };
                        if (std::abs(d.baseline) > 30)
                            continue; // fixed anatomical-edge exclusion, independent of fit
                        (((x / 8 + y / 8 + z / 8) % 5) == 0 ? heldout : train).push_back(d);
                    }
                }
        auto fit = fitEmpiricalBhc(train);
        auto holdFit = fitEmpiricalBhc(heldout);
        double before = 0, after = 0;
        for (const auto& d : heldout) {
            before += d.baseline * d.baseline;
            double e = d.baseline + fit.square * d.square + fit.cross * d.cross;
            after += e * e;
        }
        before = std::sqrt(before / heldout.size());
        after = std::sqrt(after / heldout.size());
        Json result = { { "schema", "ebhc-path-v1" }, { "square", fit.square }, { "cross", fit.cross }, { "path_scale_mm", 100 },
            { "objective", "Robust differences in eroded tissue; 4x box reduction, 2-cell differences, Huber 10 HU; no mono reference" },
            { "training_samples", train.size() }, { "heldout_samples", heldout.size() }, { "basis_correlation", fit.correlation },
            { "training_rms_before", fit.before }, { "training_rms_after", fit.after }, { "heldout_rms_before", before }, { "heldout_rms_after", after },
            { "heldout_independent_coefficients", { holdFit.square, holdFit.cross } }, { "inputs", job } };
        if (after > before)
            throw std::runtime_error("EBHC failed held-out flatness check");
        std::ofstream out(output / "reconstruction.raw", std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        double maxDelta = 0;
        std::array<double, 3> deltaSum {};
        std::array<size_t, 3> counts {};
        for (size_t i = 0; i < n; ++i) {
            const double delta = fit.square * full[1][i] + fit.cross * full[2][i];
            maxDelta = std::max(maxDelta, std::abs(delta) * 1000 / mu);
            if (labels[i] > 2)
                throw std::runtime_error("Invalid segmentation label");
            deltaSum[labels[i]] += delta * 1000 / mu;
            ++counts[labels[i]];
            full[0][i] += float(delta);
            if (!std::isfinite(full[0][i]))
                throw std::runtime_error("Invalid combined image");
        }
        out.write(reinterpret_cast<char*>(full[0].data()), n * 4);
        out.close();
        result["max_absolute_correction_HU"] = maxDelta;
        const double softMean = deltaSum[1] / std::max(size_t(1), counts[1]), boneMean = deltaSum[2] / std::max(size_t(1), counts[2]);
        // Experimental guardrails, not a universal physical acceptance criterion.
        // They depend only on the baseline/segmentation, never on a reference scan.
        result["mean_correction_HU"] = { { "soft", softMean }, { "bone", boneMean } };
        result["accepted"] = counts[1] > 0 && counts[2] > 0 && std::abs(softMean) <= 5 && std::abs(boneMean) <= 50;
        result["acceptance_policy"] = "Experimental: held-out flatness must improve; abs(mean soft correction)<=5 HU and abs(mean bone correction)<=50 HU. Passing is not proof of accuracy.";
        result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream(output / "coefficients.json") << result.dump(2);
        std::cout << result.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
