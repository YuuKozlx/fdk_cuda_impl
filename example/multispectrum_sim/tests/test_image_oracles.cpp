#include "image/ImageProcessing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
using namespace yk::spectral::image;
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
    check(caught, "invalid image options accepted");
}
Image make(int nx, int ny, int nz, int batches)
{
    Image v;
    v.grid.dimensions = { nx, ny, nz };
    v.grid.spacing_mm = { .5, .8, 1.3 };
    v.batches = batches;
    v.owner = allocate(Location::Host, v.elements());
    v.data = v.owner.get();
    return v;
}
std::vector<float> oracle(const Image& v, const Options& o)
{
    const int nx = v.grid.dimensions[0], ny = v.grid.dimensions[1], nz = v.grid.dimensions[2];
    std::vector<float> out(v.elements());
    for (int b = 0; b < v.batches; ++b)
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const size_t at = ((size_t(b) * nz + z) * ny + y) * nx + x;
                    double total = 0, weight = 0;
                    const int rz = o.neighborhood == Neighborhood::Slice2D ? 0 : o.radius[2];
                    for (int dz = -rz; dz <= rz; ++dz)
                        for (int dy = -o.radius[1]; dy <= o.radius[1]; ++dy)
                            for (int dx = -o.radius[0]; dx <= o.radius[0]; ++dx) {
                                const auto idx = ((size_t(b) * nz + std::clamp(z + dz, 0, nz - 1)) * ny + std::clamp(y + dy, 0, ny - 1)) * nx + std::clamp(x + dx, 0, nx - 1);
                                double w = 1;
                                if (o.operation != Operation::Mean) {
                                    const double qx = dx * v.grid.spacing_mm[0] / o.sigma_mm[0], qy = dy * v.grid.spacing_mm[1] / o.sigma_mm[1], qz = dz * v.grid.spacing_mm[2] / o.sigma_mm[2];
                                    w = std::exp(-.5 * (qx * qx + qy * qy + qz * qz));
                                }
                                if (o.operation == Operation::Bilateral) {
                                    const double d = (double(v.data[idx]) - v.data[at]) / o.range_sigma;
                                    w *= std::exp(-.5 * d * d);
                                }
                                total += w * v.data[idx];
                                weight += w;
                            }
                    out[at] = float(total / weight);
                }
    return out;
}
void compare(const Image& result, const std::vector<float>& expected, double inputScale = 0)
{
    check(result.elements() == expected.size(), "shape mismatch");
    for (size_t i = 0; i < expected.size(); ++i)
        if (!std::isfinite(result.data[i]) || std::abs(result.data[i] - expected[i]) > 2e-5 * std::max(1.f, std::abs(expected[i])) + 16 * std::numeric_limits<float>::epsilon() * inputScale)
            throw std::runtime_error("oracle mismatch at " + std::to_string(i) + " got=" + std::to_string(result.data[i]) + " expected=" + std::to_string(expected[i]));
}
int main(int argc, char** argv)
{
    try {
        check(argc == 2, "case required");
        const std::string name = argv[1];
        if (name == "invalid") {
            for (int k = 0; k < 10; ++k) {
                Options o;
                switch (k) {
                case 0:
                    o.operation = Operation(99);
                    break;
                case 1:
                    o.neighborhood = Neighborhood(99);
                    break;
                case 2:
                    o.interpolation = Interpolation(99);
                    break;
                case 3:
                    o.radius[1] = -1;
                    break;
                case 4:
                    o.radius[2] = 4097;
                    break;
                case 5:
                    o.sigma_mm[0] = 0;
                    break;
                case 6:
                    o.range_sigma = 0;
                    break;
                case 7:
                    o.epsilon = -1;
                    break;
                case 8:
                    o.output_dimensions[0] = 0;
                    break;
                case 9:
                    o.epsilon = std::numeric_limits<double>::quiet_NaN();
                    break;
                }
                reject([&] { validate(o); });
            }
            auto v = make(3, 2, 1, 1);
            for (int k = 0; k < 6; ++k) {
                auto bad = v;
                if (k == 0)
                    bad.batches = 0;
                if (k == 1)
                    bad.data = nullptr;
                if (k == 2)
                    bad.grid.spacing_mm[1] = 0;
                if (k == 3)
                    bad.grid.direction[0] = 2;
                if (k == 4)
                    bad.grid.origin_mm[1] = std::numeric_limits<double>::infinity();
                if (k == 5)
                    bad.grid.dimensions[0] = 0;
                reject([&] { validate(bad); });
            }
        } else
            for (auto location : { Location::Host, Location::Device }) {
                Workspace workspace(location);
                if (name == "resample") {
                    for (int mode = 0; mode < 3; ++mode) {
                        auto input = make(mode == 2 ? 5 : 3, 1, 1, 2);
                        for (size_t i = 0; i < input.elements(); ++i)
                            input.data[i] = float((i % input.grid.dimensions[0]) * 10 + (i / input.grid.dimensions[0]) * 100);
                        Options o;
                        o.operation = Operation::Resample;
                        o.output_dimensions = { mode == 2 ? 3 : 5, 1, 1 };
                        o.interpolation = mode == 1 ? Interpolation::Nearest : Interpolation::Linear;
                        std::vector<float> e = mode == 2 ? std::vector<float> { 4, 20, 36 } : mode == 1 ? std::vector<float> { 0, 0, 10, 20, 20 }
                                                                                                        : std::vector<float> { 0, 4, 10, 16, 20 };
                        const auto first = e;
                        for (float x : first)
                            e.push_back(x + 100);
                        ensureLocation(input, location);
                        process(input, o, workspace);
                        ensureLocation(input, Location::Host);
                        compare(input, e);
                    }
                    // Direct tensor-product oracle: mixed up/down sampling, singleton
                    // axes, non-cubic shapes, two batches, and anti-alias on/off.
                    for (auto dims : { std::array<int, 3> { 3, 4, 2 }, std::array<int, 3> { 1, 3, 1 } })
                        for (auto target : { std::array<int, 3> { 5, 2, 3 }, std::array<int, 3> { 2, 5, 1 }, std::array<int, 3> { 1, 1, 1 } })
                            for (bool nearest : { false, true })
                                for (bool antialias : { false, true }) {
                                    auto input = make(dims[0], dims[1], dims[2], 2);
                                    for (size_t i = 0; i < input.elements(); ++i)
                                        input.data[i] = float(std::sin(i * .7) + i * .13);
                                    Options o;
                                    o.operation = Operation::Resample;
                                    o.output_dimensions = target;
                                    o.interpolation = nearest ? Interpolation::Nearest : Interpolation::Linear;
                                    o.antialias = antialias;
                                    auto weights = [&](int axis, int c) {
                                        std::vector<double> w(dims[axis]);
                                        double scale = double(dims[axis]) / target[axis];
                                        if (nearest)
                                            w[std::clamp(int(std::floor((c + .5) * scale)), 0, dims[axis] - 1)] = 1;
                                        else if (antialias && scale > 1) {
                                            for (int j = 0; j < dims[axis]; ++j)
                                                w[j] = std::max(0., std::min((c + 1) * scale, double(j + 1)) - std::max(c * scale, double(j))) / scale;
                                        } else {
                                            double u = (c + .5) * scale - .5;
                                            int j = int(std::floor(u));
                                            double t = u - j;
                                            w[std::clamp(j, 0, dims[axis] - 1)] += 1 - t;
                                            w[std::clamp(j + 1, 0, dims[axis] - 1)] += t;
                                        }
                                        return w;
                                    };
                                    std::vector<float> expected;
                                    for (int b = 0; b < 2; ++b)
                                        for (int z = 0; z < target[2]; ++z)
                                            for (int y = 0; y < target[1]; ++y)
                                                for (int x = 0; x < target[0]; ++x) {
                                                    auto wx = weights(0, x), wy = weights(1, y), wz = weights(2, z);
                                                    double sum = 0;
                                                    for (int iz = 0; iz < dims[2]; ++iz)
                                                        for (int iy = 0; iy < dims[1]; ++iy)
                                                            for (int ix = 0; ix < dims[0]; ++ix)
                                                                sum += wx[ix] * wy[iy] * wz[iz] * input.data[((b * dims[2] + iz) * dims[1] + iy) * dims[0] + ix];
                                                    expected.push_back(float(sum));
                                                }
                                    ensureLocation(input, location);
                                    process(input, o, workspace);
                                    ensureLocation(input, Location::Host);
                                    compare(input, expected);
                                }
                } else {
                    Options o;
                    if (name == "mean")
                        o.operation = Operation::Mean;
                    else if (name == "gaussian")
                        o.operation = Operation::Gaussian;
                    else if (name == "bilateral")
                        o.operation = Operation::Bilateral;
                    else if (name == "guided")
                        o.operation = Operation::Guided;
                    else
                        throw std::runtime_error("unknown case");
                    o.radius = { 2, 1, 1 };
                    o.sigma_mm = { .7, 1.1, 1.8 };
                    o.range_sigma = .4;
                    o.epsilon = .03;
                    for (int pattern = 0; pattern < 7; ++pattern)
                        for (int radius : { 0, 1, 3 })
                            for (double scale : { .001, 1., 1000. })
                                for (bool singleton : { false, true })
                                    for (auto neighborhood : { Neighborhood::Slice2D, Neighborhood::Spatial3D }) {
                                        o.neighborhood = neighborhood;
                                        o.radius = { radius, radius / 2, radius };
                                        o.range_sigma = .4 * scale;
                                        o.epsilon = .03 * scale * scale;
                                        auto input = make(singleton ? 1 : 5, singleton ? 1 : 4, singleton ? 1 : 3, 2);
                                        std::mt19937 rng(1234);
                                        std::uniform_real_distribution<float> noise(-1, 1);
                                        for (size_t i = 0; i < input.elements(); ++i) {
                                            const size_t local = i % (input.elements() / input.batches);
                                            double value = 0;
                                            if (pattern == 0)
                                                value = .3 * std::sin(i * .71);
                                            if (pattern == 1)
                                                value = .4;
                                            if (pattern == 2)
                                                value = double(local) / input.elements();
                                            if (pattern == 3)
                                                value = local == 0 ? 1 : 0;
                                            if (pattern == 4)
                                                value = (local % 2) ? 1 : -1;
                                            if (pattern == 5)
                                                value = (local % input.grid.dimensions[0]) >= input.grid.dimensions[0] / 2 ? 1 : 0;
                                            if (pattern == 6)
                                                value = noise(rng);
                                            input.data[i] = float(scale * (value + (i / (input.elements() / input.batches)) * .8));
                                        }
                                        std::vector<float> expected;
                                        if (o.operation == Operation::Guided) {
                                            const int nx = input.grid.dimensions[0], ny = input.grid.dimensions[1], nz = input.grid.dimensions[2];
                                            auto mean = [&](const std::vector<double>& values) {
                                                std::vector<double> out(values.size());
                                                for (int batch = 0; batch < 2; ++batch)
                                                    for (int z = 0; z < nz; ++z)
                                                        for (int y = 0; y < ny; ++y)
                                                            for (int x = 0; x < nx; ++x) {
                                                                double sum = 0;
                                                                int count = 0;
                                                                const int rz = o.neighborhood == Neighborhood::Slice2D ? 0 : o.radius[2];
                                                                for (int dz = -rz; dz <= rz; ++dz)
                                                                    for (int dy = -o.radius[1]; dy <= o.radius[1]; ++dy)
                                                                        for (int dx = -o.radius[0]; dx <= o.radius[0]; ++dx) {
                                                                            sum += values[((batch * nz + std::clamp(z + dz, 0, nz - 1)) * ny + std::clamp(y + dy, 0, ny - 1)) * nx + std::clamp(x + dx, 0, nx - 1)];
                                                                            ++count;
                                                                        }
                                                                out[((batch * nz + z) * ny + y) * nx + x] = sum / count;
                                                            }
                                                return out;
                                            };
                                            std::vector<double> values(input.data, input.data + input.elements()), square = values;
                                            for (double& v : square)
                                                v *= v;
                                            auto means = mean(values), second = mean(square), a = values, b = values;
                                            for (size_t i = 0; i < values.size(); ++i) {
                                                double variance = std::max(0., second[i] - means[i] * means[i]);
                                                a[i] = variance / (variance + o.epsilon);
                                                b[i] = (1 - a[i]) * means[i];
                                            }
                                            auto ma = mean(a), mb = mean(b);
                                            for (size_t i = 0; i < values.size(); ++i)
                                                expected.push_back(float(ma[i] * values[i] + mb[i]));
                                        } else
                                            expected = oracle(input, o);
                                        auto original = input.owner;
                                        std::vector<float> saved(input.data, input.data + input.elements());
                                        ensureLocation(input, location);
                                        process(input, o, workspace);
                                        workspace.trim();
                                        ensureLocation(input, Location::Host);
                                        try {
                                            compare(input, expected, 2 * scale);
                                        } catch (const std::exception& e) {
                                            throw std::runtime_error(std::string(e.what()) + " pattern=" + std::to_string(pattern) + " radius=" + std::to_string(radius) + " scale=" + std::to_string(scale) + " location=" + std::to_string(int(location)) + " neighborhood=" + std::to_string(int(neighborhood)));
                                        }
                                        check(std::equal(saved.begin(), saved.end(), original.get()), "filter mutated borrowed input");
                                    }
                }
            }
        std::cout << name << " CPU/CUDA independent oracle passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
