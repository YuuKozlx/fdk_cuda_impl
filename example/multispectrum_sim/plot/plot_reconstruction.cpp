#include <matplot/matplot.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input;
    std::string output = "reconstruction-profile.png";
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

void usage(const char* program) {
    std::cerr << "usage: " << program
              << " --input volume.raw --size NX NY NZ [--output figure.png]\n";
}

bool parse(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--input" && i + 1 < argc) {
            args.input = argv[++i];
        } else if (option == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (option == "--size" && i + 3 < argc) {
            args.nx = std::stoi(argv[++i]);
            args.ny = std::stoi(argv[++i]);
            args.nz = std::stoi(argv[++i]);
        } else {
            return false;
        }
    }
    return !args.input.empty() && args.nx > 0 && args.ny > 0 && args.nz > 0;
}

std::vector<float> read_volume(const Args& args) {
    const auto count = static_cast<std::size_t>(args.nx) * args.ny * args.nz;
    std::vector<float> volume(count);
    std::ifstream input(args.input, std::ios::binary);
    if (!input || !input.read(reinterpret_cast<char*>(volume.data()),
                              static_cast<std::streamsize>(count * sizeof(float)))) {
        throw std::runtime_error("cannot read float32 volume: " + args.input);
    }
    return volume;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    try {
        const auto volume = read_volume(args);
        const std::size_t plane = static_cast<std::size_t>(args.nx) * args.ny;
        const int cx = args.nx / 2;
        const int cy = args.ny / 2;
        std::vector<double> z(args.nz);
        std::vector<double> mean(args.nz);
        std::vector<double> axis(args.nz);
        for (int iz = 0; iz < args.nz; ++iz) {
            const auto begin = volume.begin() + static_cast<std::size_t>(iz) * plane;
            const auto end = begin + plane;
            double sum = 0.0;
            for (auto it = begin; it != end; ++it) sum += *it;
            z[iz] = iz;
            mean[iz] = sum / static_cast<double>(plane);
            axis[iz] = volume[static_cast<std::size_t>(iz) * plane +
                              static_cast<std::size_t>(cy) * args.nx + cx];
        }

        using namespace matplot;
        auto figure = gcf();
        figure->size(1200, 700);
        auto profile = subplot(2, 1, 0);
        profile->plot(z, mean, "-o")->line_width(1.5);
        profile->hold(on);
        profile->plot(z, axis, "-")->line_width(1.5);
        profile->grid(on);
        profile->xlabel("z index");
        profile->ylabel("value");
        profile->legend({"slice mean", "central axis"});

        auto histogram = subplot(2, 1, 1);
        histogram->hist(std::vector<double>(volume.begin(), volume.end()), 80);
        histogram->grid(on);
        histogram->xlabel("reconstruction value");
        histogram->ylabel("count");
        const auto output_path = std::filesystem::path(args.output).generic_string();
        if (!save(output_path)) {
            throw std::runtime_error("Matplot++/Gnuplot failed to save: " + args.output);
        }
        std::cout << "saved: " << args.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
