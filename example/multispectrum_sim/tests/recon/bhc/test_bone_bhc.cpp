#include "recon/bhc/BoneBhcStage.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
using namespace yk::spectral::bone_bhc;
namespace fs = std::filesystem;
void require(bool ok)
{
    if (!ok)
        throw std::runtime_error("iBHC test failed");
}
void write(const fs::path& p, float v)
{
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<char*>(&v), 4);
}
float read(const fs::path& p)
{
    float v = 0;
    std::ifstream f(p, std::ios::binary);
    f.read(reinterpret_cast<char*>(&v), 4);
    require(bool(f));
    return v;
}
class TestAlgorithm final : public IBoneBhcAlgorithm {
public:
    Domain input = Domain::ScatterCorrectedLog;
    bool iterative = true;
    bool fail_prepare = false;
    int resets = 0;
    std::function<std::optional<double>(const fs::path&, const fs::path&, const fs::path&)> callback;
    const char* id() const noexcept override { return "test"; }
    Domain inputDomain() const noexcept override { return input; }
    const char* errorMetric() const noexcept override { return "test_mse"; }
    bool supportsIteration() const noexcept override { return iterative; }
    void reset() noexcept override { ++resets; }
    void prepare(const Context&) override
    {
        if (fail_prepare)
            throw std::runtime_error("prepare failure");
    }
    std::optional<double> correct(const fs::path& v, const fs::path& p, const fs::path& o) override { return callback(v, p, o); }
};
int main(int argc, char** argv)
{
    try {
        require(argc == 2);
        const auto root = fs::path(argv[1]);
        fs::create_directories(root);
        // Unique directory avoids deleting user data on repeated tests.
        const auto dir = root / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        fs::create_directory(dir);
        write(dir / "input.raw", 10);
        Context c;
        c.output_directory = dir / "run";
        c.options.max_iterations = 8;
        int reconstructions = 0, models = 0;
        c.initialize = [](const auto& in, const auto& out) { write(out, read(in) + 1); };
        c.reconstruct = [&](const auto& in, const auto& out) {++reconstructions;write(out,read(in)); };
        auto algorithm = std::make_unique<TestAlgorithm>();
        auto* selected = algorithm.get();
        selected->callback = [&](const auto& volume, const auto& in, const auto& out) {
            require(read(in) == 10);
            require(read(volume) == (models == 0 ? 11.f : 12.f));
            ++models;
            write(out, read(in) + 2);
            return 1.;
        };
        auto r = std::make_shared<Result>();
        auto s = std::make_unique<BoneBhcStage>(std::move(algorithm), r);
        Dataset d;
        d.original_projection = dir / "input.raw";
        s->process(d, c);
        require(models == 4 && reconstructions == 5 && r->stopped_on_ratio);
        require(read(d.corrected_projection) == 12 && read(d.reconstruction) == 12 && read(d.original_projection) == 10);
        require(d.domain == Domain::ScatterCorrectedLog); // runtime owns transition
        bool rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected);
        c.output_directory = dir / "limit";
        c.options.max_iterations = 2;
        models = 0;
        s->process(d, c);
        require(r->iterations.size() == 2 && !r->stopped_on_ratio);
        c.output_directory = dir / "invalid";
        c.options.stop_ratio = 0;
        rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected && !fs::exists(c.output_directory));
        c.options.stop_ratio = .78;
        c.output_directory = dir / "zero";
        selected->callback = [](const auto&, const auto& in, const auto& out) {write(out,read(in));return 0.; };
        c.options.max_iterations = 8;
        s->process(d, c);
        require(r->iterations.size() == 4 && r->stopped_on_ratio);
        c.output_directory = dir / "no-metric";
        c.options.max_iterations = 5;
        selected->callback = [](const auto&, const auto& in, const auto& out) -> std::optional<double> {write(out,read(in));return std::nullopt; };
        s->process(d, c);
        require(r->iterations.size() == 5 && !r->stopped_on_ratio);
        c.output_directory = dir / "wrong-domain";
        d.domain = Domain::WaterCorrectedLog;
        rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected && !fs::exists(c.output_directory));
        selected->input = Domain::WaterCorrectedLog;
        rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected); // no double water init
        c.initialize = {};
        c.options.max_iterations = 1;
        c.output_directory = dir / "water";
        s->process(d, c);
        require(r->iterations.size() == 1 && read(d.corrected_projection) == 10);
        c.options.max_iterations = 2;
        selected->iterative = false;
        c.output_directory = dir / "unsupported";
        rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected && !fs::exists(c.output_directory));
        selected->iterative = true;
        c.output_directory = dir / "failure";
        selected->callback = [](const auto&, const auto&, const auto&) -> std::optional<double> { throw std::runtime_error("failure"); };
        const int resets = selected->resets;
        rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected && selected->resets == resets + 1);
        selected->fail_prepare = true;
        c.output_directory = dir / "prepare-failure";
        const int before_prepare = selected->resets;
        rejected = false;
        try {
            s->process(d, c);
        } catch (...) {
            rejected = true;
        }
        require(rejected && selected->resets == before_prepare + 1 && !fs::exists(c.output_directory));
        std::cout << "BoneBhcStage strategy, domain, fixed baseline, feedback, optional metric, cleanup and overwrite tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
