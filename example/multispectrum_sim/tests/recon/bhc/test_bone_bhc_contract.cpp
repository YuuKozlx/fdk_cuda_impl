#include "recon/bhc/BoneBhcStage.hpp"
#include "recon/bhc/BoneBhcRouting.hpp"
#include <fstream>
#include <iostream>
#include <limits>
using namespace yk::spectral::bone_bhc;
namespace fs = std::filesystem;
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
    check(caught, "expected rejection");
}
struct Algorithm : IBoneBhcAlgorithm {
    int resets = 0, calls = 0;
    std::vector<std::optional<double>> errors;
    const char* id() const noexcept override { return "contract-test"; }
    Domain inputDomain() const noexcept override { return Domain::ScatterCorrectedLog; }
    const char* errorMetric() const noexcept override { return "test"; }
    bool supportsIteration() const noexcept override { return true; }
    void reset() noexcept override { ++resets; }
    std::optional<double> correct(const fs::path&, const fs::path&, const fs::path& out) override
    {
        std::ofstream(out) << "projection";
        return errors.at(calls++);
    }
};
int main(int argc, char** argv)
{
    try {
        yk::spectral::SimulationConfig config;
        auto& b=config.reconstruction.bone_hardening;b.enabled=true;
        config.reconstruction.preprocessing.beam_hardening.enabled=true;
        auto route=resolveRouting(config);check(!route.before_water&&!route.use_lut,"physical-after-water routing");
        b.method="physical_before_water";route=resolveRouting(config);check(route.before_water&&!route.use_lut,"physical-before-water routing");
        b.method="two_material_lut";rejects([&]{resolveRouting(config);});
        b.calibration_file="calibration.json";route=resolveRouting(config);check(route.before_water&&route.use_lut,"LUT must replace water BHC");
        b.method="ebhc";rejects([&]{resolveRouting(config);}); // image-domain stage is not a dataset strategy
        b.method="physical_after_water";b.max_iterations=0;rejects([&]{resolveRouting(config);});b.max_iterations=1;
        b.air_threshold_hu=b.bone_threshold_hu;rejects([&]{resolveRouting(config);});b.air_threshold_hu=-500;
        config.reconstruction.preprocessing.beam_hardening.enabled=false;rejects([&]{resolveRouting(config);});
        config.reconstruction.preprocessing.beam_hardening.enabled=true;
        config.geometry.kind=yk::spectral::GeometryKind::NikonTilted;rejects([&]{resolveRouting(config);});
        check(argc == 2, "output root required");
        const auto root = fs::path(argv[1]) / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        fs::create_directories(root);
        std::ofstream(root / "input") << "original";
        rejects([] { BoneBhcStage stage(nullptr, std::make_shared<Result>()); });
        rejects([] { BoneBhcStage stage(std::make_unique<Algorithm>(), nullptr); });
        int runs = 0;
        // Every failed phase must clean resources and must not publish partial output paths.
        for (const auto phase : { "initialize", "initial-reconstruct", "reconstruct", "progress", "nan", "infinity", "negative" }) {
            auto a = std::make_unique<Algorithm>();
            auto* raw = a.get();
            const std::string failure = phase;
            double error = 1;
            if (failure == "nan")
                error = std::numeric_limits<double>::quiet_NaN();
            if (failure == "infinity")
                error = std::numeric_limits<double>::infinity();
            if (failure == "negative")
                error = -1;
            raw->errors = { error };
            auto result = std::make_shared<Result>();
            BoneBhcStage stage(std::move(a), result);
            Context c;
            c.output_directory = root / std::to_string(runs++);
            int reconstructions = 0, notifications = 0;
            c.initialize = [&](auto&, auto&) { if (failure == "initialize") throw std::runtime_error("injected"); };
            c.reconstruct = [&](auto&, const auto& out) {
                ++reconstructions;
                if ((failure == "initial-reconstruct" && reconstructions == 1) || (failure == "reconstruct" && reconstructions == 2))
                    throw std::runtime_error("injected");
                std::ofstream(out) << "volume";
            };
            c.progress = [&](auto&) { ++notifications; if (failure == "progress") throw std::runtime_error("injected"); };
            Dataset d { root / "input", "previous-projection", "previous-volume", Domain::ScatterCorrectedLog };
            rejects([&] { stage.process(d, c); });
            check(raw->resets == 1, "exception leaked algorithm resources");
            check(d.corrected_projection == "previous-projection" && d.reconstruction == "previous-volume", "failure published partial outputs");
            check(d.original_projection == root / "input" && d.domain == Domain::ScatterCorrectedLog, "failure changed input");
            if (failure == "nan" || failure == "infinity" || failure == "negative")
                check(reconstructions == 1 && notifications == 0 && result->iterations.empty(), "invalid metric reached reconstruction/progress");
        }
        // A rapidly falling residual must continue; a stalled residual must stop.
        for (int mode = 0; mode < 4; ++mode) {
            auto a = std::make_unique<Algorithm>();
            auto* raw = a.get();
            if (mode == 0)
                raw->errors = { 8., 4., 2., 1., .5, .25 };
            if (mode == 1)
                raw->errors = { 8., 4., 8., 4., 8., 4. };
            if (mode == 2)
                raw->errors = { 1., 1., std::nullopt, 1., 1., 1. };
            if (mode == 3)
                raw->errors = { 1., 1., .5, .5, .25, .25 };
            auto result = std::make_shared<Result>();
            BoneBhcStage stage(std::move(a), result);
            Context c;
            c.output_directory = root / std::to_string(runs++);
            c.options.max_iterations = 6;
            c.options.stop_ratio = mode == 3 ? .5 : .78;
            int notices = 0;
            c.progress = [&](const Iteration& i) { check(i.number == ++notices, "progress order"); check(i.model_ms >= 0 && i.reconstruction_ms >= 0, "negative duration"); };
            c.reconstruct = [](auto&, const auto& out) { std::ofstream(out) << "volume"; };
            Dataset d;
            d.original_projection = root / "input";
            stage.process(d, c);
            const int expected = mode == 1 ? 4 : 6;
            check(raw->calls == expected && notices == expected && raw->resets == 1, "stopping/lifecycle mismatch");
            check(result->stopped_on_ratio == (mode == 1), "ratio boundary/optional metric mismatch");
        }
        auto result = std::make_shared<Result>();
        BoneBhcStage stage(std::make_unique<Algorithm>(), result);
        Context valid;
        valid.output_directory = root / "unused";
        valid.reconstruct = [](auto&, auto&) { };
        for (double ratio : { 0., -1., 1.01, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() }) {
            auto c = valid;
            c.options.stop_ratio = ratio;
            rejects([&] { stage.validateContext(c); });
        }
        auto c = valid;
        c.options.max_iterations = 0;
        rejects([&] { stage.validateContext(c); });
        c = valid;
        c.reconstruct = {};
        rejects([&] { stage.validateContext(c); });
        c = valid;
        c.output_directory.clear();
        rejects([&] { stage.validateContext(c); });
        Dataset missing;
        missing.original_projection = root / "absent";
        rejects([&] { stage.process(missing, valid); });
        check(!fs::exists(valid.output_directory), "invalid input created output");
        std::cout << "BoneBhc failure isolation, progress, stopping boundaries and context tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
