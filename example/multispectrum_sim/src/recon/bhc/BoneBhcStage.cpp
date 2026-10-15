#include "recon/bhc/BoneBhcStage.hpp"
#include <cmath>

namespace yk::spectral::bone_bhc {
BoneBhcStage::BoneBhcStage(std::unique_ptr<IBoneBhcAlgorithm> algorithm, std::shared_ptr<Result> result)
    : algorithm_(std::move(algorithm))
    , result_(std::move(result))
{
    if (!algorithm_ || !result_)
        throw std::invalid_argument("BoneBhcStage requires an algorithm and result");
}
void BoneBhcStage::reset() noexcept { algorithm_->reset(); }
const pipeline::PluginInfo& BoneBhcStage::info() const noexcept { return boneBhcInfo(); }
pipeline::Contract<Domain> BoneBhcStage::contract() const
{
    return { algorithm_->inputDomain(), Domain::ReferenceEnergyLog };
}
void BoneBhcStage::validateContext(const Context& c) const
{
    if (!c.reconstruct || c.output_directory.empty() || c.options.max_iterations < 1 || !std::isfinite(c.options.stop_ratio) || c.options.stop_ratio <= 0 || c.options.stop_ratio > 1)
        throw std::invalid_argument("invalid iBHC context/options");
}
void BoneBhcStage::process(Dataset& data, const Context& c)
{
    validateContext(c);
    if (data.domain != algorithm_->inputDomain() || !std::filesystem::is_regular_file(data.original_projection))
        throw std::invalid_argument("BoneBhcStage input file/domain does not match the selected algorithm");
    if (c.initialize && data.domain != Domain::ScatterCorrectedLog)
        throw std::invalid_argument("initial water correction is forbidden on water-corrected input");
    if (c.options.max_iterations > 1 && !algorithm_->supportsIteration())
        throw std::invalid_argument("algorithm does not support iteration");
    auto cleanup = pipeline::onExit([&] { algorithm_->reset(); });
    algorithm_->prepare(c);
    // Exclusive directory: never overwrite inputs, checkpoints or another run.
    if (!std::filesystem::create_directory(c.output_directory))
        throw std::runtime_error("iBHC output directory must not already exist");
    *result_ = {};
    const auto initial = c.output_directory / "initial.raw";
    auto input = data.original_projection;
    if (c.initialize) {
        input = c.output_directory / "initial-projection.raw";
        c.initialize(data.original_projection, input);
    }
    c.reconstruct(input, initial);
    auto volume = initial;
    std::filesystem::path corrected;
    for (int k = 1; k <= c.options.max_iterations; ++k) {
        const auto directory = c.output_directory / ("iteration-" + std::to_string(k));
        std::filesystem::create_directory(directory);
        corrected = directory / "projection.raw";
        Iteration entry;
        entry.number = k;
        auto start = pipeline::Clock::now();
        entry.model_error = algorithm_->correct(volume, data.original_projection, corrected);
        entry.model_ms = pipeline::elapsedMilliseconds(start);
        if (entry.model_error && (!std::isfinite(*entry.model_error) || *entry.model_error < 0))
            throw std::runtime_error("invalid iBHC model error");
        volume = directory / "reconstruction.raw";
        start = pipeline::Clock::now();
        c.reconstruct(corrected, volume);
        entry.reconstruction_ms = pipeline::elapsedMilliseconds(start);
        result_->iterations.push_back(entry);
        if (c.progress)
            c.progress(entry);
        const auto& r = result_->iterations;
        if (r.size() >= 4 && r[r.size() - 1].model_error && r[r.size() - 2].model_error && r[r.size() - 3].model_error && r[r.size() - 4].model_error) {
            const auto n = r.size();
            const double recent = *r[n - 1].model_error + *r[n - 2].model_error;
            const double previous = *r[n - 3].model_error + *r[n - 4].model_error;
            // Paper Eq. 7; zero residual also terminates without 0/0.
            if ((previous == 0 && recent == 0) || (previous > 0 && recent / previous > c.options.stop_ratio)) {
                result_->stopped_on_ratio = true;
                break;
            }
        }
    }
    data.corrected_projection = corrected;
    data.reconstruction = volume;
    // Domain transitions belong to the caller/runtime, as for other stages.
}
}
