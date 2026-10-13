#include "pipeline/BuiltinPluginInfo.hpp"
#include "recon/postprocess/PostprocessPipeline.hpp"
#include "SimLogger.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::postprocess {
namespace {
    void validate(const Volume& v)
    {
        size_t count = 1;
        for (int i = 0; i < 3; ++i) {
            const int dim = v.grid.dimensions[i];
            if (dim <= 0 || count > std::numeric_limits<size_t>::max() / size_t(dim) || !std::isfinite(v.grid.spacing_mm[i]) || v.grid.spacing_mm[i] <= 0 || !std::isfinite(v.grid.origin_mm[i]))
                throw std::runtime_error("invalid postprocess grid");
            count *= size_t(dim);
        }
        for (double a : v.grid.direction)
            if (!std::isfinite(a))
                throw std::runtime_error("invalid postprocess direction");
        if (count != v.elements || count > std::numeric_limits<size_t>::max() / sizeof(float) || (v.location == Location::Host ? !v.host_data : !v.device_data))
            throw std::runtime_error("invalid postprocess storage");
    }
    void ensure(Volume& v, Location location, const Context& c)
    {
        if (v.location != location) {
            if (!c.transfer)
                throw std::runtime_error("postprocess data transfer bridge is missing");
            c.transfer->ensure(v, location, c);
        }
        if (v.location != location)
            throw std::runtime_error("postprocess transfer did not produce requested location");
        validate(v);
    }

}
void Pipeline::append(std::unique_ptr<IStage> stage)
{
    if (running_ || !stage)
        throw std::invalid_argument("invalid postprocess stage");
    stages_.push_back(std::move(stage));
}
void Pipeline::appendExporter(std::unique_ptr<IExporter> e)
{
    if (running_ || !e)
        throw std::invalid_argument("invalid postprocess exporter");
    exporters_.push_back(std::move(e));
}
void Pipeline::run(Volume& v, const Context& input_context)
{
    if (running_)
        throw std::runtime_error("postprocess pipeline is already running");
    timings_.clear();
    validate(v);
    Context c = input_context;
    c.grid = v.grid;
    auto domain = v.domain;
    auto location = v.location;
    for (const auto& stage : stages_) {
        const auto contract = stage->contract();
        domain = pipeline::advanceDomain(domain, contract, stage->info().name.c_str());
        if ((location != contract.input_location || contract.input_location != contract.output_location) && !c.transfer)
            throw std::runtime_error("postprocess data transfer bridge is missing");
        location = contract.output_location;
    }
    for (const auto& exporter : exporters_) {
        if (!exporter->acceptsDomain(domain))
            throw std::runtime_error(std::string(exporter->info().name.c_str()) + ": output domain mismatch");
        if (location != exporter->inputLocation() && !c.transfer)
            throw std::runtime_error("postprocess export transfer bridge is missing");
        location = exporter->inputLocation();
    }
    pipeline::RunScope run_scope(running_);
    auto cleanup = pipeline::onExit([&] {pipeline::resetReverse(exporters_);pipeline::resetReverse(stages_); });
    pipeline::Progress progress(c.progress, "Reconstruction postprocessing", int(stages_.size() + exporters_.size()));
    progress.start();
    auto record = [&](const char* name, double ms) {timings_.push_back({name,ms});SimLogger::instance().info(std::string("[postprocess] ")+name+" host_ms="+std::to_string(ms)); };
    for (auto& stage : stages_) {
        // Geometry is available through the typed context during preparation.
        c.grid=v.grid;
        stage->validateContext(c);
        stage->prepare(c);
        const auto timing = pipeline::execute(stage->info().name.c_str(), stage->contract(), &v, 1, [&](pipeline::Location location) { ensure(v, location, c); }, [&](Volume& value) { stage->process(value, c); }, [] {});
        validate(v);
        record(stage->info().name.c_str(), timing.process_ms);
        progress.advance();
    }
    for (auto& exporter : exporters_) {
        c.grid=v.grid;
        exporter->prepare(c);
        const pipeline::Contract<Domain> contract { v.domain, v.domain, exporter->inputLocation(), exporter->inputLocation() };
        const auto timing = pipeline::execute(exporter->info().name.c_str(), contract, &v, 1, [&](pipeline::Location location) { ensure(v, location, c); }, [&](Volume& value) { exporter->write(ReadOnlyVolume(value), c); }, [] {});
        exporter->finish();
        record(exporter->info().name.c_str(), timing.process_ms);
        progress.advance();
    }
}
}
