#include "SpectralProjectionPipeline.hpp"
#include "primary/ProjectionStages.hpp"
#include "primary/ProjectionOutput.hpp"
#include "SimLogger.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace yk::spectral {
void ProjectionFrame::resize(int columns, int rows) {
    if (columns <= 0 || rows <= 0) throw std::runtime_error("invalid projection frame dimensions");
    width = columns; height = rows;
    const auto pixels = static_cast<size_t>(width) * height;
    expected_primary_energy.resize(pixels); sampled_primary_energy.resize(pixels);
    air_energy.resize(pixels); signal_energy.resize(pixels);
    reference_energy.resize(pixels); object_mask.resize(pixels);
}
SpectralProjectionPipeline::SpectralProjectionPipeline(std::unique_ptr<IProjectionBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_) throw std::invalid_argument("projection backend is required");
}
SpectralProjectionPipeline::~SpectralProjectionPipeline() = default;
void SpectralProjectionPipeline::addStage(std::unique_ptr<IProjectionStage> stage) {
    if (running_ || !stage) throw std::invalid_argument("invalid projection stage registration");
    extensions_.push_back(std::move(stage));
}
void SpectralProjectionPipeline::setOutputSink(std::unique_ptr<IProjectionOutputSink> sink) {
    if (running_ || !sink) throw std::invalid_argument("invalid projection output sink");
    sink_ = std::move(sink);
}
bool SpectralProjectionPipeline::run(const SimulationConfig& c,
    const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model,
    const std::filesystem::path& output_file, std::string& error) {
    if (running_) { error = "projection pipeline is already running"; return false; }
    error.clear();
    auto builtins = primary_detail::makeProjectionStages(c);
    std::vector<IProjectionStage*> stages;
    for (auto& stage : builtins) stages.push_back(stage.get());
    for (auto& stage : extensions_) stages.push_back(stage.get());
    std::stable_sort(stages.begin(), stages.end(), [](auto* a, auto* b) {
        return a->phase() < b->phase();
    });
    primary_detail::EnergyProjectionOutput energy_output;
    IProjectionOutputSink& output = sink_ ? *sink_ : energy_output;
    running_ = true;
    // Reset all modules and device resources on both success and failure.
    struct Cleanup {
        IProjectionBackend& backend;
        std::vector<IProjectionStage*>& stages;
        bool& running;
        IProjectionOutputSink& output;
        ~Cleanup() {
            for (auto it = stages.rbegin(); it != stages.rend(); ++it) (*it)->reset();
            backend.reset(); output.reset(); running = false;
        }
    } cleanup{*backend_, stages, running_, output};
    try {
        if (backend_->outputDomain() != output.domain())
            throw std::runtime_error("projection backend and output sink signal domains do not match");
        for (auto* stage : stages) {
            if (stage->domain() != backend_->outputDomain() ||
                stage->location() != ProjectionMemoryLocation::Host)
                throw std::runtime_error(std::string("unsupported signal or memory domain for stage: ") + stage->name());
            stage->reset();
            stage->prepare(c);
        }
        backend_->prepare(c, labels, model);
        const auto& g = c.geometry.parameters;
        if (g.views <= 0) throw std::runtime_error("projection pipeline requires positive views");
        ProjectionFrame frame;
        frame.domain = backend_->outputDomain();
        frame.resize(g.detector_u, g.detector_v);
        output.prepare(c, output_file);
        ProjectionStageContext context{c, *backend_};
        using Clock = std::chrono::steady_clock;
        auto elapsed = [](Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };
        std::vector<double> stage_ms(stages.size(), 0.0);
        double primary_ms = 0.0, output_ms = 0.0;
        const auto start = Clock::now();
        for (int view = 0; view < g.views; ++view) {
            frame.view = view;
            auto tick = Clock::now();
            backend_->project(view, frame);
            primary_ms += elapsed(tick);
            if (frame.domain != backend_->outputDomain() ||
                frame.location != ProjectionMemoryLocation::Host ||
                frame.width != g.detector_u || frame.height != g.detector_v || frame.view != view)
                throw std::runtime_error("backend returned an incompatible projection frame");
            if (frame.domain == ProjectionSignalDomain::EnergyKeV)
                frame.signal_energy = frame.expected_primary_energy;
            for (size_t i = 0; i < stages.size(); ++i) {
                tick = Clock::now();
                stages[i]->process(frame, context);
                if (frame.domain != stages[i]->domain() || frame.location != stages[i]->location())
                    throw std::runtime_error(std::string("stage changed signal or memory domain: ") + stages[i]->name());
                stage_ms[i] += elapsed(tick);
            }
            {
                // Match the current view's geometry, exposure and response.
                // Air modules advance their own temporal state for every view.
                frame.reference_energy = frame.air_energy;
                frame.photon_counts.reference_channel_counts = frame.photon_counts.air_channel_counts;
                for (size_t i = 0; i < stages.size(); ++i) {
                    tick = Clock::now();
                    stages[i]->processAir(frame, context);
                    stage_ms[i] += elapsed(tick);
                }
            }
            tick = Clock::now();
            output.write(frame);
            output_ms += elapsed(tick);
            if (view == 0 || view + 1 == g.views || (view + 1) % 10 == 0)
                std::cerr << "projection frame " << view + 1 << "/" << g.views << std::endl;
        }
        output.finish();
        std::ostringstream timing;
        timing << std::fixed << std::setprecision(1)
            << "projection_pipeline_timing_ms total=" << elapsed(start)
            << " primary_and_transfer=" << primary_ms;
        for (size_t i = 0; i < stages.size(); ++i)
            timing << " " << stages[i]->name() << "=" << stage_ms[i];
        timing << " output=" << output_ms;
        SimLogger::instance().info(timing.str());
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace yk::spectral
