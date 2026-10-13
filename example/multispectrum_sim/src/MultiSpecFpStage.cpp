#include "MultiSpecFpStage.hpp"
#include "pipeline/BuiltinPluginInfo.hpp"
#include <algorithm>
#include <stdexcept>
namespace yk::spectral {
MultiSpecFpStage::MultiSpecFpStage(std::unique_ptr<IProjectionBackend> backend)
    : backend_(std::move(backend))
{
    if (!backend_)
        throw std::invalid_argument("spectral source backend is required");
}
MultiSpecFpStage::~MultiSpecFpStage() { reset(); }
const pipeline::PluginInfo& MultiSpecFpStage::info() const noexcept { return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::MultiSpecFp); }
pipeline::SourceContract<ProjectionSignalDomain> MultiSpecFpStage::contract() const { return { backend_->outputDomain(), backend_->supportsDeviceFrames() ? pipeline::Location::Device : pipeline::Location::Host }; }
void MultiSpecFpStage::requirePrepared() const
{
    if (!prepared_)
        throw std::runtime_error("spectral source is not prepared");
}
void MultiSpecFpStage::prepare(const SpectralSourceContext& context)
{
    if (prepared_)
        throw std::runtime_error("spectral source is already prepared");
    const auto& g = context.config.geometry.parameters;
    if (g.views <= 0 || g.detector_u <= 0 || g.detector_v <= 0)
        throw std::invalid_argument("invalid spectral source geometry");
    try {
        resources_started_ = true;
        backend_->prepare(context.config, context.labels, context.model);
        capacity_ = std::min(g.views, backend_->batchCapacity());
        if (capacity_ <= 0)
            throw std::runtime_error("invalid spectral source batch capacity");
        views_ = g.views;
        width_ = g.detector_u;
        height_ = g.detector_v;
        prepared_ = true;
    } catch (...) {
        reset();
        throw;
    }
}
int MultiSpecFpStage::batchCapacity() const
{
    requirePrepared();
    return capacity_;
}
void MultiSpecFpStage::generate(int first, int count, std::vector<ProjectionFrame>& frames)
{
    requirePrepared();
    if (first < 0 || count <= 0 || count > capacity_ || first >= views_ || count > views_ - first)
        throw std::invalid_argument("invalid spectral source view batch");
    frames.resize(count);
    const auto output = contract();
    for (auto& frame : frames) {
        frame.domain = output.output_domain;
        frame.resize(width_, height_);
        frame.logarithm_valid = false;
    }
    auto* storage = frames.data();
    backend_->projectBatch(first, count, frames);
    if (frames.size() != size_t(count) || frames.data() != storage)
        throw std::runtime_error("backend changed source batch storage");
    for (int slot = 0; slot < count; ++slot) {
        auto& frame = frames[slot];
        if (frame.view != first + slot || frame.domain != output.output_domain || frame.location != output.output_location || frame.width != width_ || frame.height != height_)
            throw std::runtime_error("backend returned an incompatible spectral source frame");
        if (frame.location == ProjectionMemoryLocation::Host) {
            frame.signal_energy = frame.expected_primary_energy;
            frame.reference_energy = frame.air_energy;
            frame.photon_counts.reference_channel_counts = frame.photon_counts.air_channel_counts;
        }
    }
}
void MultiSpecFpStage::ensureLocation(std::vector<ProjectionFrame>& frames, ProjectionMemoryLocation location)
{
    requirePrepared();
    auto* storage = frames.data();
    const auto count = frames.size();
    backend_->ensureLocation(frames, location);
    if (storage != frames.data() || count != frames.size())
        throw std::runtime_error("backend changed source transfer storage");
    for (const auto& frame : frames)
        if (frame.location != location)
            throw std::runtime_error("source transfer location mismatch");
}
void MultiSpecFpStage::finalizeBatch(std::vector<ProjectionFrame>& frames)
{
    requirePrepared();
    backend_->finalizeBatch(frames);
}
void MultiSpecFpStage::synchronize()
{
    requirePrepared();
    backend_->synchronize();
}
ProjectionViewGeometry MultiSpecFpStage::geometryForDisplay(int view) const
{
    requirePrepared();
    if (view < 0 || view >= views_)
        throw std::out_of_range("invalid spectral display view");
    return backend_->geometryForDisplay(view);
}
void MultiSpecFpStage::sampleScatterNoise(ProjectionFrame& frame)
{
    requirePrepared();
    backend_->sampleScatterNoise(frame);
}
void MultiSpecFpStage::reset() noexcept
{
    if (resources_started_)
        backend_->reset();
    resources_started_ = false;
    prepared_ = false;
    views_ = width_ = height_ = capacity_ = 0;
}
}
