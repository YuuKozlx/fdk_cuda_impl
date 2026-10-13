#include "common/YkTimingWatchdog.hpp"
#include "SpectralProjectionPipeline.hpp"
#include "SimLogger.hpp"
#include "primary/ProjectionOutput.hpp"
#include "primary/ProjectionStages.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace yk::spectral {
void IProjectionBackend::ensureLocation(std::vector<ProjectionFrame>& frames,ProjectionMemoryLocation location){
    for(auto& f:frames)if(f.location!=location)throw std::runtime_error("backend does not support the requested memory location");
}
void IProjectionBackend::projectBatch(int first,int count,std::vector<ProjectionFrame>& frames){
    if(count<=0||frames.size()!=size_t(count))throw std::runtime_error("invalid projection batch storage");
    for(int i=0;i<count;++i){frames[i].view=first+i;project(first+i,frames[i]);}
}
ProjectionViewGeometry IProjectionBackend::geometryForDisplay(int) const
{
    throw std::runtime_error("projection backend does not provide display geometry");
}
void ProjectionFrame::resize(int columns, int rows)
{
    if (columns <= 0 || rows <= 0)
        throw std::runtime_error("invalid projection frame dimensions");
    width = columns;
    height = rows;
    const auto pixels = static_cast<size_t>(width) * height;
    expected_primary_energy.resize(pixels);
    sampled_primary_energy.resize(pixels);
    air_energy.resize(pixels);
    signal_energy.resize(pixels);
    reference_energy.resize(pixels);
    object_mask.resize(pixels);
}
SpectralProjectionPipeline::SpectralProjectionPipeline(std::unique_ptr<IProjectionSource> source)
    : source_(std::move(source))
{
    if (!source_)
        throw std::invalid_argument("projection source stage is required");
}
SpectralProjectionPipeline::~SpectralProjectionPipeline() = default;
const pipeline::PluginInfo& SpectralProjectionPipeline::sourceInfo()const noexcept{return source_->info();}
void SpectralProjectionPipeline::addStage(std::unique_ptr<IProjectionStage> stage)
{
    if (running_ || !stage)
        throw std::invalid_argument("invalid projection stage registration");
    extensions_.push_back(std::move(stage));
}
void SpectralProjectionPipeline::setOutputSink(std::unique_ptr<IProjectionOutputSink> sink)
{
    if (running_ || !sink)
        throw std::invalid_argument("invalid projection output sink");
    sink_ = std::move(sink);
}
void SpectralProjectionPipeline::setVisualization(yk::visualization::LiveView* view)
{
    if (running_)
        throw std::invalid_argument("cannot replace a running visualization");
    visualization_ = view;
}
bool SpectralProjectionPipeline::run(const SimulationConfig& c,
    const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model,
    const std::filesystem::path& output_file, std::string& error,TaskProgressCallback progress)
{
    if (running_) {
        error = "projection pipeline is already running";
        return false;
    }
    YK::diagnostics::TimingSpan total_watch("projection.run",std::to_string(c.geometry.parameters.views)+" views");
    error.clear();
    auto builtins = (source_->contract().output_location==pipeline::Location::Device)?primary_detail::makeCudaProjectionStages(c):primary_detail::makeProjectionStages(c);
    std::vector<IProjectionStage*> stages;
    for (auto& stage : builtins)
        stages.push_back(stage.get());
    for (auto& stage : extensions_)
        stages.push_back(stage.get());
    std::stable_sort(stages.begin(), stages.end(), [](auto* a, auto* b) {
        return a->phase() < b->phase();
    });
    primary_detail::EnergyProjectionOutput energy_output;
    IProjectionOutputSink& output = sink_ ? *sink_ : energy_output;
    pipeline::RunScope run_scope(running_);
    auto cleanup=pipeline::onExit([&]{output.reset();pipeline::resetReverse(stages);source_->reset();});
    try {
#ifdef YK_SPECTRAL_VTK
        std::unique_ptr<yk::visualization::LiveView> owned_display;
        auto* display = c.visualization.enabled ? visualization_ : nullptr;
        if (c.visualization.enabled) {
            const auto& v = c.visualization;
            if (!display) {
                owned_display = std::make_unique<yk::visualization::LiveView>(yk::visualization::LiveViewOptions {
                    v.texture_max_size, v.max_fps, v.offscreen, v.hold_open, v.snapshot_directory.string() });
                display = owned_display.get();
            }
            const auto& g = c.geometry.parameters;
            yk::visualization::LabelVolume volume;
            volume.dimensions = { g.volume_x, g.volume_y, g.volume_z };
            volume.spacing = { g.voxel_x_mm, g.voxel_y_mm, g.voxel_z_mm };
            volume.labels = labels;
            constexpr double colors[][3] = { { .75, .85, 1 }, { 1, .65, .25 }, { .85, .35, .4 }, { .4, .85, .6 }, { .8, .6, .95 }, { .85, .85, .35 } };
            for (size_t i = 0; i < c.projection.materials.size(); ++i) {
                const auto& m = c.projection.materials[i];
                const auto& color = colors[i % 6];
                volume.materials.push_back({ m.label, m.name, { color[0], color[1], color[2], m.density_g_cm3 < .01 ? 0. : .15 } });
            }
            display->setVolume(volume);
        }
#else
        if (c.visualization.enabled)
            throw std::runtime_error("live display requires a build with BUILD_VTK_VISUALIZATION=ON");
#endif
        if (!output.acceptsDomain(source_->contract().output_domain))
            throw std::runtime_error("projection backend and output sink signal domains do not match");
        ProjectionStageContext context {c,*source_};
        auto domain=source_->contract().output_domain;
        for(auto* stage:stages){domain=pipeline::advanceDomain(domain,stage->contract(),stage->info().name.c_str());stage->validateContext(context);}
        if(!output.acceptsDomain(domain))throw std::runtime_error("projection final stage domain does not match output sink");
        for(auto* stage:stages){stage->reset();
            YK::diagnostics::TimingSpan watch("projection.stage.prepare",stage->info().name.c_str());stage->prepare(context);}
        {YK::diagnostics::TimingSpan watch("projection.source.prepare");source_->prepare({c,labels,model});}
        const auto& g = c.geometry.parameters;
        if (g.views <= 0)
            throw std::runtime_error("projection pipeline requires positive views");
#ifdef YK_SPECTRAL_VTK
        if (display) {
            std::vector<yk::visualization::FrameGeometry> frames;
            frames.reserve(g.views);
            for (int view = 0; view < g.views; ++view)
                frames.push_back(source_->geometryForDisplay(view));
            display->setTrajectory(frames);
        }
#endif
        const int batch_capacity=std::min(g.views,source_->batchCapacity());
        if(batch_capacity<1)throw std::runtime_error("invalid backend batch capacity");
        std::vector<ProjectionFrame> frames(batch_capacity);
        const ProjectionOutputContext output_context{c,output_file};
        output.prepare(output_context);
        using Clock = std::chrono::steady_clock;
        auto elapsed = [](Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };
        std::vector<double> stage_ms(stages.size(), 0.0);
        double primary_ms = 0.0, transfer_ms=0.0,normalize_ms=0.0,output_ms = 0.0, display_ms = 0.0;
        pipeline::Progress progress_report(progress,"Projection simulation",g.views);progress_report.start();
        const auto start = Clock::now();
        for(int first=0;first<g.views;first+=batch_capacity){
            const int count=std::min(batch_capacity,g.views-first);
            auto batch_tick=Clock::now();{YK::diagnostics::TimingSpan watch("projection.primary", "first="+std::to_string(first)+" count="+std::to_string(count));source_->generate(first,count,frames);source_->synchronize();}
            primary_ms+=elapsed(batch_tick);
            for(size_t i=0;i<stages.size();++i){auto* stage=stages[i];
                YK::diagnostics::TimingSpan stage_watch(std::string("projection.stage.")+stage->info().name.c_str(),std::string(stage->info().name.c_str())+" first="+std::to_string(first)+" count="+std::to_string(count));
                const auto timing=pipeline::execute(stage->info().name.c_str(),stage->contract(),frames.data(),frames.size(),
                    [&](pipeline::Location location){auto* storage=frames.data();const auto size=frames.size();source_->ensureLocation(frames,location);
                        if(storage!=frames.data() || size!=frames.size())throw std::runtime_error("backend changed stage batch storage during transfer");},
                    [&](ProjectionFrame& frame){stage->process(frame,context);stage->processAir(frame,context);frame.logarithm_valid=false;},
                    [&]{source_->synchronize();});
                transfer_ms+=timing.transfer_ms;stage_ms[i]+=timing.process_ms;
            }
            auto normalize_tick=Clock::now();source_->finalizeBatch(frames);source_->synchronize();normalize_ms+=elapsed(normalize_tick);
            auto transfer_tick=Clock::now();source_->ensureLocation(frames,ProjectionMemoryLocation::Host);transfer_ms+=elapsed(transfer_tick);
            for(int slot=0;slot<count;++slot){const int view=first+slot;auto& frame=frames[slot];auto tick=Clock::now();
            tick = Clock::now();
            {YK::diagnostics::TimingSpan watch("projection.output.write","view="+std::to_string(view));const pipeline::Contract<ProjectionSignalDomain> output_contract{frame.domain,frame.domain,output.inputLocation(),output.inputLocation()};
                pipeline::execute(output.info().name.c_str(),output_contract,&frame,1,
                    [&](pipeline::Location location){source_->ensureLocation(frames,location);},
                    [&](const ProjectionFrame& value){output.write(value,output_context);},[]{});}
            output_ms += elapsed(tick);
#ifdef YK_SPECTRAL_VTK
            if (display && display->isOpen()) {
                const auto display_start = Clock::now();
                if (frame.domain != ProjectionSignalDomain::EnergyKeV)
                    throw std::runtime_error("live display currently requires energy-integrating frames");
                const auto geometry = source_->geometryForDisplay(view);
                const std::vector<float>* values = &frame.signal_energy;
                std::vector<float> custom_logarithm;
                if (sink_) {
                    custom_logarithm.resize(frame.signal_energy.size());
                    for (size_t i = 0; i < custom_logarithm.size(); ++i)
                        custom_logarithm[i] = -std::log(std::max(frame.signal_energy[i] / std::max(frame.reference_energy[i], 1e-30f), 1e-30f));
                }
                // The default output already calculated this exact same-frame log.
                const auto& logarithm = sink_ ? custom_logarithm : energy_output.attenuationForDisplay();
                if (c.visualization.projection == "primary")
                    values = &frame.expected_primary_energy;
                if (c.visualization.projection == "log")
                    values = &logarithm;
                std::ostringstream description;
                description << "Live " << c.visualization.projection << " projection | detector offsets U/N/V "
                            << g.offset_u_mm << " / " << g.offset_n_mm << " / " << g.offset_v_mm << " mm\nsource offsets "
                            << g.source_offset_x_mm << " / " << g.source_offset_y_mm << " / " << g.source_offset_z_mm
                            << " mm | detector tilt U/V/N " << g.tilt_u_rad << " / " << g.tilt_v_rad << " / " << g.tilt_n_rad << " rad";
                description << "\nobject offset " << g.phantom_offset_x_mm << " / " << g.phantom_offset_y_mm << " / " << g.phantom_offset_z_mm
                            << " mm | rotation X/Y/Z " << g.phantom_rotation_x_rad << " / " << g.phantom_rotation_y_rad << " / " << g.phantom_rotation_z_rad << " rad";
                description << "\nmaterials:";
                for (const auto& m : c.projection.materials)
                    description << " " << int(m.label) << "=" << m.name;
                display->updateProjections(geometry, frame.width, frame.height, *values, frame.signal_energy, logarithm, description.str(), view + 1 == g.views);
                display_ms += elapsed(display_start);
            }
#endif
            progress_report.advance();
            if (view == 0 || view + 1 == g.views || (view + 1) % 10 == 0)
                std::cerr << "projection frame " << view + 1 << "/" << g.views << std::endl;
#ifdef YK_SPECTRAL_VTK
            if (display && (view == 0 || view + 1 == g.views || (view + 1) % 10 == 0)) {
                const auto display_start = Clock::now();
                display->setStatus("Computing " + std::to_string(view + 1) + " / " + std::to_string(g.views) + " views. Editing and browsing unlock after completion.");
                display_ms += elapsed(display_start);
            }
#endif
        }
        }
        {YK::diagnostics::TimingSpan watch("projection.output.finish");output.finish();}
        std::ostringstream timing;
        timing << std::fixed << std::setprecision(1)
               << "projection_pipeline_timing_ms total=" << elapsed(start)
               << " primary_compute=" << primary_ms <<" transfer_boundaries="<<transfer_ms<<" normalization="<<normalize_ms;
        for (size_t i = 0; i < stages.size(); ++i)
            timing << " " << stages[i]->info().name.c_str() << "=" << stage_ms[i];
        timing << " output=" << output_ms << " display_and_cache=" << display_ms;
        SimLogger::instance().info(timing.str());
#ifdef YK_SPECTRAL_VTK
        if (owned_display)
            owned_display->finish();
#endif
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
} // namespace yk::spectral
