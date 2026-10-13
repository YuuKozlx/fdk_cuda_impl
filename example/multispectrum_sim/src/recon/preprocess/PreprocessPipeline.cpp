#include "image/ImageProcessing.hpp"
#include "pipeline/BuiltinPluginInfo.hpp"
#include "common/YkTimingWatchdog.hpp"
#include "recon/preprocess/PreprocessPipeline.hpp"
#include "CudaPixelOps.hpp"
#include "BeamHardeningFit.hpp"
#include "Stages.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "../../primary/AirReferenceReader.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace yk::spectral::preprocess {
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
class CorrectedProjectionOutput final:public pipeline::IOutput<Frame,Context,Domain> {
    std::ofstream file_;
public:
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::CorrectedOutput);}
    bool acceptsDomain(Domain domain)const override{return domain==Domain::LogAttenuation;}
    void prepare(const Context& context)override{
        if(!context.output_file.parent_path().empty())std::filesystem::create_directories(context.output_file.parent_path());
        file_.open(context.output_file,std::ios::binary);require(bool(file_),"cannot create preprocessed projection");
    }
    void write(const Frame& frame,const Context&)override{
        file_.write(reinterpret_cast<const char*>(frame.host_data->data()),std::streamsize(frame.pixels*sizeof(float)));
        require(bool(file_),"cannot write preprocessed frame");
    }
    void finish()override{file_.close();require(bool(file_),"cannot flush preprocessed projection");}
    void reset()noexcept override{if(file_.is_open())file_.close();file_.clear();}
};
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point tick) {
    return std::chrono::duration<double, std::milli>(Clock::now() - tick).count();
}
} // namespace
void Pipeline::append(std::unique_ptr<IStage> stage) {
    require(!running_, "cannot append a stage while running");
    require(bool(stage), "null preprocess stage");
    extensions_.push_back(std::move(stage));
}
bool Pipeline::run(const SimulationConfig &config, const std::filesystem::path &input_path,
                   const std::filesystem::path &output_path, std::string &diagnostic,
                   TaskProgressCallback progress) {
    if (running_) {
        diagnostic = "preprocess pipeline is already running";
        return false;
    }
    YK::diagnostics::TimingSpan total_watch("preprocess.run",std::to_string(config.geometry.parameters.views)+" views");
    pipeline::RunScope run_scope(running_);
    statistics_ = {};
    auto started = Clock::now();
    std::vector<std::unique_ptr<IStage>> builtins;
    std::vector<IStage *> stages;
    try {
        const auto &g = config.geometry.parameters;
        const auto &pp = config.reconstruction.preprocessing;
        require(pp.enabled, "reconstruction preprocessing is disabled");
        require(g.views > 0 && g.detector_u > 0 && g.detector_v > 0,
                "invalid preprocessing detector dimensions");
        size_t pixels = size_t(g.detector_u) * g.detector_v;
        require(pixels <= size_t(std::numeric_limits<int>::max()),
                "preprocessing frame is too large");
        require(std::filesystem::absolute(input_path).lexically_normal() !=
                    std::filesystem::absolute(output_path).lexically_normal(),
                "preprocessing output must not overwrite input");
        if (std::filesystem::exists(output_path))
            require(!std::filesystem::equivalent(input_path, output_path),
                    "preprocessing output aliases input");
        require(std::filesystem::file_size(input_path) ==
                    uint64_t(pixels) * g.views * sizeof(float),
                "preprocessing input projection size does not match geometry");
        auto tick = Clock::now();
        auto bhc = pp.beam_hardening;
        if (bhc.enabled && bhc.fit_from_water_plate)
            bhc = fitWaterPlate(config, bhc, progress);
        statistics_.fit_ms = elapsed(tick);
        tick = Clock::now();
        const auto pool_start=yk::spectral::cuda_resources::devicePoolStatistics();
        CudaPixelOps gpu(pixels);
        builtins = makeStages(config, bhc);
        for (auto &stage : builtins)
            stages.push_back(stage.get());
        for (auto &stage : extensions_)
            stages.push_back(stage.get());
        CorrectedProjectionOutput output;
        auto cleanup=pipeline::onExit([&]{output.reset();pipeline::resetReverse(stages);});
        image::Workspace image_workspace(Location::Device);
        Context context{config, gpu, progress,output_path,&image_workspace};
        Domain domain = Domain::LogAttenuation;
        for(auto* stage:stages){
            domain=pipeline::advanceDomain(domain,stage->contract(),stage->info().name.c_str());
            stage->validateContext(context);
            statistics_.stages_ms.emplace_back(stage->info().name.c_str(),0);
        }
        require(domain==Domain::LogAttenuation,"preprocess final output must be log attenuation");
        for(auto* stage:stages)stage->prepare(context);
        gpu.synchronize();
        statistics_.init_ms = elapsed(tick);
        std::unique_ptr<primary_detail::AirReferenceReader> air_reader;
        bool broadcast_air = false;
        if (pp.scatter.enabled) {
            air_reader = std::make_unique<primary_detail::AirReferenceReader>(
                pp.scatter.air_projection_file, pixels, g.views);
            broadcast_air = std::filesystem::file_size(pp.scatter.air_projection_file) ==
                            pixels * sizeof(float);
        }
        std::ifstream input(input_path, std::ios::binary);
        require(bool(input), "cannot open preprocessing input");
        output.prepare(context);
        std::vector<float> host(pixels), air;
        pipeline::Progress progress_report(progress,"Projection correction",g.views);
        progress_report.start();
        for (int view = 0; view < g.views; ++view) {
            tick = Clock::now();
            {YK::diagnostics::TimingSpan watch("preprocess.read","view="+std::to_string(view));input.read(reinterpret_cast<char *>(host.data()), pixels * sizeof(float));
            require(bool(input), "cannot read preprocessing frame");}
            {YK::diagnostics::TimingSpan watch("preprocess.validate","view="+std::to_string(view));for(float v:host)require(std::isfinite(v),"projection contains a non-finite value");}
            const bool new_air = air_reader && (!broadcast_air || view == 0);
            if (new_air)
                air_reader->read(view, air);
            statistics_.read_ms += elapsed(tick);
            Frame frame{view,
                        g.detector_u,
                        g.detector_v,
                        pixels,
                        Domain::LogAttenuation,
                        Location::Host,
                        &host,
                        gpu.data(),
                        air_reader ? gpu.air() : nullptr};
            // Clear validity per frame; upload air only when it changes.
            gpu.beginFrame();
            if (new_air) {
                tick = Clock::now();
                gpu.uploadTo(gpu.air(), air);
                ++statistics_.air_uploads;
                statistics_.transfer_ms += elapsed(tick);
            }
            auto ensure = [&](Location location) {
                if (frame.location == location)
                    return;
                tick = Clock::now();
                if (location == Location::Device) {
                    gpu.uploadTo(gpu.data(), host);
                    ++statistics_.image_uploads;
                } else {
                    gpu.download(host);
                    ++statistics_.image_downloads;
                }
                statistics_.transfer_ms += elapsed(tick);
                frame.location = location;
            };
            for (size_t index = 0; index < stages.size(); ++index) {
                auto *stage = stages[index];
                YK::diagnostics::TimingSpan stage_watch(std::string("preprocess.stage.")+stage->info().name.c_str(),std::string(stage->info().name.c_str())+" view="+std::to_string(view));
                const auto timing=pipeline::execute(stage->info().name.c_str(),stage->contract(),&frame,1,ensure,
                    [&](Frame& value){stage->process(value,context);
                        require(host.size()==pixels && value.host_data==&host && value.device_data==gpu.data(),"preprocess stage changed borrowed frame storage");},
                    [&]{gpu.synchronize();});
                statistics_.stages_ms[index].second+=timing.process_ms;
            }
            ensure(Location::Host);
            {YK::diagnostics::TimingSpan watch("preprocess.validate.output","view="+std::to_string(view));for(float v:host)require(std::isfinite(v),"preprocessing produced a non-finite value");}
            {YK::diagnostics::TimingSpan watch("preprocess.write","view="+std::to_string(view));
                const pipeline::Contract<Domain> contract{Domain::LogAttenuation,Domain::LogAttenuation};
                const auto timing=pipeline::execute(output.info().name.c_str(),contract,&frame,1,ensure,
                    [&](const Frame& value){output.write(value,context);},[]{});
                statistics_.write_ms+=timing.process_ms;
            }
            progress_report.advance();
        }
        output.finish();
        statistics_.total_ms = elapsed(started);
        std::cout << "preprocess_timing_ms total=" << statistics_.total_ms
                  << " init=" << statistics_.init_ms << " fit=" << statistics_.fit_ms
                  << " read=" << statistics_.read_ms << " transfer=" << statistics_.transfer_ms
                  << " write=" << statistics_.write_ms << " uploads=" << statistics_.image_uploads
                  << " downloads=" << statistics_.image_downloads
                  << " air_uploads=" << statistics_.air_uploads << '\n';
        const auto pool_end=yk::spectral::cuda_resources::devicePoolStatistics();
        std::cout << "preprocess_pool allocations=" << pool_end.allocations-pool_start.allocations
            << " reuses=" << pool_end.reuses-pool_start.reuses << " active_bytes="
            << pool_end.active_bytes << " cached_bytes=" << pool_end.cached_bytes << '\n';
        for (const auto &stage : statistics_.stages_ms)
            std::cout << "preprocess_stage name=" << stage.first << " elapsed_ms=" << stage.second
                      << '\n';
        diagnostic = "reconstruction preprocessing complete: " + output_path.string();
        return true;
    } catch (const std::exception &error) {
        diagnostic = error.what();
        return false;
    }
}
} // namespace yk::spectral::preprocess
