#include "common/YkTimingWatchdog.hpp"
#include "recon/preprocess/PreprocessPipeline.hpp"
#include "CudaPixelOps.hpp"
#include "BeamHardeningFit.hpp"
#include "Stages.hpp"
#include "recon/preprocess/DeviceMemoryPool.hpp"
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
    running_ = true;
    statistics_ = {};
    auto started = Clock::now();
    std::vector<std::unique_ptr<IStage>> builtins;
    std::vector<IStage *> stages;
    // Always reset prepared stages, including on exceptions, before releasing CUDA storage.
    struct Reset {
        std::vector<IStage *> &stages;
        bool &running;
        ~Reset() {
            for (auto i = stages.rbegin(); i != stages.rend(); ++i)
                (*i)->reset();
            running = false;
        }
    };
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
        const auto pool_start=devicePoolStatistics();
        CudaPixelOps gpu(pixels);
        builtins = makeStages(config, bhc);
        for (auto &stage : builtins)
            stages.push_back(stage.get());
        for (auto &stage : extensions_)
            stages.push_back(stage.get());
        Reset reset{stages, running_};
        Context context{config, gpu, progress};
        Domain domain = Domain::LogAttenuation;
        for (auto *stage : stages) {
            require(stage->inputDomain() == domain,
                    std::string("preprocess domain mismatch before ") + stage->name());
            domain = stage->outputDomain();
            statistics_.stages_ms.emplace_back(stage->name(), 0);
            stage->prepare(context);
        }
        require(domain == Domain::LogAttenuation,
                "preprocess final output must be log attenuation");
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
        if (!output_path.parent_path().empty())
            std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path, std::ios::binary);
        require(bool(output), "cannot create preprocessed projection");
        std::vector<float> host(pixels), air;
        if (progress)
            progress({"Projection correction", 0, g.views});
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
                ensure(stage->inputLocation());
                tick = Clock::now();
                YK::diagnostics::TimingSpan stage_watch(std::string("preprocess.stage.")+stage->name(),std::string(stage->name())+" view="+std::to_string(view));
                stage->process(frame, context);
                gpu.synchronize();
                require(host.size() == pixels && frame.host_data == &host &&
                            frame.device_data == gpu.data(),
                        "preprocess stage changed borrowed frame storage");
                frame.domain = stage->outputDomain();
                frame.location = stage->outputLocation();
                statistics_.stages_ms[index].second += elapsed(tick);
            }
            ensure(Location::Host);
            {YK::diagnostics::TimingSpan watch("preprocess.validate.output","view="+std::to_string(view));for(float v:host)require(std::isfinite(v),"preprocessing produced a non-finite value");}
            tick = Clock::now();
            {YK::diagnostics::TimingSpan watch("preprocess.write","view="+std::to_string(view));output.write(reinterpret_cast<const char *>(host.data()), pixels * sizeof(float));
            require(bool(output), "cannot write preprocessed frame");}
            statistics_.write_ms += elapsed(tick);
            if (progress)
                progress({"Projection correction", view + 1, g.views});
        }
        output.close();
        require(bool(output), "cannot flush preprocessed projection");
        statistics_.total_ms = elapsed(started);
        std::cout << "preprocess_timing_ms total=" << statistics_.total_ms
                  << " init=" << statistics_.init_ms << " fit=" << statistics_.fit_ms
                  << " read=" << statistics_.read_ms << " transfer=" << statistics_.transfer_ms
                  << " write=" << statistics_.write_ms << " uploads=" << statistics_.image_uploads
                  << " downloads=" << statistics_.image_downloads
                  << " air_uploads=" << statistics_.air_uploads << '\n';
        const auto pool_end=devicePoolStatistics();
        std::cout << "preprocess_pool allocations=" << pool_end.allocations-pool_start.allocations
            << " reuses=" << pool_end.reuses-pool_start.reuses << " active_bytes="
            << pool_end.active_bytes << " cached_bytes=" << pool_end.cached_bytes << '\n';
        for (const auto &stage : statistics_.stages_ms)
            std::cout << "preprocess_stage name=" << stage.first << " elapsed_ms=" << stage.second
                      << '\n';
        diagnostic = "reconstruction preprocessing complete: " + output_path.string();
        return true;
    } catch (const std::exception &error) {
        running_ = false;
        diagnostic = error.what();
        return false;
    }
}
} // namespace yk::spectral::preprocess
