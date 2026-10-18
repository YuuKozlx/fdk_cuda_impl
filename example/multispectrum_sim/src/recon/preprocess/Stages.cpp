#include "processing/PluginCatalog.hpp"
#include "processing/support/CudaTiming.hpp"
#include "Stages.hpp"
#include "CudaPixelOps.hpp"
#include "processing/scatter/CudaFasksScatter.hpp"
#include "processing/scatter/FasksRuntimeModel.hpp"
#include "recon/ImageStages.hpp"
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
#include "processing/scatter/AdaptiveScatterSolver.hpp"
#endif
#include <stdexcept>
#include <iostream>
namespace yk::spectral::preprocess {
namespace {
class TransmissionConversion final : public IProjectionPreprocessStage {
    bool forward_;
    bool energy_;
  public:
    TransmissionConversion(bool forward,bool energy):forward_(forward),energy_(energy){}
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(forward_?
        pipeline::BuiltinPlugin::TransmissionRestore:pipeline::BuiltinPlugin::TransmissionOutput);}
    pipeline::Contract<Domain> contract()const override{return {
        forward_?(energy_?Domain::EnergySignal:Domain::LogAttenuation):Domain::Transmission,
        forward_?Domain::Transmission:(energy_?Domain::EnergySignal:Domain::LogAttenuation),Location::Device,Location::Device};}
    void process(Frame&,const Context& c)override{c.cuda.convertTransmission(forward_,energy_);}
};
class Restore final : public IProjectionPreprocessStage {
  public:
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::EnergyRestore);}
    pipeline::Contract<Domain> contract()const override{return {Domain::LogAttenuation,Domain::EnergySignal,Location::Device,Location::Device};}
    void prepare(const Context &) override {}
    void process(Frame &, const Context &c) override { c.cuda.restoreEnergy(); }
    void reset() noexcept override {}
};
class Scatter final : public IProjectionPreprocessStage {
    std::unique_ptr<FasksRuntimeModel> model_;
    std::unique_ptr<CudaFasksScatter> scatter_;
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
    std::unique_ptr<experimental::AdaptiveScatterSolver> adaptive_;
#endif
  public:
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::ScatterCorrection);}
    pipeline::Contract<Domain> contract()const override{return {Domain::EnergySignal,Domain::EnergySignal,Location::Device,Location::Device};}
    void prepare(const Context &c) override {
        const auto &g = c.config.geometry.parameters;
        const auto &s = c.config.reconstruction.preprocessing.scatter;
        model_ = std::make_unique<FasksRuntimeModel>(s.model_file);
        scatter_ = std::make_unique<CudaFasksScatter>(
            g.detector_u, g.detector_v, g.pixel_u_mm, g.pixel_v_mm, *model_,
            s.kernel.mode == "segmented" ? s.kernel.segments : 0);
        scatter_->prepareCorrectionWorkspace(s.guide.enabled);
        if(s.solver.type=="fixed")scatter_->prepareDeviceMask();
        if (s.solver.type == "adaptive_experimental") {
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
            adaptive_ = std::make_unique<experimental::AdaptiveScatterSolver>(size_t(g.detector_u) *
                                                                              g.detector_v);
#else
            throw std::runtime_error("adaptive scatter plugin is not built; enable "
                                     "MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER");
#endif
        } else if (s.solver.type != "fixed")
            throw std::runtime_error("unknown scatter solver");
    }
    void process(Frame &f, const Context &c) override {
        const auto &s = c.config.reconstruction.preprocessing.scatter;
        auto &gpu = c.cuda;
        float *measured = gpu.data();
        const float *air = gpu.air();
        if (s.guide.enabled) {
            yk::spectral::diagnostics::TimingSpan guide_watch("preprocess.guide","view="+std::to_string(f.view));
            scatter_->smoothGuideDevice(gpu.original(), s.guide.sigma_u_pixels,
                                        s.guide.sigma_v_pixels, gpu.guide());
            scatter_->smoothGuideDevice(gpu.air(), s.guide.sigma_u_pixels, s.guide.sigma_v_pixels,
                                        gpu.guideAir());
            if(yk::spectral::diagnostics::timingEnabled())gpu.synchronize();
            measured = gpu.guide();
            air = gpu.guideAir();
        }
        // The current edge-window model consumes a CPU mask, not frame signals.
        std::vector<unsigned char> mask;const unsigned char* device_mask=nullptr;
        const bool device_edge=s.solver.type=="fixed"&&!(yk::spectral::diagnostics::timingEnabled()&&std::getenv("YK_TIMING_CPU_EDGE_REFERENCE"));
        {yk::spectral::diagnostics::TimingSpan watch("preprocess.mask","view="+std::to_string(f.view));
            if(device_edge)device_mask=gpu.makeMaskDevice(measured,air,s.correction.object_mask_threshold);
            else mask=gpu.mask(measured,air,s.correction.object_mask_threshold);
        }
        if (s.solver.type == "adaptive_experimental") {
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
            const auto report = adaptive_->solveDevice(
                measured, air, measured, s.solver.adaptive, s.correction.scatter_scale,
                [&](const float *p, const float *a, float *out) {
                    scatter_->predictDevice(p, a, mask, out);
                },
                &mask);
            for (const auto &step : report.history)
                std::cout << "scatter_iteration frame=" << f.view + 1
                          << " iteration=" << step.iteration << " damping=" << step.damping
                          << " residual=" << step.relative_residual
                          << " change=" << step.relative_change << '\n';
            std::cout << "adaptive scatter frame=" << f.view + 1 << " status=" << report.status
                      << " predictions=" << report.predictions
                      << " low_transmission=" << report.low_transmission << '\n';
#endif
        } else if(device_mask)
            scatter_->correctFixedDeviceMask(measured,air,device_mask,s.correction.iterations,s.correction.damping,s.correction.scatter_scale,measured);
        else scatter_->correctFixedDevice(measured,air,mask,s.correction.iterations,s.correction.damping,s.correction.scatter_scale,measured);
        if (s.guide.enabled) {
            yk::spectral::diagnostics::TimingSpan watch("preprocess.guide.residual","view="+std::to_string(f.view));
            if(device_mask)scatter_->predictDeviceMask(measured,air,device_mask,gpu.prediction());
            else scatter_->predictDevice(measured,air,mask,gpu.prediction());
            gpu.guidedResidual(s.correction.scatter_scale);
        }
    }
    void reset() noexcept override {
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
        adaptive_.reset();
#endif
        scatter_.reset();
        model_.reset();
    }
};
class Finalize final : public IProjectionPreprocessStage {
    bool energy_;
    BeamHardeningConfig bhc_;

  public:
    Finalize(bool energy, BeamHardeningConfig bhc) : energy_(energy), bhc_(std::move(bhc)) {}
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(bhc_.enabled?pipeline::BuiltinPlugin::Bhc:pipeline::BuiltinPlugin::LogConversion);}
    pipeline::Contract<Domain> contract()const override {
        return {energy_ ? Domain::EnergySignal : Domain::LogAttenuation,
            Domain::LogAttenuation,Location::Device,Location::Device};
    }
    void prepare(const Context &c) override {
        if (bhc_.enabled)
            c.cuda.configureBhc(bhc_);
    }
    void process(Frame &, const Context &c) override { c.cuda.finalize(energy_, bhc_.enabled); }
    void reset() noexcept override {}
};
} // namespace
std::vector<std::unique_ptr<IProjectionPreprocessStage>> makeStages(const SimulationConfig &c,
                                                const BeamHardeningConfig &bhc) {
    std::vector<std::unique_ptr<IProjectionPreprocessStage>> stages;
    const bool scatter = c.reconstruction.preprocessing.scatter.enabled;
    if (scatter) {
        stages.push_back(std::make_unique<Restore>());
        stages.push_back(std::make_unique<Scatter>());
    }
    const auto& filters=c.reconstruction.preprocessing.filters;
    if(!filters.empty()) {
        stages.push_back(std::make_unique<TransmissionConversion>(true,scatter));
        for(const auto& f:filters){
            image::Options o;o.neighborhood=image::Neighborhood::Slice2D;
            if(f.operation=="mean")o.operation=image::Operation::Mean;
            else if(f.operation=="gaussian")o.operation=image::Operation::Gaussian;
            else if(f.operation=="guided")o.operation=image::Operation::Guided;
            else throw std::invalid_argument("Unknown projection filter operation");
            o.radius={f.radius_pixels,f.radius_pixels,0};
            o.sigma_mm={f.sigma_pixels*c.geometry.parameters.pixel_u_mm,
                f.sigma_pixels*c.geometry.parameters.pixel_v_mm,1};
            o.epsilon=f.epsilon;
            stages.push_back(image::makeProjectionStage(o,Domain::Transmission));
        }
        stages.push_back(std::make_unique<TransmissionConversion>(false,scatter));
    }
    if (scatter || bhc.enabled)
        stages.push_back(std::make_unique<Finalize>(scatter, bhc));
    return stages;
}
} // namespace yk::spectral::preprocess
