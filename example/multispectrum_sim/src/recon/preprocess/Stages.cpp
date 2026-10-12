#include "common/YkCudaTiming.hpp"
#include "Stages.hpp"
#include "CudaPixelOps.hpp"
#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
#include "AdaptiveScatterSolver.hpp"
#endif
#include <stdexcept>
#include <iostream>
namespace yk::spectral::preprocess {
namespace {
class Restore final : public IStage {
  public:
    const char *name() const override { return "Energy restore"; }
    Domain inputDomain() const override { return Domain::LogAttenuation; }
    Domain outputDomain() const override { return Domain::EnergySignal; }
    void prepare(const Context &) override {}
    void process(Frame &, const Context &c) override { c.cuda.restoreEnergy(); }
    void reset() noexcept override {}
};
class Scatter final : public IStage {
    std::unique_ptr<FasksRuntimeModel> model_;
    std::unique_ptr<CudaFasksScatter> scatter_;
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
    std::unique_ptr<experimental::AdaptiveScatterSolver> adaptive_;
#endif
  public:
    const char *name() const override { return "Scatter correction"; }
    Domain inputDomain() const override { return Domain::EnergySignal; }
    Domain outputDomain() const override { return Domain::EnergySignal; }
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
            YK::diagnostics::TimingSpan guide_watch("preprocess.guide","view="+std::to_string(f.view));
            scatter_->smoothGuideDevice(gpu.original(), s.guide.sigma_u_pixels,
                                        s.guide.sigma_v_pixels, gpu.guide());
            scatter_->smoothGuideDevice(gpu.air(), s.guide.sigma_u_pixels, s.guide.sigma_v_pixels,
                                        gpu.guideAir());
            if(YK::diagnostics::timingEnabled())gpu.synchronize();
            measured = gpu.guide();
            air = gpu.guideAir();
        }
        // The current edge-window model consumes a CPU mask, not frame signals.
        std::vector<unsigned char> mask;const unsigned char* device_mask=nullptr;
        const bool device_edge=s.solver.type=="fixed"&&!(YK::diagnostics::timingEnabled()&&std::getenv("YK_TIMING_CPU_EDGE_REFERENCE"));
        {YK::diagnostics::TimingSpan watch("preprocess.mask","view="+std::to_string(f.view));
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
            YK::diagnostics::TimingSpan watch("preprocess.guide.residual","view="+std::to_string(f.view));
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
class Finalize final : public IStage {
    bool energy_;
    BeamHardeningConfig bhc_;

  public:
    Finalize(bool energy, BeamHardeningConfig bhc) : energy_(energy), bhc_(std::move(bhc)) {}
    const char *name() const override {
        return bhc_.enabled ? "BHC / log conversion" : "Log conversion";
    }
    Domain inputDomain() const override {
        return energy_ ? Domain::EnergySignal : Domain::LogAttenuation;
    }
    Domain outputDomain() const override { return Domain::LogAttenuation; }
    void prepare(const Context &c) override {
        if (bhc_.enabled)
            c.cuda.configureBhc(bhc_);
    }
    void process(Frame &, const Context &c) override { c.cuda.finalize(energy_, bhc_.enabled); }
    void reset() noexcept override {}
};
} // namespace
std::vector<std::unique_ptr<IStage>> makeStages(const SimulationConfig &c,
                                                const BeamHardeningConfig &bhc) {
    std::vector<std::unique_ptr<IStage>> stages;
    const bool scatter = c.reconstruction.preprocessing.scatter.enabled;
    if (scatter) {
        stages.push_back(std::make_unique<Restore>());
        stages.push_back(std::make_unique<Scatter>());
    }
    if (scatter || bhc.enabled)
        stages.push_back(std::make_unique<Finalize>(scatter, bhc));
    return stages;
}
} // namespace yk::spectral::preprocess
