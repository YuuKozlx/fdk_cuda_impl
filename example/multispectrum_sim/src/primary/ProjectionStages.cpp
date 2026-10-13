#include "pipeline/BuiltinPluginInfo.hpp"
#include "ProjectionStages.hpp"
#include "DetectorSignal.hpp"
#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"
#include <random>

namespace yk::spectral::primary_detail {
namespace {
class EnergyStage : public IProjectionStage {
public:
    pipeline::Contract<ProjectionSignalDomain> contract()const override{return {ProjectionSignalDomain::EnergyKeV,ProjectionSignalDomain::EnergyKeV};}
};
class ScatterStage final : public EnergyStage {
public:
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::CpuScatter);}
    ProjectionStagePhase phase() const override { return ProjectionStagePhase::Scatter; }
    void prepare(const ProjectionStageContext& context) override {
        const auto& c=context.config;
        const auto& g = c.geometry.parameters;
        model_ = std::make_unique<FasksRuntimeModel>(c.projection.scatter.model_file);
        scatter_ = std::make_unique<CudaFasksScatter>(g.detector_u, g.detector_v,
            g.pixel_u_mm, g.pixel_v_mm, *model_,
            c.projection.scatter.kernel.mode == "segmented"
                ? c.projection.scatter.kernel.segments : 0);
    }
    void process(ProjectionFrame& f, const ProjectionStageContext&) override {
        scatter_->apply(f.signal_energy, f.air_energy, f.object_mask);
    }
    void reset() noexcept override { scatter_.reset(); model_.reset(); }
private:
    std::unique_ptr<FasksRuntimeModel> model_;
    std::unique_ptr<CudaFasksScatter> scatter_;
};
class QuantumStage final : public EnergyStage {
public:
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::CpuQuantum);}
    ProjectionStagePhase phase() const override { return ProjectionStagePhase::QuantumNoise; }
    void process(ProjectionFrame& f, const ProjectionStageContext& c) override {
        if (c.config.projection.scatter.enabled) c.source.sampleScatterNoise(f);
        else f.signal_energy = f.sampled_primary_energy;
    }
};
class CrosstalkStage final : public EnergyStage {
public:
    CrosstalkStage(CrosstalkConfig config, ProjectionStagePhase phase) : config_(std::move(config)), phase_(phase) {}
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(phase_==ProjectionStagePhase::OpticalCrosstalk?pipeline::BuiltinPlugin::CpuOptical:pipeline::BuiltinPlugin::CpuElectronic);}
    ProjectionStagePhase phase() const override { return phase_; }
    void process(ProjectionFrame& f, const ProjectionStageContext&) override {
        applyCrosstalkSignal(f.signal_energy, f.width, f.height, config_);
    }
    void processAir(ProjectionFrame& f, const ProjectionStageContext&) override {
        applyCrosstalkSignal(f.reference_energy, f.width, f.height, config_);
    }
private:
    CrosstalkConfig config_;
    ProjectionStagePhase phase_;
};
class AfterglowStage final : public EnergyStage {
public:
    explicit AfterglowStage(DetectorPostprocessConfig::AfterglowConfig config)
        : config_(std::move(config)) {}
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::CpuAfterglow);}
    ProjectionStagePhase phase() const override { return ProjectionStagePhase::Afterglow; }
    void process(ProjectionFrame& f, const ProjectionStageContext&) override {
        applyAfterglowSignal(f.signal_energy, config_, object_state_);
    }
    void processAir(ProjectionFrame& f, const ProjectionStageContext&) override {
        applyAfterglowSignal(f.reference_energy, config_, air_state_);
    }
    void reset() noexcept override { object_state_.clear(); air_state_.clear(); }
private:
    DetectorPostprocessConfig::AfterglowConfig config_;
    std::vector<std::vector<float>> object_state_, air_state_;
};
class DasStage final : public EnergyStage {
public:
    explicit DasStage(DetectorPostprocessConfig::DasConfig config)
        : config_(std::move(config)), random_(config_.noise_seed) {}
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::CpuDas);}
    ProjectionStagePhase phase() const override { return ProjectionStagePhase::Readout; }
    void process(ProjectionFrame& f, const ProjectionStageContext&) override {
        applyDasSignal(f.signal_energy, config_, random_);
    }
    void reset() noexcept override { random_.seed(config_.noise_seed); }
private:
    DetectorPostprocessConfig::DasConfig config_;
    std::mt19937 random_;
};
}
std::vector<std::unique_ptr<IProjectionStage>> makeProjectionStages(const SimulationConfig& c) {
    std::vector<std::unique_ptr<IProjectionStage>> stages;
    const auto& p = c.projection;
    if (p.scatter.enabled) stages.push_back(std::make_unique<ScatterStage>());
    if (p.quantum_noise.enabled) stages.push_back(std::make_unique<QuantumStage>());
    const auto& d = p.detector_postprocess;
    if (d.optical_crosstalk.enabled)
        stages.push_back(std::make_unique<CrosstalkStage>(d.optical_crosstalk,
            ProjectionStagePhase::OpticalCrosstalk));
    if (d.afterglow.enabled) stages.push_back(std::make_unique<AfterglowStage>(d.afterglow));
    if (d.electronic_crosstalk.enabled)
        stages.push_back(std::make_unique<CrosstalkStage>(d.electronic_crosstalk,
            ProjectionStagePhase::ElectronicCrosstalk));
    if (d.das.enabled) stages.push_back(std::make_unique<DasStage>(d.das));
    return stages;
}
} // namespace yk::spectral::primary_detail
