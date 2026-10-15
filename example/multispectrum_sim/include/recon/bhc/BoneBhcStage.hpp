#pragma once
#include "config/SimConfig.hpp"
#include "pipeline/Stage.hpp"
#include "recon/preprocess/calibration/TwoMaterialCalibration.hpp"
#include <filesystem>
#include <optional>
#include <vector>

namespace yk::spectral::bone_bhc {
// Whole-acquisition contract, deliberately separate from per-frame preprocessing.
enum class Domain { ScatterCorrectedLog,
    WaterCorrectedLog,
    ReferenceEnergyLog };
struct Dataset {
    std::filesystem::path original_projection, corrected_projection, reconstruction;
    Domain domain = Domain::ScatterCorrectedLog;
};
struct Options {
    int max_iterations = 1;
    double stop_ratio = .78;
};
struct Iteration {
    int number = 0;
    std::optional<double> model_error;
    double model_ms = 0, reconstruction_ms = 0;
};
struct Result {
    std::vector<Iteration> iterations;
    bool stopped_on_ratio = false;
};
struct Context {
    std::filesystem::path output_directory;
    Options options;
    // Reconstruction bypasses the enclosing pipeline. Algorithms always read the
    // fixed acquisition in their declared input domain, never a previous result.
    std::function<void(const std::filesystem::path&, const std::filesystem::path&)> reconstruct;
    // Optional initial water correction; subsequent model output is already mono.
    std::function<void(const std::filesystem::path&, const std::filesystem::path&)> initialize;
    std::function<void(const Iteration&)> progress;
};
using IStage = pipeline::IStage<Dataset, Context, Domain>;
class IBoneBhcAlgorithm {
public:
    virtual ~IBoneBhcAlgorithm() = default;
    virtual const char* id() const noexcept = 0;
    virtual Domain inputDomain() const noexcept = 0;
    virtual const char* errorMetric() const noexcept = 0;
    virtual bool supportsIteration() const noexcept { return false; }
    virtual void prepare(const Context&) { }
    virtual std::optional<double> correct(const std::filesystem::path& volume,
        const std::filesystem::path& fixed_projection, const std::filesystem::path& output)
        = 0;
    virtual void reset() noexcept { }
};
struct SegmentationOptions {
    double air_hu = -500, bone_hu = 300;
};
class PhysicalBoneBhc final : public IBoneBhcAlgorithm {
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    PhysicalBoneBhc(SimulationConfig, Domain, BeamHardeningConfig, SegmentationOptions = {});
    ~PhysicalBoneBhc() override;
    const char* id() const noexcept override { return "physical"; }
    bool supportsIteration() const noexcept override { return true; }
    Domain inputDomain() const noexcept override;
    const char* errorMetric() const noexcept override;
    std::optional<double> correct(const std::filesystem::path&, const std::filesystem::path&, const std::filesystem::path&) override;
};
class TwoMaterialLutBhc final : public IBoneBhcAlgorithm {
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    TwoMaterialLutBhc(SimulationConfig, preprocess::TwoMaterialCalibration, double bone_thickness_scale = 1, SegmentationOptions = {});
    ~TwoMaterialLutBhc() override;
    const char* id() const noexcept override { return "two_material_lut"; }
    bool supportsIteration() const noexcept override { return true; }
    Domain inputDomain() const noexcept override { return Domain::ScatterCorrectedLog; }
    const char* errorMetric() const noexcept override { return "none"; }
    std::optional<double> correct(const std::filesystem::path&, const std::filesystem::path&, const std::filesystem::path&) override;
};
class BoneBhcStage final : public IStage {
    std::unique_ptr<IBoneBhcAlgorithm> algorithm_;
    std::shared_ptr<Result> result_;

public:
    BoneBhcStage(std::unique_ptr<IBoneBhcAlgorithm>, std::shared_ptr<Result>);
    const pipeline::PluginInfo& info() const noexcept override;
    pipeline::Contract<Domain> contract() const override;
    void validateContext(const Context&) const override;
    void process(Dataset&, const Context&) override;
    void reset() noexcept override;
    const IBoneBhcAlgorithm& algorithm() const noexcept { return *algorithm_; }
};
inline const pipeline::PluginInfo& boneBhcInfo()
{
    static const pipeline::PluginInfo p = [] {
        pipeline::PluginInfo v;
        v.id = "preprocess.bone_bhc";
        v.name = "Bone beam hardening correction";
        v.description = "Whole-acquisition correction with an injected physical or two-material LUT algorithm; shared single/iterative execution.";
        v.version = "1.0";
        v.kind = pipeline::PluginKind::PreprocessStage;
        v.category = "beam_hardening";
        v.tags = { "dataset", "bone", "strategy" };
        v.implementation_status = pipeline::ImplementationStatus::Experimental;
        v.documentation = "include/recon/bhc/README.md";
        return v;
    }();
    return p;
}
// Optional native command adapter; input domain and strategy are explicit in the
// JSON job. Internal callers construct the algorithm and stage directly.
void runBoneBhcJob(const std::filesystem::path& job);
}
