#include "SpectralProjectionPipeline.hpp"
#include "../src/primary/CudaProjectionBackend.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace yk::spectral;
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Attenuation final : IMassAttenuationProvider {
    bool query(const MaterialSpec&, const std::vector<double>& e,
        std::vector<double>& values, std::string&) const override {
        values.assign(e.size(), 1.0); return true;
    }
};
struct State { int backend_resets = 0, finished = 0, frames = 0; bool fail = false, vacuum = false; };
class Backend final : public IProjectionBackend {
public:
    Backend(State& state, ProjectionSignalDomain domain)
        : state_(state), domain_(domain) {}
    ProjectionSignalDomain outputDomain() const override { return domain_; }
    void prepare(const SimulationConfig&, const std::vector<std::uint8_t>&,
        const SpectralTransmissionModel&) override {}
    void project(int view, ProjectionFrame& f) override {
        f.view = view;
        if (domain_ == ProjectionSignalDomain::EnergyKeV) {
            f.expected_primary_energy.assign(1, state_.vacuum ? 10.f*(view+1) : 3.f);
            f.sampled_primary_energy = f.expected_primary_energy;
            f.air_energy.assign(1, 10.f*(view+1));
        } else {
            f.photon_counts.bin_edges_keV = {20.f, 40.f, 80.f};
            f.photon_counts.channel_counts = {3.f, 4.f};
            f.photon_counts.air_channel_counts = {10.f*(view+1), 20.f*(view+1)};
        }
    }
    void sampleScatterNoise(ProjectionFrame&) override {}
    void reset() noexcept override { ++state_.backend_resets; }
private:
    State& state_;
    ProjectionSignalDomain domain_;
};
class ArithmeticStage final : public IProjectionStage {
public:
    ArithmeticStage(ProjectionStagePhase phase, bool multiply, State& state)
        : phase_(phase), multiply_(multiply), state_(state) {}
    const char* name() const override { return multiply_ ? "multiply" : "add"; }
    ProjectionStagePhase phase() const override { return phase_; }
    ProjectionSignalDomain domain() const override { return ProjectionSignalDomain::EnergyKeV; }
    void process(ProjectionFrame& f, const ProjectionStageContext&) override {
        if (state_.fail) throw std::runtime_error("intentional stage failure");
        for (auto& value : f.signal_energy) value = multiply_ ? value * 2.f : value + 2.f;
    }
    void processAir(ProjectionFrame& f, const ProjectionStageContext&) override {
        for (auto& value : f.reference_energy) value = multiply_ ? value * 2.f : value + 2.f;
    }
private:
    ProjectionStagePhase phase_;
    bool multiply_;
    State& state_;
};
class CountSink final : public IProjectionOutputSink {
public:
    explicit CountSink(State& state) : state_(state) {}
    ProjectionSignalDomain domain() const override { return ProjectionSignalDomain::PhotonCounts; }
    void prepare(const SimulationConfig&, const std::filesystem::path&) override {}
    void write(const ProjectionFrame& f) override {
        require(f.domain == ProjectionSignalDomain::PhotonCounts, "count signal converted to energy");
        require(f.photon_counts.bin_edges_keV == std::vector<float>({20,40,80}), "count thresholds changed");
        require(f.photon_counts.channel_counts == std::vector<float>({3,4}), "channel counts changed");
        require(f.photon_counts.reference_channel_counts ==
            std::vector<float>({10.f*(f.view+1),20.f*(f.view+1)}), "current-view count reference missing");
        ++state_.frames;
    }
    void finish() override { ++state_.finished; }
    void reset() noexcept override {}
private:
    State& state_;
};
float readFirst(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    float value = 0.f; input.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(input), "missing test energy output"); return value;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "test output directory required");
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        Attenuation attenuation;
        SpectralTransmissionModel model({}, {}, attenuation, {});
        SimulationConfig c;
        c.geometry.parameters.detector_u = 1;
        c.geometry.parameters.detector_v = 1;
        c.geometry.parameters.views = 2;
        c.projection.quantum_noise.enabled = false;
        c.projection.energy_output_file = directory / "energy.raw";
        c.projection.air_output_file = directory / "air.raw";
        State state;
        SpectralProjectionPipeline energy(std::make_unique<Backend>(state, ProjectionSignalDomain::EnergyKeV));
        // Register in reverse order. Phase ordering must yield (3 * 2) + 2.
        energy.addStage(std::make_unique<ArithmeticStage>(ProjectionStagePhase::Afterglow, false, state));
        energy.addStage(std::make_unique<ArithmeticStage>(ProjectionStagePhase::OpticalCrosstalk, true, state));
        std::string error;
        require(energy.run(c, {}, model, directory / "attenuation.raw", error), error.c_str());
        require(readFirst(c.projection.energy_output_file) == 8.f, "incorrect extension order");
        require(readFirst(c.projection.air_output_file) == 22.f, "incorrect air stage order");
        {
            std::ifstream air(c.projection.air_output_file, std::ios::binary);
            float values[2]{}; air.read(reinterpret_cast<char*>(values),sizeof(values));
            require(bool(air) && values[1] == 42.f, "second-view air not refreshed");
        }
        state.fail = true;
        require(!energy.run(c, {}, model, directory / "failed.raw", error), "stage failure was ignored");
        require(error == "intentional stage failure", "stage diagnostic lost");
        state.fail = false;
        require(energy.run(c, {}, model, directory / "recovered.raw", error), error.c_str());
        require(readFirst(c.projection.energy_output_file) == 8.f, "rerun state not reset");
        require(state.backend_resets == 3, "backend cleanup omitted");

        State vacuum; vacuum.vacuum = true;
        SpectralProjectionPipeline airRun(std::make_unique<Backend>(vacuum, ProjectionSignalDomain::EnergyKeV));
        c.projection.detector_postprocess.afterglow.enabled = true;
        c.projection.detector_postprocess.afterglow.p = .2;
        c.projection.detector_postprocess.afterglow.weights = {1};
        c.projection.detector_postprocess.afterglow.time_constants_ms = {2};
        const auto vacuumPath = directory / "vacuum.raw";
        require(airRun.run(c, {}, model, vacuumPath, error), error.c_str());
        {
            std::ifstream input(vacuumPath,std::ios::binary);
            float values[2]{}; input.read(reinterpret_cast<char*>(values),sizeof(values));
            require(bool(input) && std::abs(values[0]) < 1e-6 && std::abs(values[1]) < 1e-6,
                "changing-view vacuum must normalize to zero with independent afterglow states");
        }
        c.projection.detector_postprocess.afterglow.enabled = false;

        // Actual CUDA vacuum projection on a moving-source trajectory. Each
        // frame has a different inverse-square air fluence and must yield zero log.
        SimulationConfig gpuConfig;
        gpuConfig.geometry.kind = GeometryKind::PlanarSourceOrbit;
        auto& g = gpuConfig.geometry.parameters;
        g.detector_u=16;g.detector_v=16;g.views=3;g.views_per_turn=3;
        g.volume_x=g.volume_y=g.volume_z=8;
        g.sid_mm=200;g.sdd_mm=300;g.planar_detector_y_mm=100;
        // A wider detector makes the air-fluence difference between views
        // comfortably measurable while this remains a tiny vacuum fixture.
        g.source_orbit_radius_mm=15;g.pixel_u_mm=g.pixel_v_mm=1.0;
        gpuConfig.projection.quantum_noise.enabled=false;
        gpuConfig.projection.sampling.samples_per_pixel=4;
        gpuConfig.projection.geometry_flux.enabled=true;
        gpuConfig.projection.air_output_file=directory/"gpu-air.raw";
        gpuConfig.projection.energy_output_file.clear();
        MaterialSpec material;material.label=1;material.name="test";material.density_g_cm3=1;
        gpuConfig.projection.materials={material};
        SpectralTransmissionModel gpuModel({material}, {{40,100,1},{80,50,1}}, attenuation, {});
        const auto gpuPath=directory/"gpu-vacuum.raw";
        SpectralProjectionPipeline gpu(std::make_unique<primary_detail::CudaProjectionBackend>());
        require(gpu.run(gpuConfig,std::vector<std::uint8_t>(512,0),gpuModel,gpuPath,error),error.c_str());
        const size_t pixels=256;
        std::vector<float> logs(pixels*3),airFrames(pixels*3);
        std::ifstream logsFile(gpuPath,std::ios::binary);
        logsFile.read(reinterpret_cast<char*>(logs.data()),logs.size()*sizeof(float));
        std::ifstream airFile(gpuConfig.projection.air_output_file,std::ios::binary);
        airFile.read(reinterpret_cast<char*>(airFrames.data()),airFrames.size()*sizeof(float));
        require(bool(logsFile)&&bool(airFile),"GPU per-view output incomplete");
        for(float value:logs)require(std::abs(value)<1e-6,"GPU vacuum nonzero attenuation");
        require(std::abs(airFrames[0]-airFrames[pixels])>1.f,"GPU fixture did not vary air fluence");

        State counts;
        SpectralProjectionPipeline counting(
            std::make_unique<Backend>(counts, ProjectionSignalDomain::PhotonCounts));
        require(!counting.run(c, {}, model, directory / "wrong-sink.raw", error),
            "count backend accepted an energy sink");
        counting.setOutputSink(std::make_unique<CountSink>(counts));
        require(counting.run(c, {}, model, {}, error), error.c_str());
        require(counts.frames == 2 && counts.finished == 1, "counting sink lifecycle failed");
        counting.addStage(std::make_unique<ArithmeticStage>(ProjectionStagePhase::Readout, false, counts));
        require(!counting.run(c, {}, model, {}, error), "energy stage accepted photon counts");
        std::cout << "projection pipeline extension/order/recovery/photon-count contracts: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
