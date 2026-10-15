#include "recon/preprocess/CudaPixelOps.hpp"
#include "recon/preprocess/calibration/EmpiricalTwoMaterialFit.hpp"
#include "recon/preprocess/correction/TwoMaterialBhcStage.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral;
void require(bool b, const char* s)
{
    if (!b)
        throw std::runtime_error(s);
}
void check(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
template <class F>
void reject(F f)
{
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "invalid input accepted");
}
class Provider : public preprocess::IBoneThicknessProvider {
public:
    std::vector<float> values;
    void writeDevice(int, int, int, float* p) override { check(cudaMemcpy(p, values.data(), values.size() * 4, cudaMemcpyHostToDevice)); }
};
double poly(double w, double b) { return -std::log(.4 * std::exp(-.03 * w - .08 * b) + .6 * std::exp(-.017 * w - .035 * b)); }
int main(int argc, char** argv)
{
    try {
        require(argc == 2, "test output path required");
        std::vector<preprocess::ThicknessMeasurement> measurements;
        for (int x = 0; x <= 200; x += 10)
            for (int y = 0; y <= 60; y += 3)
                measurements.push_back({ double(x), double(y), poly(x, y) });
        const auto fitted = preprocess::fitEmpiricalTwoMaterial(measurements);
        require(fitted.rmse < 1e-5, "empirical fit failed analytic two-exponential response");
        auto tangent = fitted.table(200, 60, "analytic empirical test");
        tangent.save(argv[1]);
        tangent = preprocess::TwoMaterialCalibration::load(argv[1]);
        require(tangent.reference_energy_keV == 0, "empirical target incorrectly assigned an energy");
        require(std::abs(tangent.target_water_mu_mm - .0222) < 1e-5, "wrong zero-thickness slope");
        require(std::abs(tangent.correct(float(poly(73.2, 17.4)), 17.4f) - (73.2 * .0222 + 17.4 * .053)) < 1e-4, "held-out tangent correction wrong");
        reject([&] { preprocess::fitEmpiricalTwoMaterial(std::vector<preprocess::ThicknessMeasurement>(30, { 1, 1, 1 })); });
        preprocess::TwoMaterialSamples s;
        s.bone_rows = 81;
        s.water_columns = 201;
        s.bone_step_mm = .25;
        s.water_step_mm = 1;
        s.reference_energy_keV = 70;
        s.target_water_mu_mm = .02;
        s.target_bone_mu_mm = .045;
        s.provenance = "independent two-line analytic test";
        for (int b = 0; b < s.bone_rows; ++b)
            for (int w = 0; w < s.water_columns; ++w)
                s.log_values.push_back(poly(w, b * .25));
        auto m = preprocess::calibrateTwoMaterial(s);
        m.save(argv[1]);
        m = preprocess::TwoMaterialCalibration::load(argv[1]);
        double peak = 0;
        for (double b = .13; b < 20; b += .83)
            for (double w = .37; w < 200; w += 3.71)
                peak = std::max(peak, std::abs(m.correct(float(poly(w, b)), float(b)) - (w * .02 + b * .045)));
        require(peak < 4e-5, "independent held-out inversion inaccurate");
        require(std::abs(m.correct(0, 0)) < 1e-7, "air offset");
        auto bad = s;
        bad.log_values[2] = bad.log_values[1];
        reject([&] { preprocess::calibrateTwoMaterial(bad); });
        reject([&] { m.correct(1, 21); });
        reject([&] { m.correct(NAN, 1); });
        SimulationConfig config;
        auto& g = config.geometry.parameters;
        g.detector_u = 5;
        g.detector_v = 1;
        g.views = 1;
        config.reconstruction.preprocessing.beam_hardening.enabled = false;
        preprocess::CudaPixelOps ops(1);
        TaskProgressCallback progress;
        std::filesystem::path output = argv[1];
        preprocess::Context context { config, ops, progress, output };
        auto provider = std::make_shared<Provider>();
        provider->values = { 0, 0, 3.13f, 19.9f, 0 };
        auto stage = preprocess::makeTwoMaterialBhcStage(m, provider);
        pipeline::validatePluginInfo(stage->info());
        config.reconstruction.preprocessing.beam_hardening.enabled = true;
        reject([&] { stage->validateContext(context); });
        config.reconstruction.preprocessing.beam_hardening.enabled = false;
        std::vector<float> input = { 0, float(poly(80, 0)), float(poly(130.4, 3.13)), float(poly(180, 19.9)), -.01f }, out(5);
        float* device = nullptr;
        check(cudaMalloc(&device, 20));
        for (int repeat = 0; repeat < 2; ++repeat) {
            stage->prepare(context);
            check(cudaMemcpy(device, input.data(), 20, cudaMemcpyHostToDevice));
            preprocess::Frame f { 0, 5, 1, 5, preprocess::Domain::LogAttenuation, preprocess::Location::Device, nullptr, device, nullptr };
            pipeline::execute(stage->info().name.c_str(), stage->contract(), &f, 1, [](auto) { throw std::runtime_error("unexpected transfer"); }, [&](auto& frame) { stage->process(frame, context); }, [] { check(cudaDeviceSynchronize()); });
            check(cudaMemcpy(out.data(), device, 20, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < 5; ++i)
                require(std::abs(out[i] - m.correct(input[i], provider->values[i])) < 2e-6, "CPU/CUDA LUT mismatch");
            stage->reset();
        }
        provider->values[0] = 30;
        stage->prepare(context);
        check(cudaMemcpy(device, input.data(), 20, cudaMemcpyHostToDevice));
        preprocess::Frame f { 0, 5, 1, 5, preprocess::Domain::LogAttenuation, preprocess::Location::Device, nullptr, device, nullptr };
        stage->process(f, context);
        check(cudaMemcpy(out.data(), device, 20, cudaMemcpyDeviceToHost));
        require(!std::isfinite(out[0]), "out-of-range bone not signaled");
        check(cudaFree(device));
        std::cout << "analytic holdout max=" << peak << "; CPU/CUDA, lifecycle, invalid models, double BHC guard passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
