#include "recon/preprocess/CudaPixelOps.hpp"
#include "recon/preprocess/calibration/EmpiricalBhcFit.hpp"
#include "recon/preprocess/correction/EmpiricalBhcStage.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
using namespace yk::spectral;
void require(bool x, const char* message)
{
    if (!x)
        throw std::runtime_error(message);
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
    require(caught, "Expected rejection");
}
class Provider final : public preprocess::IBoneThicknessProvider {
public:
    std::vector<float> paths { 0, 10, 50, 100 };
    void writeDevice(int, int, int, float* out) override { check(cudaMemcpy(out, paths.data(), paths.size() * 4, cudaMemcpyHostToDevice)); }
};
int main()
{
    try {
        std::vector<preprocess::EmpiricalBhcDifference> samples;
        for (int i = 0; i < 500; ++i) {
            double a = std::sin(i * .17), b = std::cos(i * .31);
            samples.push_back({ -.7 * a + .2 * b, a, b });
        }
        auto fit = preprocess::fitEmpiricalBhc(samples);
        require(std::abs(fit.square - .7) < 1e-4 && std::abs(fit.cross + .2) < 1e-4, "Failed independent synthetic coefficient recovery");
        for (int i = 0; i < 10; ++i)
            samples[i].baseline += 1000;
        fit = preprocess::fitEmpiricalBhc(samples, .1);
        require(std::abs(fit.square - .7) < .02 && std::abs(fit.cross + .2) < .02, "Outliers dominate robust fitting");
        reject([] { preprocess::fitEmpiricalBhc(std::vector<preprocess::EmpiricalBhcDifference>(30, { 1, 1, 1 })); });
        preprocess::EmpiricalBhcParameters p { .7, -.2, 100 };
        reject([] { preprocess::EmpiricalBhcParameters { 0, 0, 0 }.validate(); });
        SimulationConfig config;
        auto& g = config.geometry.parameters;
        g.detector_u = 4;
        g.detector_v = 1;
        g.views = 1;
        preprocess::CudaPixelOps ops(1);
        TaskProgressCallback progress;
        std::filesystem::path output;
        preprocess::Context context { config, ops, progress, output };
        auto provider = std::make_shared<Provider>();
        auto stage = preprocess::makeEmpiricalBhcStage(p, provider);
        pipeline::validatePluginInfo(stage->info());
        float* device = nullptr;
        check(cudaMalloc(&device, 16));
        const float input[4] = { -.1f, 1, 3, 5 };
        float result[4];
        for (int repeat = 0; repeat < 2; ++repeat) {
            stage->prepare(context);
            check(cudaMemcpy(device, input, 16, cudaMemcpyHostToDevice));
            preprocess::Frame frame { 0, 4, 1, 4, preprocess::Domain::LogAttenuation, preprocess::Location::Device, nullptr, device, nullptr };
            stage->process(frame, context);
            check(cudaMemcpy(result, device, 16, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 4; ++i) {
                double q = provider->paths[i] / 100.;
                require(std::abs(result[i] - (input[i] + .7 * q * q - .2 * input[i] * q)) < 1e-6, "CUDA correction mismatch");
            }
            frame.view = 1;
            reject([&] { stage->process(frame, context); });
            stage->reset();
        }
        provider->paths[0] = -1;
        stage->prepare(context);
        check(cudaMemcpy(device, input, 16, cudaMemcpyHostToDevice));
        preprocess::Frame frame { 0, 4, 1, 4, preprocess::Domain::LogAttenuation, preprocess::Location::Device, nullptr, device, nullptr };
        stage->process(frame, context);
        check(cudaMemcpy(result, device, 16, cudaMemcpyDeviceToHost));
        require(!std::isfinite(result[0]), "Invalid path silently accepted");
        check(cudaFree(device));
        std::cout << "EBHC fit, robustness, degeneracy, CUDA parity and lifecycle passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
