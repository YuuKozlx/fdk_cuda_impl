#include "AdaptiveScatterSolver.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace yk::spectral;
using namespace yk::spectral::experimental;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        constexpr size_t count = 513; // Exercise partial reduction blocks.
        AdaptiveScatterSolver solver(count);
        std::vector<float> air(count, 100), measured(count, 72), primary = measured;
        ScatterCorrectionConfig::SolverConfig::AdaptiveConfig options;
        options.iterations = 60;
        options.initial_damping = 1;
        options.min_damping = 0.1;
        // P + 2 P = I: damping=1 overshoots; backtracking must recover P=24.
        auto predict = [](const auto& p, auto& s) {
            s.resize(p.size());
            for (size_t i = 0; i < p.size(); ++i) s[i] = 2 * p[i];
        };
        auto report = solver.solve(measured, air, primary, options, 1, predict);
        require(std::string(report.status) == "completed", "linear solver did not complete its iterations");
        require(report.damping <= options.initial_damping, "configured damping ceiling was exceeded");
        require(report.positivity_limited_steps > 0, "positivity step bound was not exercised");
        require(report.iterations == options.iterations, "solver stopped before configured iteration count");
        require(report.history.size() == options.iterations + 1, "iteration history is incomplete");
        for (float p : primary) require(std::abs(p - 24) < 1e-3, "wrong linear fixed point");
        const auto host_result = primary;
        primary = measured;
        report = solver.solve(measured, air, primary, options, 1,
            [](const auto&, auto&) { throw std::runtime_error("host predictor used during device solve"); },
            nullptr, [&](const float* p, const float*, float* s) {
                std::vector<float> values(count), result;
                require(cudaMemcpy(values.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
                        "cannot read device test signal");
                predict(values, result);
                require(cudaMemcpy(s, result.data(), count * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess,
                        "cannot write device test prediction");
            });
        require(report.iterations == options.iterations, "device solver stopped before configured count");
        for (size_t i = 0; i < count; ++i)
            require(std::abs(primary[i] - host_result[i]) < 1e-5, "host/device iteration results differ");

        primary = measured;
        report = solver.solve(measured, air, primary, options, 0, predict);
        require(report.iterations == options.iterations && primary == measured,
                "zero-scale solve did not preserve signal and exact iteration count");

        primary = measured;
        options.iterations = 1;
        options.initial_damping = 0.6;
        report = solver.solve(measured, air, primary, options, 1,
            [](const auto& p, auto& s) { s.assign(p.size(), 10); });
        require(std::string(report.status) == "completed" && report.iterations == 1, "iteration count not reported");
        require(std::abs(primary[0] - 66) < 1e-4, "one-step damping is incorrect");

        // An infeasible clipped fixed point must report stagnation, not convergence.
        options.iterations = 5;
        options.damping_decay = 0.8;
        primary = measured;
        report = solver.solve(measured, air, primary, options, 1,
            [](const auto& p, auto& s) { s.assign(p.size(), 10); });
        require(report.iterations == 5, "decaying steps stopped early");
        for (const auto& step : report.history)
            if (step.iteration > 0)
                require(step.damping <= options.initial_damping*std::pow(.8,step.iteration-1)+1e-12,
                        "decaying damping ceiling was exceeded");
        options.damping_decay = 1.0;

        primary = measured;
        options.iterations = 60;
        report = solver.solve(measured, air, primary, options, 1,
            [](const auto& p, auto& s) { s.assign(p.size(), 1000); });
        require(report.relative_residual > 1, "infeasible residual was hidden");
        require(primary[0] > 0 && std::isfinite(primary[0]), "positivity floor failed");
        primary = measured;
        options.initial_damping = 0.6;
        options.min_damping = 0.6;
        report = solver.solve(measured, air, primary, options, 1,
            [](const auto& p, auto& s) {
                s.resize(p.size());
                for (size_t i = 0; i < p.size(); ++i) s[i] = p[i] < 70 ? 1000 : 10;
            });
        require(std::string(report.status) == "stalled" && report.iterations == 0,
                "rejected minimum-damping step was accepted");
        require(report.rejected_steps > 0, "residual rejection was not exercised");
        require(primary == measured, "rejected step changed the last accepted signal");
        options = {};
        measured.assign(count, 50); measured[0] = 1;
        primary = measured;
        auto no_scatter = [](const auto& p, auto& s) { s.assign(p.size(), 0); };
        report = solver.solve(measured, air, primary, options, 1, no_scatter);
        require(!report.low_transmission_risk && std::abs(report.low_transmission - 0.5) < 1e-12,
                "isolated dark pixel triggered transmission risk");
        std::vector<unsigned char> mask(count, 0);
        mask[0] = 1;
        report = solver.solve(measured, air, primary, options, 1, no_scatter, &mask);
        require(report.low_transmission_risk && std::abs(report.low_transmission - 0.01) < 1e-12,
                "object-only transmission risk was not detected");
        struct DeviceFrames {
            float *m=nullptr,*a=nullptr,*p=nullptr;
            ~DeviceFrames(){cudaFree(m);cudaFree(a);cudaFree(p);}
        } device;
        require(cudaMalloc(&device.m,count*sizeof(float))==cudaSuccess,"allocate device measurement");
        require(cudaMalloc(&device.a,count*sizeof(float))==cudaSuccess,"allocate device air");
        require(cudaMalloc(&device.p,count*sizeof(float))==cudaSuccess,"allocate device primary");
        require(cudaMemcpy(device.m,measured.data(),count*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess,"upload device measurement");
        require(cudaMemcpy(device.a,air.data(),count*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess,"upload device air");
        require(cudaMemcpy(device.p,measured.data(),count*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess,"upload device primary");
        const auto device_report=solver.solveDevice(device.m,device.a,device.p,options,1,
            [&](const float*,const float*,float* prediction){require(cudaMemset(prediction,0,count*sizeof(float))==cudaSuccess,"zero device prediction");},&mask);
        require(std::abs(device_report.low_transmission-report.low_transmission)<1e-12&&device_report.low_transmission_risk,"device quantile differs from object-only CPU quantile");
        std::vector<float> device_result(count);
        require(cudaMemcpy(device_result.data(),device.p,count*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess,"download device primary");
        require(device_result==measured,"device zero scatter changed measurement");
        for (size_t p = 0; p < count; ++p) { measured[p] *= 1000; air[p] *= 1000; }
        primary = measured;
        const auto dose_report = solver.solve(measured, air, primary, options, 1, no_scatter, &mask);
        require(dose_report.low_transmission_risk &&
                std::abs(dose_report.low_transmission - report.low_transmission) < 1e-12,
                "transmission precheck depends on dose");
        std::cout << "adaptive scatter CUDA solver tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
