#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace yk::spectral;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

nlohmann::json makeModel() {
    constexpr int radial_count = 256;
    std::vector<double> radii(radial_count);
    for (int i = 0; i < radial_count; ++i)
        radii[static_cast<size_t>(i)] = 0.1 * i;

    std::vector<std::vector<double>> densities(3,
                                               std::vector<double>(radial_count));
    for (int i = 0; i < radial_count; ++i) {
        densities[0][static_cast<size_t>(i)] = 1.0;
        densities[1][static_cast<size_t>(i)] = 3.0;
        densities[2][static_cast<size_t>(i)] = 5.0;
    }

    return {
        {"format", "fasks-runtime-model/v4"},
        {"schema_version", 4},
        {"signal_domain", "energy_deposition_per_pixel"},
        {"validity", {{"transmission_min", std::exp(-1.7)},
                       {"transmission_max", std::exp(-0.1)}}},
        {"kernel_sampling", {{"representation", "discrete_radial_table"},
                              {"radial_sample_count", radial_count},
                              {"support_u_mm", 3.0},
                              {"support_v_mm", 3.0}}},
        {"calibration_signature", {{"sid_mm", 500.0}}},
        {"detector_response", {{"model", "air-pencil-beam-scatter-per-primary"},
                                {"scatter_kernel_per_primary", {{0.0}}}}},
        {"edge_amplitude_window", {{"model", "paper-linear-distance-to-boundary"},
                                    {"boundary_scale", 1.0},
                                    {"ramp_mm", 80.0}}},
        {"water_equivalent_thickness", {{"interpolation", "pchip"},
                                          {"log_attenuation_nodes", {0.0, 0.1, 0.5, 1.7}},
                                          {"thickness_mm_nodes", {0.0, 10.0, 50.0, 170.0}}}},
        {"fasks", {{"gamma_per_mm", 0.0}, {"wet_smooth_sigma_mm", 0.0}}},
        {"center_kernel", {
            {"amplitude", {{"interpolation", "linear-log-attenuation"},
                            {"log_attenuation_nodes", {0.1, 0.5, 1.7}},
                            {"values", {0.4, 0.8, 1.2}}}},
            {"shape", {{"family", "discrete-radial-kernel"},
                        {"radial_interpolation", "linear"},
                        {"attenuation_interpolation", "linear-log-attenuation"},
                        {"radial_positions_mm", radii},
                        {"log_attenuation_nodes", {0.1, 0.5, 1.7}},
                        {"radial_density_nodes_mm2", densities}}}}}
    };
}

} // namespace

int main() {
    try {
        const auto path = std::filesystem::current_path() /
                          "scatter-discrete-test-model.json";
        {
            std::ofstream output(path);
            output << makeModel().dump(2);
        }

        FasksRuntimeModel model(path);
        require(model.discreteKernelCount() == 3,
                "discrete node count mismatch");
        require(std::abs(model.discreteAttenuationNode(1) - 0.5) < 1e-12,
                "nonuniform attenuation node mismatch");

        const double du = 0.4;
        const double dv = 0.7;
        const auto lower_kernel = model.makeDiscreteKernel(std::exp(-0.1), du, dv, 3.0, 3.0);
        const auto upper_kernel = model.makeDiscreteKernel(std::exp(-0.5), du, dv, 3.0, 3.0);
        const auto middle_kernel = model.makeDiscreteKernel(std::exp(-0.3), du, dv, 3.0, 3.0);
        require(middle_kernel.size() == lower_kernel.size(),
                "discrete kernel size mismatch");
        for (size_t i = 0; i < middle_kernel.size(); ++i) {
            const double expected = 0.5 * (lower_kernel[i] + upper_kernel[i]);
            require(std::abs(middle_kernel[i] - expected) < 1e-6,
                    "radial kernel interpolation mismatch");
        }
        require(std::abs(lower_kernel.front() - 0.4 * 0.7) < 1e-6,
                "kernel mass was unexpectedly renormalized");

        constexpr int width = 31;
        constexpr int height = 23;
        const size_t pixels = static_cast<size_t>(width) * height;
        const int center = (height / 2) * width + width / 2;
        std::vector<float> air(pixels, 1.0f);
        std::vector<float> primary = air;
        std::vector<unsigned char> object_mask(pixels, 1);
        primary[static_cast<size_t>(center)] = 0.5f;

        CudaFasksScatter scatter(width, height, du, dv, model);
        scatter.apply(primary, air, object_mask);
        const std::vector<float> gamma_zero_result = primary;

        const double test_transmission = 0.5;
        const double source = test_transmission * model.amplitude(test_transmission);
        const auto expected_kernel = model.makeDiscreteKernel(
            test_transmission, du, dv, 3.0, 3.0);
        const int radius_u = static_cast<int>(std::ceil(3.0 / du));
        const int radius_v = static_cast<int>(std::ceil(3.0 / dv));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double expected = 0.0;
                const int dx = x - width / 2;
                const int dy = y - height / 2;
                if (std::abs(dx) <= radius_u && std::abs(dy) <= radius_v) {
                    expected = source * expected_kernel[
                        static_cast<size_t>(dy + radius_v) * (2 * radius_u + 1)
                        + dx + radius_u];
                }
                const double actual = static_cast<double>(primary[static_cast<size_t>(y) * width + x])
                                    - (x == width / 2 && y == height / 2 ? 0.5 : 1.0);
                require(std::abs(actual - expected) < 2e-5,
                        "CUDA discrete convolution mismatch");
            }
        }

        const double full_sum = source * [&] {
            double result = 0.0;
            for (float value : expected_kernel) result += value;
            return result;
        }();

        primary = air;
        object_mask.assign(pixels, 0);
        primary[0] = 0.5f;
        object_mask[0] = 1;
        scatter.apply(primary, air, object_mask);
        double edge_sum = 0.0;
        for (size_t i = 0; i < pixels; ++i)
            edge_sum += static_cast<double>(primary[i]) - (i == 0 ? 0.5 : 1.0);
        require(edge_sum < full_sum * 0.8,
                "cropped detector kernel was unexpectedly renormalized");

        primary = air;
        object_mask.assign(pixels, 0);
        scatter.apply(primary, air, object_mask);
        require(primary == air, "air scan must not add object scatter");

        // Multiple sources exercise spectral accumulation across several
        // attenuation intervals, including the thin and thick endpoints.
        primary = air;
        object_mask.assign(pixels, 1);
        const int sx[] = {1, 8, 15, 22, 29};
        const int sy[] = {1, 8, 12, 18, 21};
        const double attenuations[] = {0.02, 0.3, 0.7, 1.3, 2.1};
        std::vector<double> direct(pixels, 0.0);
        for (int ray = 0; ray < 5; ++ray) {
            const float t = static_cast<float>(std::exp(-attenuations[ray]));
            primary[static_cast<size_t>(sy[ray]) * width + sx[ray]] = t;
            const auto kernel = model.makeDiscreteKernel(t, du, dv, 3.0, 3.0);
            const double strength = t * model.amplitude(t);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const int dx = x - sx[ray], dy = y - sy[ray];
                    if (std::abs(dx) <= radius_u && std::abs(dy) <= radius_v)
                        direct[static_cast<size_t>(y) * width + x] += strength * kernel[
                            static_cast<size_t>(dy + radius_v) * (2 * radius_u + 1) + dx + radius_u];
                }
        }
        std::vector<float> multi_scatter;
        scatter.predict(primary, air, object_mask, multi_scatter);
        for (size_t i = 0; i < pixels; ++i)
            require(std::abs(multi_scatter[i] - direct[i]) < 2e-5,
                    "single-IFFT multi-source convolution mismatch");

        const auto gamma_path = std::filesystem::current_path() /
                                "scatter-discrete-gamma-test-model.json";
        auto gamma_json = makeModel();
        gamma_json["fasks"]["gamma_per_mm"] = 0.01;
        gamma_json["fasks"]["wet_smooth_sigma_mm"] = 0.8;
        {
            std::ofstream output(gamma_path);
            output << gamma_json.dump(2);
        }
        FasksRuntimeModel gamma_model(gamma_path);
        CudaFasksScatter gamma_scatter(width, height, du, dv, gamma_model);
        primary = air;
        object_mask.assign(pixels, 1);
        primary[static_cast<size_t>(center)] = 0.5f;
        gamma_scatter.apply(primary, air, object_mask);
        bool gamma_changed_result = false;
        for (size_t i = 0; i < pixels; ++i) {
            require(std::isfinite(primary[i]) && primary[i] >= 0.0f,
                    "CUDA gamma correction produced an invalid value");
            gamma_changed_result = gamma_changed_result ||
                std::abs(primary[i] - gamma_zero_result[i]) > 1e-7f;
        }
        require(gamma_changed_result,
                "CUDA gamma correction did not alter the scatter estimate");

        float *device_primary = nullptr, *device_air = nullptr, *device_scatter = nullptr;
        require(cudaMalloc(&device_primary, pixels * sizeof(float)) == cudaSuccess, "device primary allocation failed");
        require(cudaMalloc(&device_air, pixels * sizeof(float)) == cudaSuccess, "device air allocation failed");
        require(cudaMalloc(&device_scatter, pixels * sizeof(float)) == cudaSuccess, "device scatter allocation failed");
        auto device_input = air;
        device_input[static_cast<size_t>(center)] = 0.5f;
        require(cudaMemcpy(device_primary, device_input.data(), pixels * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess,
                "device primary upload failed");
        require(cudaMemcpy(device_air, air.data(), pixels * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess,
                "device air upload failed");
        std::vector<float> device_result(pixels);
        for (int mode = 0; mode < 2; ++mode) {
            auto& projector = mode == 0 ? scatter : gamma_scatter;
            projector.predictDevice(device_primary, device_air, object_mask, device_scatter);
            require(cudaMemcpy(device_result.data(), device_scatter, pixels * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
                    "device scatter download failed");
            const auto& reference = mode == 0 ? gamma_zero_result : primary;
            for (size_t i = 0; i < pixels; ++i)
                require(std::abs(device_result[i] - (reference[i] - device_input[i])) < 2e-5,
                        "device scatter output differs from host output");
        }
        cudaFree(device_primary); cudaFree(device_air); cudaFree(device_scatter);

        // Compare resident fixed iterations with the former host update,
        // including gamma, reduced bases, floor clipping and repeated frames.
        CudaFasksScatter segmented_scatter(width, height, du, dv, model, 3);
        for (auto* solver : {&scatter, &gamma_scatter, &segmented_scatter}) {
            for (int scenario = 0; scenario < 3; ++scenario) {
                const int iterations = scenario == 0 ? 1 : scenario == 1 ? 5 : 10;
                const double damping = scenario == 0 ? 1.0 : scenario == 1 ? 0.7 : 0.2;
                const double scale = scenario == 0 ? 0.0 : scenario == 1 ? 1.0 : 3.0;
                std::vector<float> measurement(pixels), flat(pixels);
                std::vector<unsigned char> mask(pixels);
                for (size_t p = 0; p < pixels; ++p) {
                    flat[p] = 1.0f + static_cast<float>(p % 7) * 0.1f;
                    measurement[p] = flat[p] * (p % 3 == 0 ? 0.001f : 0.5f);
                    mask[p] = scenario == 1 ? 0 : 1;
                }
                auto expected = measurement;
                std::vector<float> prediction;
                for (int it = 0; it < iterations; ++it) {
                    solver->predict(expected, flat, mask, prediction);
                    for (size_t p = 0; p < pixels; ++p) {
                        const float target = std::max(flat[p] * 1.0e-8f,
                            measurement[p] - static_cast<float>(scale) * std::max(0.0f, prediction[p]));
                        expected[p] = static_cast<float>((1.0-damping)*expected[p] + damping*target);
                    }
                }
                // Exercise input/output aliasing as well as workspace reuse.
                auto actual = measurement;
                solver->correctFixed(actual, flat, mask, iterations, damping, scale, actual);
                for (size_t p = 0; p < pixels; ++p) {
                    require(std::isfinite(actual[p]) && actual[p] >= flat[p]*1.0e-8f,
                            "fixed GPU correction violates positive floor");
                    require(std::abs(actual[p]-expected[p]) <= 2e-5f*flat[p],
                            "fixed GPU correction differs from host update");
                }
            }
        }
        bool invalid_rejected = false;
        std::vector<float> guide_input(pixels, 7.0f), guide_output;
        scatter.smoothGuide(guide_input, 2.0, 1.5, guide_output);
        for (float x : guide_output) require(std::abs(x-7.0f) < 2e-6f, "guide changes a flat field");
        guide_input[center] = 100;
        scatter.smoothGuide(guide_input, 2.0, 1.5, guide_output);
        require(guide_output[center] > 7 && guide_output[center] < 100,
                "guide does not smooth an impulse");
        require(guide_input[center] == 100, "guide changes original data");
        try {
            std::vector<float> result;
            scatter.correctFixed(air, air, object_mask, 0, 0.7, 1.0, result);
        } catch (const std::runtime_error&) { invalid_rejected = true; }
        require(invalid_rejected, "fixed correction accepted zero iterations");

        std::filesystem::remove(path);
        std::filesystem::remove(gamma_path);
        std::cout << "discrete fASKS model and CUDA convolution tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
