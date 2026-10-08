#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"

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

        std::filesystem::remove(path);
        std::cout << "discrete fASKS model and CUDA convolution tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
