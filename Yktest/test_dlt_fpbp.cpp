#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "DLTFpBp/YkDltAlgebraicReconstructor.hpp"
#include "DLTFpBp/YkDltSiddonProjector.hpp"
#include "global/YkMem3d.hpp"

namespace {

using YK::DltFpBp::SDltProjectionMatrix;
using YK::DltFpBp::SDltRayGeometry;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 add(const Vec3& a, const Vec3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(const Vec3& a, const Vec3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 scale(const Vec3& a, double value)
{
    return {value * a.x, value * a.y, value * a.z};
}

double dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b)
{
    return {a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

Vec3 normalized(const Vec3& value)
{
    const double length = std::sqrt(dot(value, value));
    return scale(value, 1.0 / length);
}

// Build a projective camera directly.  No physical detector origin is
// introduced: focal lengths and principal point are expressed in pixels.
SDltProjectionMatrix makeProjection(int view, int views, int channels, int rows)
{
    constexpr double pi = 3.14159265358979323846;
    const double angle = 2.0 * pi * view / views;
    const Vec3 source{
        82.0 * std::cos(angle) + 1.1,
        79.0 * std::sin(angle) - 0.7,
        2.3 * std::sin(2.0 * angle) + 0.4
    };
    const Vec3 target{2.4, -1.6, 0.8 + 0.5 * std::cos(3.0 * angle)};
    const Vec3 forward = normalized(subtract(target, source));
    Vec3 right = normalized(cross(forward, {0.03, -0.02, 1.0}));
    Vec3 up = normalized(cross(right, forward));

    // Per-view detector roll makes the matrices deliberately non-ideal.
    const double roll = (1.4 * pi / 180.0) * std::sin(1.7 * angle + 0.2);
    const Vec3 rolled_right = add(scale(right, std::cos(roll)),
        scale(up, std::sin(roll)));
    const Vec3 rolled_up = add(scale(right, -std::sin(roll)),
        scale(up, std::cos(roll)));
    right = rolled_right;
    up = rolled_up;

    const double focal_u = 74.0 + 1.8 * std::sin(angle + 0.4);
    const double focal_v = 71.0 + 1.2 * std::cos(2.0 * angle - 0.1);
    const double principal_u = 0.5 * (channels - 1.0) +
        0.9 * std::sin(2.0 * angle);
    const double principal_v = 0.5 * (rows - 1.0) -
        0.6 * std::cos(angle);
    const Vec3 rows_world[3] = {
        add(scale(right, focal_u), scale(forward, principal_u)),
        add(scale(up, focal_v), scale(forward, principal_v)),
        forward
    };

    SDltProjectionMatrix projection{};
    for (int row = 0; row < 3; ++row) {
        projection(row, 0) = rows_world[row].x;
        projection(row, 1) = rows_world[row].y;
        projection(row, 2) = rows_world[row].z;
        projection(row, 3) = -dot(rows_world[row], source);
    }
    return projection;
}

SDltProjectionMatrix scaledProjection(const SDltProjectionMatrix& input,
    double factor)
{
    SDltProjectionMatrix output = input;
    for (double& value : output.value) value *= factor;
    return output;
}

double maximumGeometryDifference(const SDltRayGeometry& a,
    const SDltRayGeometry& b)
{
    const float values_a[] = {
        a.source.x, a.source.y, a.source.z,
        a.ray00.x, a.ray00.y, a.ray00.z,
        a.rayU.x, a.rayU.y, a.rayU.z,
        a.rayV.x, a.rayV.y, a.rayV.z
    };
    const float values_b[] = {
        b.source.x, b.source.y, b.source.z,
        b.ray00.x, b.ray00.y, b.ray00.z,
        b.rayU.x, b.rayU.y, b.rayU.z,
        b.rayV.x, b.rayV.y, b.rayV.z
    };
    double maximum = 0.0;
    for (size_t i = 0; i < std::size(values_a); ++i)
        maximum = std::max(maximum,
            std::fabs(static_cast<double>(values_a[i]) - values_b[i]));
    return maximum;
}

double dotProduct(const std::vector<float>& a, const std::vector<float>& b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        result += static_cast<double>(a[i]) * b[i];
    return result;
}

} // namespace

int main_dlt_fpbp_adjoint()
{
    constexpr int channels = 24;
    constexpr int rows = 20;
    constexpr int views = 12;
    YK::SVolGeom volume = YK::SVolGeom::make_centered(
        18, 16, 14, 1.1f, 0.85f, 1.25f);
    volume.center = make_float3(2.4f, -1.6f, 0.8f);

    std::vector<SDltProjectionMatrix> projections;
    std::vector<SDltProjectionMatrix> rescaled_projections;
    projections.reserve(views);
    rescaled_projections.reserve(views);
    for (int view = 0; view < views; ++view) {
        projections.push_back(makeProjection(view, views, channels, rows));
        const double factor = (view & 1) ? -3.7 : 2.3;
        rescaled_projections.push_back(scaledProjection(projections.back(), factor));
    }

    std::vector<SDltRayGeometry> geometry;
    std::vector<SDltRayGeometry> rescaled_geometry;
    bool ok = YK::DltFpBp::buildRayGeometry(projections, channels, rows,
            volume.center, geometry) &&
        YK::DltFpBp::buildRayGeometry(rescaled_projections, channels, rows,
            volume.center, rescaled_geometry);
    double scale_invariance_error = 0.0;
    if (ok) {
        for (int view = 0; view < views; ++view)
            scale_invariance_error = std::max(scale_invariance_error,
                maximumGeometryDifference(geometry[view], rescaled_geometry[view]));
        ok = scale_invariance_error < 2e-5;
    }

    const size_t volume_count = static_cast<size_t>(volume.Nx) * volume.Ny * volume.Nz;
    const size_t projection_count = static_cast<size_t>(channels) * rows * views;
    std::mt19937 generator(0xD17F0B9u);
    std::uniform_real_distribution<float> distribution(0.1f, 1.0f);
    std::vector<float> x(volume_count), y(projection_count);
    for (float& value : x) value = distribution(generator);
    for (float& value : y) value = distribution(generator);
    std::vector<float> ax(projection_count, 0.f);
    std::vector<float> at_y(volume_count, 0.f);

    cudaStream_t stream = nullptr;
    if (ok) ok = cudaStreamCreate(&stream) == cudaSuccess;
    YK::Mem::MemoryController memory;
    auto d_x = memory.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, 0);
    auto d_at_y = memory.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, 0);
    auto d_y = memory.allocateDevice3D<float>(channels, rows, views, 0);
    auto d_ax = memory.allocateDevice3D<float>(channels, rows, views, 0);
    if (ok) {
        ok = cudaMemcpyAsync(d_x.data(), x.data(), volume_count * sizeof(float),
                cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(d_y.data(), y.data(), projection_count * sizeof(float),
                cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }

    YK::DltFpBp::DltSiddonProjector projector;
    if (ok) ok = projector.prepare(geometry, volume, channels, rows) &&
        projector.forward(d_x.data(), d_ax.data(), false, stream) &&
        projector.backproject(d_y.data(), d_at_y.data(), true, stream) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    if (ok) {
        ok = cudaMemcpy(ax.data(), d_ax.data(), projection_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(at_y.data(), d_at_y.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    const double lhs = ok ? dotProduct(ax, y) : NAN;
    const double rhs = ok ? dotProduct(x, at_y) : NAN;
    const double relative_error = std::fabs(lhs - rhs) /
        std::max({std::fabs(lhs), std::fabs(rhs), 1e-30});
    const double agreement_percent = 100.0 * (1.0 - relative_error);
    ok = ok && std::isfinite(relative_error) && relative_error < 1e-5;

    std::printf("[DLTFpBp matched Siddon] <Ax,y>=%.9e <x,A^Ty>=%.9e "
        "relative=%.3e agreement=%.9f%% scale_error=%.3e %s\n",
        lhs, rhs, relative_error, agreement_percent, scale_invariance_error,
        relative_error < 1e-5 ? "PASS" : "FAIL");
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_dlt_algebraic_methods()
{
    constexpr int channels = 24;
    constexpr int rows = 20;
    constexpr int views = 12;
    YK::SVolGeom volume = YK::SVolGeom::make_centered(18, 16, 14, 1.f);
    std::vector<SDltProjectionMatrix> matrices;
    for (int view = 0; view < views; ++view)
        matrices.push_back(makeProjection(view, views, channels, rows));
    std::vector<SDltRayGeometry> geometry;
    if (!YK::DltFpBp::buildRayGeometry(matrices, channels, rows,
            volume.center, geometry)) return 1;

    const size_t volume_count = static_cast<size_t>(volume.Nx) * volume.Ny * volume.Nz;
    const size_t projection_count = static_cast<size_t>(channels) * rows * views;
    std::vector<float> truth(volume_count, 0.f);
    for (size_t i = 0; i < truth.size(); ++i)
        truth[i] = (i % 17 == 0 || i % 29 == 0) ? 0.8f : 0.03f;
    std::vector<float> measured(projection_count);
    std::vector<float> predicted(projection_count);

    YK::Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, 0, false);
    auto d_volume = memory.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, 0, false);
    auto d_measured = memory.allocateDevice3D<float>(channels, rows, views, 0, false);
    auto d_predicted = memory.allocateDevice3D<float>(channels, rows, views, 0, false);
    if (cudaMemcpy(d_truth.data(), truth.data(), truth.size() * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    YK::DltFpBp::DltSiddonProjector projector;
    if (!projector.prepare(geometry, volume, channels, rows) ||
        !projector.forward(d_truth.data(), d_measured.data(), false, nullptr) ||
        cudaMemcpy(measured.data(), d_measured.data(), measured.size() * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    const double base_norm = dotProduct(measured, measured);

    struct Case {
        YK::DltFpBp::EDltAlgebraicMethod method;
        YK::DltFpBp::EDltAlgebraicWeightModel weight_model;
        int requested_subset_size;
        int expected_subset_size;
        int expected_subset_count;
        float relaxation;
        const char* name;
    };
    const Case cases[] = {
        {YK::DltFpBp::EDltAlgebraicMethod::Sirt,
            YK::DltFpBp::EDltAlgebraicWeightModel::ExactSubset,
            4, 12, 1, 1.f, "SIRT exact"},
        {YK::DltFpBp::EDltAlgebraicMethod::Sart,
            YK::DltFpBp::EDltAlgebraicWeightModel::ExactSubset,
            4, 1, 12, 0.6f, "SART exact"},
        {YK::DltFpBp::EDltAlgebraicMethod::Ossart,
            YK::DltFpBp::EDltAlgebraicWeightModel::ExactSubset,
            4, 4, 3, 0.6f, "OS-SART exact"},
        {YK::DltFpBp::EDltAlgebraicMethod::Sart,
            YK::DltFpBp::EDltAlgebraicWeightModel::TigreApprox,
            4, 1, 12, 0.2f, "SART TIGRE"}
    };
    bool ok = true;
    for (const auto& item : cases) {
        YK::DltFpBp::SDltAlgebraicConfig config{};
        config.method = item.method;
        config.weight_model = item.weight_model;
        config.ossart_subset_size = item.requested_subset_size;
        config.relaxation = item.relaxation;
        YK::DltFpBp::DltAlgebraicReconstructor reconstructor;
        ok = ok && reconstructor.prepare(geometry, volume, channels, rows, config) &&
            reconstructor.subsetSize() == item.expected_subset_size &&
            reconstructor.subsetCount() == item.expected_subset_count;
        if (!ok) break;
        cudaMemset(d_volume.data(), 0, volume_count * sizeof(float));
        ok = reconstructor.iterate(d_measured.data(), d_volume.data(), 2) &&
            projector.forward(d_volume.data(), d_predicted.data(), false, nullptr) &&
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(predicted.data(), d_predicted.data(),
                predicted.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        double residual = 0.0;
        for (size_t i = 0; i < measured.size(); ++i) {
            const double difference = measured[i] - predicted[i];
            residual += difference * difference;
        }
        const double relative = std::sqrt(residual / base_norm);
        std::printf("[DLT algebraic] %-15s subset=%d count=%d residual=%.6f %s\n",
            item.name, reconstructor.subsetSize(), reconstructor.subsetCount(),
            relative, relative < 1.0 ? "PASS" : "FAIL");
        ok = ok && std::isfinite(relative) && relative < 1.0;
    }

    std::vector<YK::SConeProjGeomVec> cone_geometry;
    ok = ok && YK::DltFpBp::buildConeProjectionGeometry(
        geometry, cone_geometry);
    YK::DltFpBp::SDltAlgebraicConfig input_config{};
    input_config.method = YK::DltFpBp::EDltAlgebraicMethod::Sirt;
    input_config.weight_model =
        YK::DltFpBp::EDltAlgebraicWeightModel::ExactSubset;
    std::vector<float> from_rays(volume_count);
    std::vector<float> from_matrices(volume_count);
    std::vector<float> from_vec(volume_count);
    const auto reconstruct_from = [&](const auto& input,
                                      std::vector<float>& result) {
        YK::DltFpBp::DltAlgebraicReconstructor reconstructor;
        if (!reconstructor.prepare(input, volume, channels, rows, input_config) ||
            cudaMemset(d_volume.data(), 0, volume_count * sizeof(float)) != cudaSuccess ||
            !reconstructor.iterate(d_measured.data(), d_volume.data(), 1) ||
            cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(result.data(), d_volume.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        return true;
    };
    ok = ok && reconstruct_from(geometry, from_rays) &&
        reconstruct_from(matrices, from_matrices) &&
        reconstruct_from(cone_geometry, from_vec);
    double reference_norm = 0.0;
    double matrix_difference = 0.0;
    double vec_difference = 0.0;
    if (ok) {
        for (size_t i = 0; i < volume_count; ++i) {
            reference_norm += static_cast<double>(from_rays[i]) * from_rays[i];
            const double dm = static_cast<double>(from_matrices[i]) - from_rays[i];
            const double dv = static_cast<double>(from_vec[i]) - from_rays[i];
            matrix_difference += dm * dm;
            vec_difference += dv * dv;
        }
    }
    const double matrix_relative = std::sqrt(matrix_difference /
        std::max(reference_norm, 1e-30));
    const double vec_relative = std::sqrt(vec_difference /
        std::max(reference_norm, 1e-30));
    const bool input_equivalence = ok && matrix_relative < 1e-5 &&
        vec_relative < 1e-5;
    std::printf("[DLT algebraic inputs] matrix/ray=%.3e vec/ray=%.3e %s\n",
        matrix_relative, vec_relative, input_equivalence ? "PASS" : "FAIL");
    ok = input_equivalence;
    return ok ? 0 : 1;
}
