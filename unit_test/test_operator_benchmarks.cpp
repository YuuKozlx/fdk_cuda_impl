#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "CylFpBp/BP/YkCylBackOperator.hpp"
#include "CylFpBp/BP/YkCylJosephV3Strategy.hpp"
#include "CylFpBp/FP/YkCylForwardOperator.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

constexpr float kTwoPi = 6.28318530717958647692f;

struct BenchmarkShape {
    int nx, ny, nz;
    int channels, rows, views;
    float voxel_mm;
    float pixel_mm;
    float sid_mm;
    float sdd_mm;
    float cyl_radius_mm;
    const char* name;
};

BenchmarkShape benchmarkShape()
{
    const char* scale = std::getenv("YKCBCT_OPERATOR_BENCHMARK_SCALE");
    if (scale && std::string(scale) == "clinical") {
        return {512, 512, 512, 1024, 1024, 72, 0.35f, 0.417f,
            440.f, 770.f, 770.f, "clinical"};
    }
    return {96, 96, 64, 192, 96, 180, 1.f, 0.8f,
        440.f, 770.f, 650.f, "representative"};
}

bool includeSlowReferenceOperators()
{
    const char* enabled = std::getenv("YKCBCT_INCLUDE_SLOW_REFERENCE");
    return enabled && std::string(enabled) == "1";
}

double flatCoveragePercent(const BenchmarkShape& shape)
{
    const double object_width = shape.nx * shape.voxel_mm;
    const double projected_width = object_width * shape.sdd_mm / shape.sid_mm;
    return 100.0 * projected_width / (shape.channels * shape.pixel_mm);
}

double cylCoveragePercent(const BenchmarkShape& shape)
{
    const double half_width = 0.5 * shape.nx * shape.voxel_mm;
    const double projected_arc = 2.0 * shape.cyl_radius_mm *
        std::atan(half_width / shape.sid_mm);
    return 100.0 * projected_arc / (shape.channels * shape.pixel_mm);
}

double gibibytes(size_t elements)
{
    return static_cast<double>(elements * sizeof(float)) /
        (1024.0 * 1024.0 * 1024.0);
}

bool cudaAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

double dot(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    double value = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i)
        value += static_cast<double>(lhs[i]) * rhs[i];
    return value;
}

double relativeAdjointError(double lhs, double rhs)
{
    return std::fabs(lhs - rhs) /
        std::max({std::fabs(lhs), std::fabs(rhs), 1e-30});
}

void printTableBorder()
{
    std::cout << "+----------+----------------+------------+------------+--------------+----------+\n";
}

void printTableHeader()
{
    printTableBorder();
    std::cout << "| FP       | BP             | FP avg ms  | BP avg ms  | adjoint err  | relation |\n";
    printTableBorder();
}

void printTableRow(const char* fp, const char* bp, float fp_ms, float bp_ms,
    double error, const char* relation)
{
    std::cout << "| " << std::left << std::setw(8) << fp
              << " | " << std::setw(14) << bp
              << " | " << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << fp_ms
              << " | " << std::setw(10) << bp_ms
              << " | " << std::scientific << std::setprecision(3)
              << std::setw(12) << error
              << " | " << std::left << std::setw(8) << relation << " |\n";
}

float timeGpu(cudaStream_t stream, const std::function<bool()>& launch)
{
    const int repetitions = benchmarkShape().name == std::string("clinical")
        ? 3 : 5;
    if (!launch() || cudaStreamSynchronize(stream) != cudaSuccess) return NAN;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess ||
        cudaEventCreate(&end) != cudaSuccess) return NAN;
    cudaEventRecord(begin, stream);
    bool launched = true;
    for (int repetition = 0; repetition < repetitions; ++repetition)
        launched = launch() && launched;
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float elapsed = NAN;
    if (launched) {
        cudaEventElapsedTime(&elapsed, begin, end);
        elapsed /= repetitions;
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed;
}

SReconstructionParams representativeFlatParams(const BenchmarkShape& shape)
{
    SReconstructionParams p{};
    p.scan.Nu = shape.channels;
    p.scan.Nv = shape.rows;
    p.scan.NAng = shape.views;
    p.scan.totalViews = shape.views;
    p.scan.du_mm = shape.pixel_mm;
    p.scan.dv_mm = shape.pixel_mm;
    p.scan.sid_mm = shape.sid_mm;
    p.scan.sdd_mm = shape.sdd_mm;
    p.scan.range_rad = kTwoPi;
    p.volume.Nx = shape.nx;
    p.volume.Ny = shape.ny;
    p.volume.Nz = shape.nz;
    p.volume.voxelX_mm = shape.voxel_mm;
    p.volume.voxelY_mm = shape.voxel_mm;
    p.volume.voxelZ_mm = shape.voxel_mm;
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = kTwoPi * static_cast<float>(i) / p.scan.NAng;
    return p;
}

std::vector<float> randomVector(size_t count, unsigned int seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(generator);
    return values;
}

TEST(RepresentativeFlatOperators, CrossMatrixAdjointAndTiming)
{
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device is unavailable";
    const auto shape = benchmarkShape();
    const double exact_siddon_tolerance =
        std::string(shape.name) == "clinical" ? 5e-4 : 1e-4;
    const auto p = representativeFlatParams(shape);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) *
        p.scan.Nu * p.scan.Nv;
    const auto x = randomVector(volume_count, 12345u);
    const auto y = randomVector(projection_count, 54321u);

    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    GeometryContext context;
    ASSERT_TRUE(context.initialize(p, geometry));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ResourceContext resources;
    resources.attach(stream, 0);
    Mem::MemoryController memory;
    auto d_x = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    auto d_y = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng, 0);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng, 0);
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    ASSERT_EQ(cudaMemcpyAsync(d_x.data(), x.data(), volume_count * sizeof(float),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_y.data(), y.data(), projection_count * sizeof(float),
        cudaMemcpyHostToDevice, stream), cudaSuccess);

    struct ForwardResult {
        const char* name;
        ETask task;
        double inner_product = 0.0;
        float milliseconds = NAN;
    };
    ForwardResult forwards[] = {
        {"Joseph", ETask::FP_Joseph},
        {"Siddon", ETask::FP_Siddon},
    };
    for (auto& item : forwards) {
        auto op = makeForwardOperator(item.task);
        ASSERT_TRUE(op && op->prepare(context, resources));
        item.milliseconds = timeGpu(stream, [&] {
            return op->apply(d_x.data(), p, d_projection.data(), resources);
        });
        std::vector<float> ax(projection_count);
        ASSERT_EQ(cudaMemcpy(ax.data(), d_projection.data(),
            projection_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        item.inner_product = dot(ax, y);
        op->release();
        ASSERT_TRUE(std::isfinite(item.inner_product));
        ASSERT_TRUE(std::isfinite(item.milliseconds));
    }

    struct BackResult {
        const char* name;
        ETask task;
        double inner_product = 0.0;
        float milliseconds = NAN;
    };
    BackResult backs[] = {
        {"SiddonRay", ETask::BP_Siddon_RayDriven},
        {"SiddonVoxel", ETask::BP_Siddon_VoxDriven},
        {"SiddonVoxelV2", ETask::BP_Siddon_VoxDriven_v2},
        {"SiddonVoxelV3", ETask::BP_Siddon_VoxDriven_v3},
        {"Joseph", ETask::BP_Joseph},
        {"JosephV2", ETask::BP_Joseph_v2},
        {"JosephV3", ETask::BP_Joseph_v3},
        {"JosephGather", ETask::BP_Joseph_gather},
        {"FDK", ETask::BP_FDK},
        {"FDKMatched", ETask::BP_FDK_matched},
    };
    for (auto& item : backs) {
        auto op = makeBackOperator(item.task);
        ASSERT_TRUE(op && op->prepare(context, resources));
        item.milliseconds = timeGpu(stream, [&] {
            return op->apply(d_y.data(), p, d_volume.data(), true, resources);
        });
        std::vector<float> aty(volume_count);
        ASSERT_EQ(cudaMemcpy(aty.data(), d_volume.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        item.inner_product = dot(x, aty);
        op->release();
        ASSERT_TRUE(std::isfinite(item.inner_product));
        ASSERT_TRUE(std::isfinite(item.milliseconds));
    }

    std::cout << std::defaultfloat << "\nFlat " << shape.name << " matrix: volume="
              << shape.nx << 'x' << shape.ny << 'x' << shape.nz
              << " detector=" << shape.channels << 'x' << shape.rows
              << " views=" << shape.views << " voxel=" << shape.voxel_mm
              << " mm pixel=" << shape.pixel_mm << " mm SID/SDD="
              << shape.sid_mm << '/' << shape.sdd_mm << " mm coverage="
              << std::fixed << std::setprecision(1)
              << flatCoveragePercent(shape) << "% buffers(volume/projection)="
              << std::setprecision(2) << gibibytes(volume_count) << '/'
              << gibibytes(projection_count) << " GiB\n";
    printTableHeader();
    for (const auto& forward : forwards) {
        for (const auto& back : backs) {
            const double error = relativeAdjointError(forward.inner_product,
                back.inner_product);
            const bool exact_siddon = forward.task == ETask::FP_Siddon &&
                back.task == ETask::BP_Siddon_RayDriven;
            const bool matched_joseph = forward.task == ETask::FP_Joseph &&
                back.task == ETask::BP_Joseph;
            printTableRow(forward.name, back.name, forward.milliseconds,
                back.milliseconds, error,
                exact_siddon ? "exact" : matched_joseph ? "matched" : "cross");
            EXPECT_TRUE(std::isfinite(error));
            if (exact_siddon) EXPECT_LT(error, exact_siddon_tolerance);
            if (matched_joseph) EXPECT_LT(error, 5e-3);
        }
    }
    printTableBorder();
    cudaStreamDestroy(stream);
}

Mem::Tex3DHandle makeTexture(const float* source, int nx, int ny, int nz,
    cudaTextureFilterMode filter, cudaStream_t stream)
{
    auto texture = Mem::TextureController::createEmptyTex3D(nx, ny, nz,
        filter, cudaAddressModeBorder);
    Mem::TextureController::updateTex3DFromDeviceAsync(texture, source, nx, ny,
        nz, stream);
    return texture;
}

TEST(RepresentativeCylOperators, CrossMatrixAdjointAndTiming)
{
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device is unavailable";
    const auto shape = benchmarkShape();
    const int nx = shape.nx, ny = shape.ny, nz = shape.nz;
    const int channels = shape.channels, rows = shape.rows, views = shape.views;
    const size_t volume_count = static_cast<size_t>(nx) * ny * nz;
    const size_t projection_count = static_cast<size_t>(channels) * rows * views;
    const auto x = randomVector(volume_count, 24680u);
    const auto y = randomVector(projection_count, 86420u);

    SCircularTrajectorySpec trajectory{};
    trajectory.sid_mm = shape.sid_mm;
    trajectory.sdd_mm = shape.sdd_mm;
    trajectory.angles_rad.resize(views);
    for (int i = 0; i < views; ++i)
        trajectory.angles_rad[i] = kTwoPi * static_cast<float>(i) / views;
    SCylDetectorSpec detector{};
    detector.channels = channels;
    detector.rows = rows;
    detector.channel_arc_mm = shape.pixel_mm;
    detector.row_size_mm = shape.pixel_mm;
    detector.curvature_radius_mm = shape.cyl_radius_mm;
    std::vector<SCylConeProjGeomVec> geometry;
    ASSERT_TRUE(buildProjectionGeometry(trajectory, detector, geometry));
    const SVolGeom volume = SVolGeom::make_centered(nx, ny, nz,
        shape.voxel_mm, shape.voxel_mm, shape.voxel_mm);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    Mem::MemoryController memory;
    auto d_x = memory.allocateDevice3D<float>(nx, ny, nz, 0);
    auto d_y = memory.allocateDevice3D<float>(channels, rows, views, 0);
    auto d_projection = memory.allocateDevice3D<float>(channels, rows, views, 0);
    auto d_volume = memory.allocateDevice3D<float>(nx, ny, nz, 0);
    ASSERT_EQ(cudaMemcpyAsync(d_x.data(), x.data(), volume_count * sizeof(float),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_y.data(), y.data(), projection_count * sizeof(float),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    auto linear_volume = makeTexture(d_x.data(), nx, ny, nz,
        cudaFilterModeLinear, stream);
    auto point_volume = makeTexture(d_x.data(), nx, ny, nz,
        cudaFilterModePoint, stream);
    auto point_projection = makeTexture(d_y.data(), channels, rows, views,
        cudaFilterModePoint, stream);
    const auto prepared = CylFpBp::detail::prepareJosephGeometry(volume,
        channels, rows, geometry);
    ASSERT_TRUE(prepared);
    CylFpBp::CylForwardOperator forward;
    CylFpBp::CylBackOperator back;
    ASSERT_TRUE(forward.prepare(prepared));
    ASSERT_TRUE(back.prepare(prepared));

    struct Result { const char* name; double value; float milliseconds; };
    Result fp[3]{};
    fp[0].name = "Joseph";
    fp[0].milliseconds = timeGpu(stream, [&] {
        return forward.forward(linear_volume, d_projection.data(), stream);
    });
    std::vector<float> ax(projection_count);
    ASSERT_EQ(cudaMemcpy(ax.data(), d_projection.data(),
        projection_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    fp[0].value = dot(ax, y);
    fp[1].name = "Siddon";
    fp[1].milliseconds = timeGpu(stream, [&] {
        return forward.forwardSiddon(point_volume, d_projection.data(), stream);
    });
    ASSERT_EQ(cudaMemcpy(ax.data(), d_projection.data(),
        projection_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    fp[1].value = dot(ax, y);
    fp[2].name = "JosephLayered";
    fp[2].milliseconds = timeGpu(stream, [&] {
        return forward.forwardLayered(point_volume, d_projection.data(), stream);
    });
    ASSERT_EQ(cudaMemcpy(ax.data(), d_projection.data(),
        projection_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    fp[2].value = dot(ax, y);

    Result bp[4]{};
    bp[0].name = "Joseph";
    std::vector<float> aty(volume_count);
    const bool include_slow_reference = includeSlowReferenceOperators();
    if (include_slow_reference) {
        bp[0].milliseconds = timeGpu(stream, [&] {
            return back.backproject(point_projection, d_volume.data(), stream);
        });
        ASSERT_EQ(cudaMemcpy(aty.data(), d_volume.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        bp[0].value = dot(x, aty);
    }
    bp[1].name = "Siddon";
    bp[1].milliseconds = timeGpu(stream, [&] {
        return back.backprojectSiddon(point_projection, d_volume.data(), stream);
    });
    ASSERT_EQ(cudaMemcpy(aty.data(), d_volume.data(),
        volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    bp[1].value = dot(x, aty);
    bp[2].name = "JosephV3";
    CylFpBp::detail::JosephV3Strategy joseph_v3;
    ASSERT_TRUE(joseph_v3.prepare(volume, channels, rows, geometry, stream));
    ASSERT_TRUE(joseph_v3.uploadProjection(d_y.data()));
    bp[2].milliseconds = timeGpu(stream, [&] {
        return joseph_v3.backproject(d_volume.data());
    });
    ASSERT_EQ(cudaMemcpy(aty.data(), d_volume.data(),
        volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    bp[2].value = dot(x, aty);
    bp[3].name = "JosephGather";
    bp[3].milliseconds = timeGpu(stream, [&] {
        return back.backprojectGather(point_projection, d_volume.data(), stream);
    });
    ASSERT_EQ(cudaMemcpy(aty.data(), d_volume.data(),
        volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    bp[3].value = dot(x, aty);

    std::cout << std::defaultfloat << "\nCyl " << shape.name << " matrix: volume="
              << nx << 'x' << ny << 'x' << nz << " detector=" << channels
              << 'x' << rows << " views=" << views << " voxel="
              << shape.voxel_mm << " mm pixel=" << shape.pixel_mm
              << " mm SID/SDD/R=" << shape.sid_mm << '/' << shape.sdd_mm
              << '/' << shape.cyl_radius_mm << " mm coverage=" << std::fixed
              << std::setprecision(1) << cylCoveragePercent(shape)
              << "% buffers(volume/projection)=" << std::setprecision(2)
              << gibibytes(volume_count) << '/' << gibibytes(projection_count)
              << " GiB\n";
    printTableHeader();
    if (!include_slow_reference)
        std::cout << "| slow atomic Joseph BP omitted; set "
                  << "YKCBCT_INCLUDE_SLOW_REFERENCE=1 to include it.       |\n";
    for (int f = 0; f < 3; ++f) {
        for (int b = include_slow_reference ? 0 : 1; b < 4; ++b) {
            const double error = relativeAdjointError(fp[f].value, bp[b].value);
            const bool exact = f == b && f == 1;
            const bool matched = f == b && f == 0;
            const bool exact_gather = f == 2 && b == 3;
            printTableRow(fp[f].name, bp[b].name, fp[f].milliseconds,
                bp[b].milliseconds, error,
                exact || exact_gather ? "exact" : matched ? "matched" : "cross");
            EXPECT_TRUE(std::isfinite(error));
            if (matched) EXPECT_LT(error, 5e-3);
            if (exact) EXPECT_LT(error, 1e-4);
            if (exact_gather) EXPECT_LT(error, 1e-4);
        }
    }
    printTableBorder();
    joseph_v3.release();
    forward.release();
    back.release();
    cudaStreamDestroy(stream);
}

} // namespace
