#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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

struct AdjointSimilarity {
    double inner_product_error = 0.0;
    double fitted_relative_l2 = 0.0;
    double scale_to_reference = 0.0;
    double cosine_similarity = 0.0;
};

AdjointSimilarity compareAdjointResult(double forward_inner_product,
    const std::vector<float>& candidate, const std::vector<float>& reference,
    const std::vector<float>& volume)
{
    const double candidate_inner_product = dot(volume, candidate);
    double candidate_sq = 0.0;
    double reference_sq = 0.0;
    double cross = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = candidate[i];
        const double b = reference[i];
        candidate_sq += a * a;
        reference_sq += b * b;
        cross += a * b;
    }

    AdjointSimilarity result;
    result.inner_product_error = relativeAdjointError(forward_inner_product,
        candidate_inner_product);
    result.scale_to_reference = cross / std::max(candidate_sq, 1e-30);
    double fitted_difference_sq = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = result.scale_to_reference * candidate[i] -
            reference[i];
        fitted_difference_sq += difference * difference;
    }
    result.fitted_relative_l2 = std::sqrt(fitted_difference_sq /
        std::max(reference_sq, 1e-30));
    result.cosine_similarity = cross / std::sqrt(std::max(
        candidate_sq * reference_sq, 1e-30));
    return result;
}

void printAdjointSimilarityRow(const char* name,
    const AdjointSimilarity& result)
{
    std::cout << "| " << std::left << std::setw(18) << name
              << " | " << std::right << std::scientific
              << std::setprecision(3) << std::setw(12)
              << result.inner_product_error
              << " | " << std::fixed << std::setprecision(6)
              << std::setw(12) << result.scale_to_reference
              << " | " << std::setw(12)
              << 100.0 * result.fitted_relative_l2 << "%"
              << " | " << std::setw(12) << result.cosine_similarity
              << " |\n";
}

struct TimingReportRow {
    std::string name;
    float milliseconds;
};

struct AdjointReportRow {
    std::string forward;
    std::string back;
    float forward_ms;
    float back_ms;
    double error;
    std::string relation;
};

void printTimingTableBorder(std::ostream& output)
{
    output << "+--------------------+--------------+--------------+\n";
}

void printTimingTableHeader(std::ostream& output, const char* direction)
{
    output << '\n' << direction << " timing:\n";
    printTimingTableBorder(output);
    output << "| Operator           | avg ms       | ms/view      |\n";
    printTimingTableBorder(output);
}

void printTimingTableRow(std::ostream& output, const char* name,
    float milliseconds, int views)
{
    output << "| " << std::left << std::setw(18) << name
           << " | " << std::right << std::fixed << std::setprecision(3)
           << std::setw(12) << milliseconds
           << " | " << std::setw(12) << milliseconds / views << " |\n";
}

void printAdjointTableBorder(std::ostream& output)
{
    output << "+--------------------+--------------------+------------+------------+------------+--------------+----------+\n";
}

void printAdjointTableHeader(std::ostream& output)
{
    output << "\nFP/BP adjoint combinations:\n";
    printAdjointTableBorder(output);
    output << "| FP                 | BP                 | FP ms      | BP ms      | total ms   | adjoint err% | relation |\n";
    printAdjointTableBorder(output);
}

void printAdjointTableRow(std::ostream& output, const char* fp, const char* bp,
    float fp_ms, float bp_ms, double error, const char* relation)
{
    output << "| " << std::left << std::setw(18) << fp
           << " | " << std::setw(18) << bp
           << " | " << std::right << std::fixed << std::setprecision(3)
           << std::setw(10) << fp_ms
           << " | " << std::setw(10) << bp_ms
           << " | " << std::setw(10) << fp_ms + bp_ms
           << " | " << std::scientific << std::setprecision(3)
           << std::setw(12) << 100.0 * error
           << " | " << std::left << std::setw(8) << relation << " |\n";
}

void printTextReport(std::ostream& output, const std::string& description,
    int views, const std::vector<TimingReportRow>& forwards,
    const std::vector<TimingReportRow>& backs,
    const std::vector<AdjointReportRow>& combinations)
{
    output << description << '\n';
    printTimingTableHeader(output, "Forward projection");
    for (const auto& row : forwards)
        printTimingTableRow(output, row.name.c_str(), row.milliseconds, views);
    printTimingTableBorder(output);
    printTimingTableHeader(output, "Backprojection");
    for (const auto& row : backs)
        printTimingTableRow(output, row.name.c_str(), row.milliseconds, views);
    printTimingTableBorder(output);
    printAdjointTableHeader(output);
    for (const auto& row : combinations) {
        printAdjointTableRow(output, row.forward.c_str(), row.back.c_str(),
            row.forward_ms, row.back_ms, row.error, row.relation.c_str());
    }
    printAdjointTableBorder(output);
}

void writeSvgDashboard(const std::filesystem::path& file,
    const std::string& description,
    const std::vector<TimingReportRow>& forwards,
    const std::vector<TimingReportRow>& backs,
    const std::vector<AdjointReportRow>& combinations)
{
    constexpr int width = 1400;
    constexpr int panel_width = 650;
    constexpr int label_width = 190;
    constexpr int bar_width = 390;
    constexpr int row_height = 30;
    const int timing_rows = static_cast<int>(std::max(forwards.size(),
        backs.size()));
    const int heatmap_y = 150 + timing_rows * row_height + 100;
    const int back_count = std::max(static_cast<int>(backs.size()), 1);
    const int cell_width = std::min(90,
        (width - label_width - 60) / back_count);
    constexpr int cell_height = 48;
    const int height = heatmap_y + 110 +
        static_cast<int>(forwards.size()) * cell_height + 70;

    std::ofstream output(file);
    ASSERT_TRUE(output) << "Failed to open SVG benchmark report";
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
           << "\" height=\"" << height << "\" viewBox=\"0 0 " << width
           << ' ' << height << "\">\n"
           << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n"
           << "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#1f2937}"
           << ".title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#4b5563}"
           << ".heading{font-size:18px;font-weight:600}.label{font-size:13px}"
           << ".value{font-size:12px}.cell{font-size:10px;text-anchor:middle}</style>\n"
           << "<text x=\"30\" y=\"36\" class=\"title\">Operator benchmark overview</text>\n"
           << "<text x=\"30\" y=\"62\" class=\"subtitle\">" << description
           << "</text>\n";

    const auto draw_timing = [&](int origin_x, const char* title,
                                 const std::vector<TimingReportRow>& rows,
                                 const char* color) {
        float maximum = 0.f;
        for (const auto& row : rows) maximum = std::max(maximum, row.milliseconds);
        output << "<text x=\"" << origin_x << "\" y=\"105\" class=\"heading\">"
               << title << "</text>\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const int y = 130 + static_cast<int>(i) * row_height;
            const int pixels = maximum > 0.f
                ? static_cast<int>(bar_width * rows[i].milliseconds / maximum)
                : 0;
            output << "<text x=\"" << origin_x << "\" y=\"" << y + 15
                   << "\" class=\"label\">" << rows[i].name << "</text>\n"
                   << "<rect x=\"" << origin_x + 145 << "\" y=\"" << y
                   << "\" width=\"" << pixels << "\" height=\"20\" rx=\"3\" fill=\""
                   << color << "\"/>\n"
                   << "<text x=\"" << origin_x + 155 + pixels << "\" y=\""
                   << y + 15 << "\" class=\"value\">" << std::fixed
                   << std::setprecision(1) << rows[i].milliseconds << " ms</text>\n";
        }
    };
    draw_timing(30, "Forward projection timing", forwards, "#2563eb");
    draw_timing(30 + panel_width, "Backprojection timing", backs, "#7c3aed");

    output << "<text x=\"30\" y=\"" << heatmap_y
           << "\" class=\"heading\">Adjoint error heatmap (%)</text>\n";
    const int grid_y = heatmap_y + 80;
    for (size_t b = 0; b < backs.size(); ++b) {
        const int x = label_width + static_cast<int>(b) * cell_width + cell_width / 2;
        output << "<text x=\"" << x << "\" y=\"" << grid_y - 12
               << "\" class=\"label\" text-anchor=\"start\" transform=\"rotate(-45 "
               << x << ' ' << grid_y - 12 << ")\">" << backs[b].name << "</text>\n";
    }
    for (size_t f = 0; f < forwards.size(); ++f) {
        const int y = grid_y + static_cast<int>(f) * cell_height;
        output << "<text x=\"30\" y=\"" << y + 29
               << "\" class=\"label\">" << forwards[f].name << "</text>\n";
        for (size_t b = 0; b < backs.size(); ++b) {
            const auto& row = combinations[f * backs.size() + b];
            const double percent = std::max(100.0 * row.error, 1e-8);
            const double level = std::clamp((std::log10(percent) + 5.0) / 7.0,
                0.0, 1.0);
            const int red = level < 0.5 ? static_cast<int>(510.0 * level) : 255;
            const int green = level < 0.5 ? 190 :
                static_cast<int>(190.0 * (2.0 - 2.0 * level));
            const int blue = level < 0.5 ? 90 : 40;
            const int x = label_width + static_cast<int>(b) * cell_width;
            std::ostringstream value;
            value << std::scientific << std::setprecision(1) << percent;
            output << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\""
                   << cell_width - 2 << "\" height=\"" << cell_height - 2
                   << "\" fill=\"rgb(" << red << ',' << green << ',' << blue
                   << ")\"/>\n"
                   << "<text x=\"" << x + cell_width / 2 << "\" y=\"" << y + 29
                   << "\" class=\"cell\">" << value.str() << "</text>\n";
        }
    }
    output << "</svg>\n";
}

void writeBenchmarkReports(const std::string& stem,
    const std::string& description, int views,
    const std::vector<TimingReportRow>& forwards,
    const std::vector<TimingReportRow>& backs,
    const std::vector<AdjointReportRow>& combinations)
{
    const char* directory_value =
        std::getenv("YKCBCT_OPERATOR_BENCHMARK_OUTPUT_DIR");
    if (!directory_value || !*directory_value) return;

    const std::filesystem::path directory(directory_value);
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    ASSERT_FALSE(error) << "Failed to create benchmark output directory: "
                        << error.message();

    const auto path = [&](const std::string& suffix) {
        return directory / (stem + suffix);
    };
    writeSvgDashboard(path(".svg"), description, forwards, backs, combinations);
    {
        std::ofstream output(path(".txt"));
        ASSERT_TRUE(output) << "Failed to open TXT benchmark report";
        printTextReport(output, description, views, forwards, backs,
            combinations);
    }
    const auto write_timing_csv = [&](const std::filesystem::path& file,
                                      const std::vector<TimingReportRow>& rows) {
        std::ofstream output(file);
        ASSERT_TRUE(output) << "Failed to open timing CSV report";
        output << "operator,avg_ms,ms_per_view\n";
        output << std::fixed << std::setprecision(6);
        for (const auto& row : rows)
            output << row.name << ',' << row.milliseconds << ','
                   << row.milliseconds / views << '\n';
    };
    write_timing_csv(path("_fp_timing.csv"), forwards);
    write_timing_csv(path("_bp_timing.csv"), backs);
    {
        std::ofstream output(path("_adjoint.csv"));
        ASSERT_TRUE(output) << "Failed to open adjoint CSV report";
        output << "fp,bp,fp_ms,bp_ms,total_ms,adjoint_error_percent,relation\n";
        output << std::fixed << std::setprecision(6);
        for (const auto& row : combinations) {
            output << row.forward << ',' << row.back << ',' << row.forward_ms
                   << ',' << row.back_ms << ',' << row.forward_ms + row.back_ms
                   << ',' << 100.0 * row.error << ',' << row.relation << '\n';
        }
    }
    {
        std::ofstream output(path(".md"));
        ASSERT_TRUE(output) << "Failed to open Markdown benchmark report";
        output << "# Operator benchmark\n\n"
               << "![Operator benchmark overview](" << stem << ".svg)\n\n"
               << description << "\n\n"
               << "## Forward projection timing\n\n"
               << "| Operator | Average (ms) | Per view (ms) |\n"
               << "|---|---:|---:|\n" << std::fixed << std::setprecision(3);
        for (const auto& row : forwards)
            output << "| " << row.name << " | " << row.milliseconds << " | "
                   << row.milliseconds / views << " |\n";
        output << "\n## Backprojection timing\n\n"
               << "| Operator | Average (ms) | Per view (ms) |\n"
               << "|---|---:|---:|\n";
        for (const auto& row : backs)
            output << "| " << row.name << " | " << row.milliseconds << " | "
                   << row.milliseconds / views << " |\n";
        output << "\n## FP/BP adjoint combinations\n\n"
               << "| FP | BP | FP (ms) | BP (ms) | Total (ms) | Adjoint error (%) | Relation |\n"
               << "|---|---|---:|---:|---:|---:|---|\n";
        for (const auto& row : combinations) {
            output << "| " << row.forward << " | " << row.back << " | "
                   << row.forward_ms << " | " << row.back_ms << " | "
                   << row.forward_ms + row.back_ms << " | "
                   << 100.0 * row.error << " | " << row.relation << " |\n";
        }
    }
    std::cout << "Benchmark reports written to: "
              << std::filesystem::absolute(directory).string() << '\n';
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

float smoothInside(float normalized_radius, float transition)
{
    if (normalized_radius <= 1.f - transition) return 1.f;
    if (normalized_radius >= 1.f + transition) return 0.f;
    const float t = (1.f + transition - normalized_radius) /
        (2.f * transition);
    return t * t * (3.f - 2.f * t);
}

std::vector<float> smoothQualityPhantom(const SReconstructionParams& p)
{
    const int nx = p.volume.Nx;
    const int ny = p.volume.Ny;
    const int nz = p.volume.Nz;
    std::vector<float> volume(static_cast<size_t>(nx) * ny * nz, 0.f);
    for (int z = 0; z < nz; ++z) {
        const float pz = (z + 0.5f - 0.5f * nz) * p.volume.voxelZ_mm;
        for (int y = 0; y < ny; ++y) {
            const float py = (y + 0.5f - 0.5f * ny) * p.volume.voxelY_mm;
            for (int x = 0; x < nx; ++x) {
                const float px = (x + 0.5f - 0.5f * nx) * p.volume.voxelX_mm;
                const float body_radius = std::sqrt(
                    (px / 40.f) * (px / 40.f) +
                    (py / 34.f) * (py / 34.f) +
                    (pz / 28.f) * (pz / 28.f));
                float value = 0.018f * smoothInside(body_radius, 0.025f);

                const float insert_radius = std::sqrt(
                    ((px - 15.f) / 7.f) * ((px - 15.f) / 7.f) +
                    ((py + 8.f) / 7.f) * ((py + 8.f) / 7.f) +
                    ((pz - 4.f) / 10.f) * ((pz - 4.f) / 10.f));
                value += 0.012f * smoothInside(insert_radius, 0.10f);

                const float low_contrast_radius = std::sqrt(
                    ((px + 14.f) / 10.f) * ((px + 14.f) / 10.f) +
                    ((py - 7.f) / 8.f) * ((py - 7.f) / 8.f) +
                    ((pz + 5.f) / 12.f) * ((pz + 5.f) / 12.f));
                value += 0.002f * smoothInside(low_contrast_radius, 0.08f);
                volume[(static_cast<size_t>(z) * ny + y) * nx + x] = value;
            }
        }
    }
    return volume;
}

std::vector<float> thinWirePhantom(const SReconstructionParams& p)
{
    const int nx = p.volume.Nx;
    const int ny = p.volume.Ny;
    const int nz = p.volume.Nz;
    constexpr float radius_mm = 0.15f;
    constexpr float half_length_mm = 18.f;
    constexpr int subdivisions = 4;
    const float inverse_axis_norm = 1.f / std::sqrt(1.f + 0.42f * 0.42f);
    const float axis_x = 0.42f * inverse_axis_norm;
    const float axis_z = inverse_axis_norm;
    const float half_diagonal = 0.5f * std::sqrt(
        p.volume.voxelX_mm * p.volume.voxelX_mm +
        p.volume.voxelY_mm * p.volume.voxelY_mm +
        p.volume.voxelZ_mm * p.volume.voxelZ_mm);
    std::vector<float> volume(static_cast<size_t>(nx) * ny * nz, 0.f);
    for (int z = 0; z < nz; ++z) {
        const float center_z = (z + 0.5f - 0.5f * nz) *
            p.volume.voxelZ_mm;
        for (int y = 0; y < ny; ++y) {
            const float center_y = (y + 0.5f - 0.5f * ny) *
                p.volume.voxelY_mm;
            for (int x = 0; x < nx; ++x) {
                const float center_x = (x + 0.5f - 0.5f * nx) *
                    p.volume.voxelX_mm;
                const float center_t = center_x * axis_x + center_z * axis_z;
                const float center_radial_sq = std::max(0.f,
                    center_x * center_x + center_y * center_y +
                    center_z * center_z - center_t * center_t);
                if (std::fabs(center_t) > half_length_mm + half_diagonal ||
                    std::sqrt(center_radial_sq) > radius_mm + half_diagonal)
                    continue;

                int inside = 0;
                for (int sz = 0; sz < subdivisions; ++sz) {
                    const float pz = center_z +
                        ((sz + 0.5f) / subdivisions - 0.5f) *
                        p.volume.voxelZ_mm;
                    for (int sy = 0; sy < subdivisions; ++sy) {
                        const float py = center_y +
                            ((sy + 0.5f) / subdivisions - 0.5f) *
                            p.volume.voxelY_mm;
                        for (int sx = 0; sx < subdivisions; ++sx) {
                            const float px = center_x +
                                ((sx + 0.5f) / subdivisions - 0.5f) *
                                p.volume.voxelX_mm;
                            const float t = px * axis_x + pz * axis_z;
                            const float radial_sq = px * px + py * py + pz * pz -
                                t * t;
                            if (std::fabs(t) <= half_length_mm &&
                                radial_sq <= radius_mm * radius_mm)
                                ++inside;
                        }
                    }
                }
                volume[(static_cast<size_t>(z) * ny + y) * nx + x] =
                    0.5f * inside /
                    static_cast<float>(subdivisions * subdivisions *
                        subdivisions);
            }
        }
    }
    return volume;
}

struct ProjectionDifference {
    double relative_l2 = 0.0;
    double fitted_relative_l2 = 0.0;
    double scale_to_reference = 0.0;
    double cosine_similarity = 0.0;
    double normalized_max_error = 0.0;
};

ProjectionDifference compareProjection(const std::vector<float>& candidate,
    const std::vector<float>& reference)
{
    double candidate_sq = 0.0;
    double reference_sq = 0.0;
    double cross = 0.0;
    double difference_sq = 0.0;
    double reference_max = 0.0;
    double max_error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = candidate[i];
        const double b = reference[i];
        candidate_sq += a * a;
        reference_sq += b * b;
        cross += a * b;
        const double difference = a - b;
        difference_sq += difference * difference;
        reference_max = std::max(reference_max, std::fabs(b));
        max_error = std::max(max_error, std::fabs(difference));
    }
    ProjectionDifference result;
    result.relative_l2 = std::sqrt(difference_sq /
        std::max(reference_sq, 1e-30));
    result.scale_to_reference = cross / std::max(candidate_sq, 1e-30);
    double fitted_difference_sq = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = result.scale_to_reference * candidate[i] -
            reference[i];
        fitted_difference_sq += difference * difference;
    }
    result.fitted_relative_l2 = std::sqrt(fitted_difference_sq /
        std::max(reference_sq, 1e-30));
    result.cosine_similarity = cross / std::sqrt(std::max(
        candidate_sq * reference_sq, 1e-30));
    result.normalized_max_error = max_error / std::max(reference_max, 1e-30);
    return result;
}

void printProjectionDifference(const char* candidate, const char* reference,
    const ProjectionDifference& difference)
{
    std::cout << "| " << std::left << std::setw(10) << candidate
              << " | " << std::setw(8) << reference
              << " | " << std::right << std::fixed << std::setprecision(3)
              << std::setw(8) << 100.0 * difference.relative_l2 << "%"
              << " | " << std::setw(8)
              << 100.0 * difference.fitted_relative_l2 << "%"
              << " | " << std::setw(9) << difference.scale_to_reference
              << " | " << std::setw(9) << difference.cosine_similarity
              << " | " << std::setw(8)
              << 100.0 * difference.normalized_max_error << "% |\n";
}

TEST(RepresentativeFlatOperators, FootprintProjectionAgreement)
{
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device is unavailable";
    const BenchmarkShape shapes[] = {
        {128, 128, 96, 256, 160, 90, 0.8f, 0.8f,
            440.f, 770.f, 650.f, "balanced"},
        {96, 96, 64, 320, 208, 45, 1.2f, 0.5f,
            440.f, 770.f, 650.f, "large-voxel-small-pixel"},
        {192, 192, 128, 128, 96, 45, 0.5f, 1.2f,
            440.f, 770.f, 650.f, "small-voxel-large-pixel"},
        {256, 256, 192, 384, 256, 90, 0.25f, 0.417f,
            440.f, 770.f, 650.f, "thin-wire-0.30mm"},
    };
    for (const auto& shape : shapes) {
      const auto p = representativeFlatParams(shape);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) *
        p.scan.Nu * p.scan.Nv;
    const auto phantom = std::string(shape.name) == "thin-wire-0.30mm"
        ? thinWirePhantom(p) : smoothQualityPhantom(p);

    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    GeometryContext context;
    ASSERT_TRUE(context.initialize(p, geometry));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ResourceContext resources;
    resources.attach(stream, 0);
    Mem::MemoryController memory;
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng, 0);
    ASSERT_EQ(cudaMemcpyAsync(d_volume.data(), phantom.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream),
        cudaSuccess);

    struct ProjectionResult {
        const char* name;
        ETask task;
        std::vector<float> values;
    };
    ProjectionResult results[] = {
        {"Joseph", ETask::FP_Joseph, {}},
        {"Siddon", ETask::FP_Siddon, {}},
        {"Footprint", ETask::FP_SeparableFootprint, {}},
    };
    for (auto& result : results) {
        auto op = makeForwardOperator(result.task);
        ASSERT_TRUE(op && op->prepare(context, resources));
        ASSERT_TRUE(op->apply(d_volume.data(), p, d_projection.data(), resources));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        result.values.resize(projection_count);
        ASSERT_EQ(cudaMemcpy(result.values.data(), d_projection.data(),
            projection_count * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
        op->release();
    }

    const auto footprint_joseph = compareProjection(results[2].values,
        results[0].values);
    const auto footprint_siddon = compareProjection(results[2].values,
        results[1].values);
    const auto joseph_siddon = compareProjection(results[0].values,
        results[1].values);
    std::cout << "\nProjection agreement (" << shape.name
              << "): volume=" << shape.nx << 'x' << shape.ny << 'x'
              << shape.nz << " detector=" << shape.channels << 'x'
              << shape.rows << " views=" << shape.views << " voxel/pixel="
              << shape.voxel_mm << '/' << shape.pixel_mm << " mm\n"
              << "+------------+----------+-----------+-----------+-----------+-----------+----------+\n"
              << "| candidate  | reference| rel L2    | fit L2    | fit scale | cosine    | max/ref  |\n"
              << "+------------+----------+-----------+-----------+-----------+-----------+----------+\n";
    printProjectionDifference("Footprint", "Joseph", footprint_joseph);
    printProjectionDifference("Footprint", "Siddon", footprint_siddon);
    printProjectionDifference("Joseph", "Siddon", joseph_siddon);
    std::cout << "+------------+----------+-----------+-----------+-----------+-----------+----------+\n";

    const double minimum_cosine = std::string(shape.name) ==
        "thin-wire-0.30mm" ? 0.95 : 0.99;
    EXPECT_GT(footprint_joseph.cosine_similarity, minimum_cosine);
    EXPECT_GT(footprint_siddon.cosine_similarity, minimum_cosine);
    cudaStreamDestroy(stream);
    }
}

TEST(RepresentativeFlatOperators, FootprintAccumulationDynamicRange)
{
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device is unavailable";
    const BenchmarkShape shape{96, 96, 96, 192, 128, 8, 0.5f, 0.5f,
        440.f, 770.f, 650.f, "dynamic-range"};
    const auto p = representativeFlatParams(shape);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) *
        p.scan.Nu * p.scan.Nv;
    std::vector<float> background(volume_count, 1.f);
    std::vector<float> wire(volume_count, 0.f);
    for (int z = 0; z < p.volume.Nz; ++z) {
        for (int y = p.volume.Ny / 2 - 1; y <= p.volume.Ny / 2; ++y) {
            for (int x = p.volume.Nx / 2 - 1; x <= p.volume.Nx / 2; ++x) {
                wire[(static_cast<size_t>(z) * p.volume.Ny + y) *
                    p.volume.Nx + x] = 1.f;
            }
        }
    }

    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    GeometryContext context;
    ASSERT_TRUE(context.initialize(p, geometry));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ResourceContext resources;
    resources.attach(stream, 0);
    Mem::MemoryController memory;
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng, 0);
    struct DynamicRangeOperator { const char* name; ETask task; };
    const DynamicRangeOperator operators[] = {
        {"Joseph", ETask::FP_Joseph},
        {"Siddon", ETask::FP_Siddon},
        {"Footprint", ETask::FP_SeparableFootprint},
    };
    std::cout << "\nFloat FP accumulation dynamic range\n"
              << "+-----------+---------------+----------------+----------------+----------------+\n"
              << "| operator  | wire/background | recovered scale| relative error | zeroed samples |\n"
              << "+-----------+---------------+----------------+----------------+----------------+\n";
    const float contrasts[] = {1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f};
    for (const auto& item : operators) {
      auto op = makeForwardOperator(item.task);
      ASSERT_TRUE(op && op->prepare(context, resources));
      auto project = [&](const std::vector<float>& volume) {
          std::vector<float> projection(projection_count);
          EXPECT_EQ(cudaMemcpyAsync(d_volume.data(), volume.data(),
              volume_count * sizeof(float), cudaMemcpyHostToDevice, stream),
              cudaSuccess);
          EXPECT_TRUE(op->apply(d_volume.data(), p, d_projection.data(),
              resources));
          EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
          EXPECT_EQ(cudaMemcpy(projection.data(), d_projection.data(),
              projection_count * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
          return projection;
      };
      const auto background_projection = project(background);
    for (float contrast : contrasts) {
        std::vector<float> combined = background;
        std::vector<float> delta(volume_count, 0.f);
        for (size_t i = 0; i < volume_count; ++i) {
            const float addition = contrast * wire[i];
            combined[i] += addition;
            delta[i] = addition;
        }
        const auto combined_projection = project(combined);
        const auto expected_projection = project(delta);
        double expected_sq = 0.0;
        double error_sq = 0.0;
        double recovered_cross = 0.0;
        size_t supported = 0;
        size_t zeroed = 0;
        for (size_t i = 0; i < projection_count; ++i) {
            const double expected = expected_projection[i];
            const double recovered = static_cast<double>(combined_projection[i]) -
                background_projection[i];
            expected_sq += expected * expected;
            recovered_cross += recovered * expected;
            const double error = recovered - expected;
            error_sq += error * error;
            if (expected != 0.0) {
                ++supported;
                if (recovered == 0.0) ++zeroed;
            }
        }
        const double recovered_scale = recovered_cross /
            std::max(expected_sq, 1e-30);
        const double relative_error = std::sqrt(error_sq /
            std::max(expected_sq, 1e-30));
        const double zeroed_fraction = static_cast<double>(zeroed) /
            std::max<size_t>(supported, 1);
        std::cout << "| " << std::left << std::setw(9) << item.name
                  << " | " << std::right << std::scientific
                  << std::setprecision(1) << std::setw(13) << contrast
                  << " | " << std::fixed
                  << std::setprecision(6) << std::setw(14) << recovered_scale
                  << " | " << std::setw(13) << 100.0 * relative_error
                  << "% | " << std::setw(13) << 100.0 * zeroed_fraction
                  << "% |\n";
        EXPECT_TRUE(std::isfinite(recovered_scale));
        EXPECT_TRUE(std::isfinite(relative_error));
    }
    op->release();
    }
    std::cout << "+-----------+---------------+----------------+----------------+----------------+\n";
    cudaStreamDestroy(stream);
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
        {"Footprint", ETask::FP_SeparableFootprint},
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
        {"Footprint", ETask::BP_SeparableFootprint},
        {"FDK", ETask::BP_FDK},
        {"FDKPseudoMatched", ETask::BP_FDK_PseudoMatched},
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

    std::ostringstream description;
    description << std::defaultfloat << "Flat " << shape.name
                << " matrix: volume=" << shape.nx << 'x' << shape.ny << 'x'
                << shape.nz << " detector=" << shape.channels << 'x'
                << shape.rows << " views=" << shape.views << " voxel="
                << shape.voxel_mm << " mm pixel=" << shape.pixel_mm
                << " mm SID/SDD=" << shape.sid_mm << '/' << shape.sdd_mm
                << " mm coverage=" << std::fixed << std::setprecision(1)
                << flatCoveragePercent(shape)
                << "% buffers(volume/projection)=" << std::setprecision(2)
                << gibibytes(volume_count) << '/' << gibibytes(projection_count)
                << " GiB";
    std::vector<TimingReportRow> forward_rows;
    std::vector<TimingReportRow> back_rows;
    std::vector<AdjointReportRow> combination_rows;
    for (const auto& forward : forwards)
        forward_rows.push_back({forward.name, forward.milliseconds});
    for (const auto& back : backs)
        back_rows.push_back({back.name, back.milliseconds});
    for (const auto& forward : forwards) {
        for (const auto& back : backs) {
            const double error = relativeAdjointError(forward.inner_product,
                back.inner_product);
            const bool exact_siddon = forward.task == ETask::FP_Siddon &&
                back.task == ETask::BP_Siddon_RayDriven;
            const bool matched_joseph = forward.task == ETask::FP_Joseph &&
                back.task == ETask::BP_Joseph;
            const bool exact_footprint =
                forward.task == ETask::FP_SeparableFootprint &&
                back.task == ETask::BP_SeparableFootprint;
            combination_rows.push_back({forward.name, back.name,
                forward.milliseconds, back.milliseconds, error,
                exact_siddon || exact_footprint ? "exact" :
                matched_joseph ? "matched" : "cross"});
            EXPECT_TRUE(std::isfinite(error));
            if (exact_siddon) EXPECT_LT(error, exact_siddon_tolerance);
            if (matched_joseph) EXPECT_LT(error, 5e-3);
            if (exact_footprint) EXPECT_LT(error, exact_siddon_tolerance);
        }
    }
    std::cout << '\n';
    printTextReport(std::cout, description.str(), shape.views, forward_rows,
        back_rows, combination_rows);
    writeBenchmarkReports(std::string("flat_") + shape.name,
        description.str(), shape.views, forward_rows, back_rows,
        combination_rows);
    cudaStreamDestroy(stream);
}

TEST(RepresentativeFlatOperators, TigreStylePseudoMatchedAdjoint)
{
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device is unavailable";

    // TIGRE-like cone-beam proportions with a compact matrix: smooth,
    // non-negative attenuation data and a physically generated projection
    // vector.  The exact Siddon ray BP is the discrete A^T reference.
    const BenchmarkShape shape{128, 128, 96, 256, 192, 90, 1.f, 0.8f,
        1000.f, 1536.f, 1536.f, "tigre-style"};
    const auto p = representativeFlatParams(shape);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) *
        p.scan.Nu * p.scan.Nv;
    const auto x = smoothQualityPhantom(p);

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
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    ASSERT_EQ(cudaMemcpyAsync(d_x.data(), x.data(), volume_count * sizeof(float),
        cudaMemcpyHostToDevice, stream), cudaSuccess);

    auto forward = makeForwardOperator(ETask::FP_Siddon);
    ASSERT_TRUE(forward && forward->prepare(context, resources));
    ASSERT_TRUE(forward->apply(d_x.data(), p, d_y.data(), resources));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<float> y(projection_count);
    ASSERT_EQ(cudaMemcpy(y.data(), d_y.data(), projection_count * sizeof(float),
        cudaMemcpyDeviceToHost), cudaSuccess);
    const double forward_inner_product = dot(y, y);
    forward->release();

    struct BackprojectionResult {
        const char* name;
        ETask task;
        std::vector<float> values;
    };
    BackprojectionResult results[] = {
        {"SiddonRay", ETask::BP_Siddon_RayDriven, {}},
        {"FDKPseudoMatched", ETask::BP_FDK_PseudoMatched, {}},
    };
    for (auto& result : results) {
        auto back = makeBackOperator(result.task);
        ASSERT_TRUE(back && back->prepare(context, resources));
        ASSERT_TRUE(back->apply(d_y.data(), p, d_volume.data(), true, resources));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        result.values.resize(volume_count);
        ASSERT_EQ(cudaMemcpy(result.values.data(), d_volume.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        back->release();
    }

    const auto exact = compareAdjointResult(forward_inner_product,
        results[0].values, results[0].values, x);
    const auto pseudo_matched = compareAdjointResult(forward_inner_product,
        results[1].values, results[0].values, x);

    std::cout << std::defaultfloat
              << "\nTIGRE-style non-negative smooth adjoint comparison: volume="
              << shape.nx << 'x' << shape.ny << 'x' << shape.nz
              << " detector=" << shape.channels << 'x' << shape.rows
              << " views=" << shape.views << " voxel/pixel="
              << shape.voxel_mm << '/' << shape.pixel_mm
              << " mm SID/SDD=" << shape.sid_mm << '/' << shape.sdd_mm
              << " mm coverage=" << std::fixed << std::setprecision(1)
              << flatCoveragePercent(shape) << "%\n"
              << "+--------------------+--------------+--------------+--------------+--------------+\n"
              << "| BP                 | inner error  | fit scale    | fitted L2    | cosine       |\n"
              << "+--------------------+--------------+--------------+--------------+--------------+\n";
    printAdjointSimilarityRow(results[0].name, exact);
    printAdjointSimilarityRow(results[1].name, pseudo_matched);
    std::cout << "+--------------------+--------------+--------------+--------------+--------------+\n";

    EXPECT_LT(exact.inner_product_error, 1e-4);
    EXPECT_LT(exact.fitted_relative_l2, 1e-6);
    EXPECT_GT(exact.cosine_similarity, 1.0 - 1e-6);
    EXPECT_TRUE(std::isfinite(pseudo_matched.inner_product_error));
    EXPECT_TRUE(std::isfinite(pseudo_matched.fitted_relative_l2));
    EXPECT_TRUE(std::isfinite(pseudo_matched.scale_to_reference));
    EXPECT_TRUE(std::isfinite(pseudo_matched.cosine_similarity));
    EXPECT_LT(pseudo_matched.inner_product_error, 1e-2);
    EXPECT_LT(pseudo_matched.fitted_relative_l2, 0.10);
    EXPECT_GT(pseudo_matched.cosine_similarity, 0.99);
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
    Result fp[4]{};
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
    fp[3].name = "Footprint";
    fp[3].milliseconds = timeGpu(stream, [&] {
        return forward.forwardFootprint(d_x.data(), d_projection.data(), stream);
    });
    ASSERT_EQ(cudaMemcpy(ax.data(), d_projection.data(),
        projection_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    fp[3].value = dot(ax, y);

    Result bp[5]{};
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
    bp[4].name = "Footprint";
    bp[4].milliseconds = timeGpu(stream, [&] {
        return back.backprojectFootprint(d_y.data(), d_volume.data(), stream);
    });
    ASSERT_EQ(cudaMemcpy(aty.data(), d_volume.data(),
        volume_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    bp[4].value = dot(x, aty);

    std::ostringstream description;
    description << std::defaultfloat << "Cyl " << shape.name
                << " matrix: volume=" << nx << 'x' << ny << 'x' << nz
                << " detector=" << channels << 'x' << rows << " views="
                << views << " voxel=" << shape.voxel_mm << " mm pixel="
                << shape.pixel_mm << " mm SID/SDD/R=" << shape.sid_mm << '/'
                << shape.sdd_mm << '/' << shape.cyl_radius_mm
                << " mm coverage=" << std::fixed << std::setprecision(1)
                << cylCoveragePercent(shape)
                << "% buffers(volume/projection)=" << std::setprecision(2)
                << gibibytes(volume_count) << '/' << gibibytes(projection_count)
                << " GiB";
    std::vector<TimingReportRow> forward_rows;
    std::vector<TimingReportRow> back_rows;
    std::vector<AdjointReportRow> combination_rows;
    for (const auto& item : fp)
        forward_rows.push_back({item.name, item.milliseconds});
    for (int b = include_slow_reference ? 0 : 1; b < 5; ++b)
        back_rows.push_back({bp[b].name, bp[b].milliseconds});
    if (!include_slow_reference)
        std::cout << "Slow atomic Joseph BP omitted; set "
                  << "YKCBCT_INCLUDE_SLOW_REFERENCE=1 to include it.\n";
    for (int f = 0; f < 4; ++f) {
        for (int b = include_slow_reference ? 0 : 1; b < 5; ++b) {
            const double error = relativeAdjointError(fp[f].value, bp[b].value);
            const bool exact = f == b && f == 1;
            const bool matched = f == b && f == 0;
            const bool exact_gather = f == 2 && b == 3;
            const bool exact_footprint = f == 3 && b == 4;
            combination_rows.push_back({fp[f].name, bp[b].name,
                fp[f].milliseconds, bp[b].milliseconds, error,
                exact || exact_gather || exact_footprint ? "exact" :
                matched ? "matched" : "cross"});
            EXPECT_TRUE(std::isfinite(error));
            if (matched) EXPECT_LT(error, 5e-3);
            if (exact) EXPECT_LT(error, 1e-4);
            if (exact_gather) EXPECT_LT(error, 1e-4);
            if (exact_footprint) EXPECT_LT(error, 1e-4);
        }
    }
    std::cout << '\n';
    printTextReport(std::cout, description.str(), views, forward_rows,
        back_rows, combination_rows);
    writeBenchmarkReports(std::string("cyl_") + shape.name,
        description.str(), views, forward_rows, back_rows, combination_rows);
    joseph_v3.release();
    forward.release();
    back.release();
    cudaStreamDestroy(stream);
}

} // namespace
