#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "DLTFpBp/YkDltAlgebraicReconstructor.hpp"
#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "Iter/YkAlgebraicReconstructorEx.hpp"
#include "YkTestImage.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int kChannels = 1024;
constexpr int kRows = 1024;
constexpr int kViews = 360;
constexpr int kNx = 768;
constexpr int kNy = 768;
// The calibration phantom is 108 mm long.  Keep the reconstruction centred
// on the source-trajectory frame and cover it with margin without using the
// unknown phantom installation translation or rotation.
constexpr int kNz = 600;
constexpr int kIterations = 1;
constexpr float kRelaxation = 0.2f;
constexpr double kPi = 3.14159265358979323846;

const fs::path kRoot = fs::path(YKCBCT_TEST_SOURCE_DIR);
const fs::path kProjectionFile = kRoot /
    "example/multispectrum_sim/outputs/calibration/"
    "double-ring-100mm-spacing100-bead3-cylinder6-6beads-spp10-"
    "projection-1024x1024x360-f32.raw";
const fs::path kMatrixFile = kRoot /
    "out/cbct_calibration/yang_standard_raw/"
    "projection_matrices_machine_360x3x4.f64";
const fs::path kOutput = kRoot /
    "out/cbct_calibration/yang_sart_full_resolution_comparison";
const fs::path kMechanicalOutput = kOutput /
    "mechanical-sart-768x768x600-f32.raw";
const fs::path kDltOutput = kOutput / "dlt-sart-768x768x600-f32.raw";

bool cudaOk(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

SReconstructionParams makeMechanicalParameters()
{
    SReconstructionParams p{};
    p.scan.Nu = kChannels; p.scan.Nv = kRows;
    p.scan.NAng = kViews; p.scan.totalViews = kViews;
    p.scan.du_mm = 0.417f; p.scan.dv_mm = 0.417f;
    p.scan.sid_mm = 440.f; p.scan.sdd_mm = 770.f;
    p.scan.offsetU_mm = 2.085f; p.scan.offsetV_mm = 4.170f;
    p.scan.tiltU_rad = static_cast<float>(1.0 * kPi / 180.0);
    p.scan.tiltV_rad = static_cast<float>(2.0 * kPi / 180.0);
    p.scan.tiltN_rad = static_cast<float>(3.0 * kPi / 180.0);
    p.scan.sourceOffsetX_mm = 0.f;
    p.scan.sourceOffsetY_mm = 0.f;
    p.scan.sourceOffsetZ_mm = 0.f;
    p.scan.start_angle_rad = 0.f;
    p.scan.range_rad = static_cast<float>(2.0 * kPi);
    p.scan.direction = 1;
    p.scan.angles.resize(kViews);
    for (int i = 0; i < kViews; ++i)
        p.scan.angles[i] = static_cast<float>(2.0 * kPi * i / kViews);
    p.volume.Nx = kNx; p.volume.Ny = kNy; p.volume.Nz = kNz;
    p.volume.voxelX_mm = 0.30f;
    p.volume.voxelY_mm = 0.30f;
    p.volume.voxelZ_mm = 0.30f;
    p.volume.centerX_mm = 0.f;
    p.volume.centerY_mm = 0.f;
    p.volume.centerZ_mm = 0.f;
    return p;
}

YK::SVolGeom makeVolumeGeometry()
{
    auto volume = YK::SVolGeom::make_centered(
        kNx, kNy, kNz, 0.30f, 0.30f, 0.30f);
    volume.center = make_float3(0.f, 0.f, 0.f);
    return volume;
}

struct MappedProjection {
    float* host = nullptr;
    float* device = nullptr;

    ~MappedProjection()
    {
        if (host) cudaFreeHost(host);
    }

    bool load()
    {
        const size_t count = static_cast<size_t>(kViews) * kRows * kChannels;
        const size_t bytes = count * sizeof(float);
        if (!fs::exists(kProjectionFile) || fs::file_size(kProjectionFile) != bytes)
            return false;
        if (!cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes,
                cudaHostAllocMapped | cudaHostAllocPortable),
                "allocate mapped projection")) return false;
        if (!cudaOk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0),
                "map projection")) return false;
        std::ifstream input(kProjectionFile, std::ios::binary);
        input.read(reinterpret_cast<char*>(host), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(input);
    }
};

bool readDltMatrices(std::vector<YK::DltFpBp::SDltProjectionMatrix>& matrices)
{
    const size_t count = static_cast<size_t>(kViews) * 12;
    if (!fs::exists(kMatrixFile) ||
        fs::file_size(kMatrixFile) != count * sizeof(double)) return false;
    std::vector<double> values(count);
    std::ifstream input(kMatrixFile, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(double)));
    if (!input) return false;
    matrices.resize(kViews);
    for (int view = 0; view < kViews; ++view) {
        for (int i = 0; i < 12; ++i)
            matrices[view].value[i] = values[static_cast<size_t>(view) * 12 + i];
        // Yang uses v-up; the RAW array uses row zero at the top.
        matrices[view] = YK::DltFpBp::flipImageV(matrices[view], kRows);
    }
    return true;
}

bool writeRaw(const fs::path& path, const std::vector<float>& values)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

float displayScale(const std::vector<float>& volume)
{
    std::vector<float> sample;
    sample.reserve(volume.size() / 97 + 1);
    for (size_t i = 0; i < volume.size(); i += 97)
        if (std::isfinite(volume[i]) && volume[i] > 0.f) sample.push_back(volume[i]);
    if (sample.empty()) return 1.f;
    const size_t index = std::min(sample.size() - 1,
        static_cast<size_t>(0.995 * sample.size()));
    std::nth_element(sample.begin(), sample.begin() + index, sample.end());
    return 1.f / std::max(sample[index], 1.e-8f);
}

struct VolumeStats {
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double l2 = 0.0;
};

VolumeStats statistics(const std::vector<float>& values)
{
    VolumeStats result{};
    if (values.empty()) return result;
    result.minimum = result.maximum = values.front();
    double sum = 0.0, squared = 0.0;
    for (float value : values) {
        result.minimum = std::min(result.minimum, static_cast<double>(value));
        result.maximum = std::max(result.maximum, static_cast<double>(value));
        sum += value;
        squared += static_cast<double>(value) * value;
    }
    result.mean = sum / values.size();
    result.l2 = std::sqrt(squared);
    return result;
}

double centerSliceCorrelation(const std::vector<float>& mechanical_slice,
    const std::vector<float>& dlt)
{
    double sum_a = 0.0, sum_b = 0.0, sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
    constexpr int lo_x = kNx / 4, hi_x = 3 * kNx / 4;
    constexpr int lo_y = kNy / 4, hi_y = 3 * kNy / 4;
    constexpr int z = kNz / 2;
    for (int y = lo_y; y < hi_y; ++y) {
        for (int x = lo_x; x < hi_x; ++x) {
            const size_t slice_i = static_cast<size_t>(y) * kNx + x;
            const size_t volume_i = (static_cast<size_t>(z) * kNy + y) * kNx + x;
            const double va = mechanical_slice[slice_i], vb = dlt[volume_i];
            sum_a += va; sum_b += vb;
            sum_aa += va * va; sum_bb += vb * vb; sum_ab += va * vb;
        }
    }
    constexpr double count = static_cast<double>(hi_x - lo_x) * (hi_y - lo_y);
    return (sum_ab - sum_a * sum_b / count) /
        std::sqrt((sum_aa - sum_a * sum_a / count) *
            (sum_bb - sum_b * sum_b / count));
}

std::vector<double> axialScores(const std::vector<float>& volume)
{
    std::vector<double> score(kNz, 0.0);
    for (int z = 0; z < kNz; ++z) {
        double sum = 0.0;
        for (int y = kNy / 4; y < 3 * kNy / 4; ++y) {
            for (int x = kNx / 4; x < 3 * kNx / 4; ++x) {
                const size_t i = (static_cast<size_t>(z) * kNy + y) * kNx + x;
                sum += std::max(0.f, volume[i]);
            }
        }
        score[z] = sum;
    }
    return score;
}

std::vector<int> selectAxialSlices(const std::vector<double>& mechanical,
    const std::vector<double>& dlt)
{
    std::vector<double> score(kNz);
    for (int z = 0; z < kNz; ++z) score[z] = mechanical[z] + dlt[z];
    std::vector<int> selected;
    for (int count = 0; count < 3; ++count) {
        int best = -1;
        for (int z = 0; z < kNz; ++z) {
            const bool separated = std::all_of(selected.begin(), selected.end(),
                [z](int prior) { return std::abs(z - prior) >= 40; });
            if (separated && (best < 0 || score[z] > score[best])) best = z;
        }
        if (best >= 0) selected.push_back(best);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

bool readSelectedSlices(const fs::path& path, const std::vector<int>& slices,
    std::vector<float>& output)
{
    const size_t slice_count = static_cast<size_t>(kNx) * kNy;
    output.resize(slice_count * slices.size());
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    for (size_t i = 0; i < slices.size(); ++i) {
        input.seekg(static_cast<std::streamoff>(slices[i] * slice_count * sizeof(float)));
        input.read(reinterpret_cast<char*>(output.data() + i * slice_count),
            static_cast<std::streamsize>(slice_count * sizeof(float)));
        if (!input) return false;
    }
    return true;
}

bool saveComparison(const std::vector<float>& dlt,
    const std::vector<double>& mechanical_scores,
    const std::vector<float>& mechanical_center_slice,
    float mechanical_display_scale, double mechanical_seconds,
    double dlt_seconds, const VolumeStats& mechanical_stats,
    const VolumeStats& dlt_stats)
{
    fs::create_directories(kOutput);
    if (!writeRaw(kDltOutput, dlt)) return false;
    // Both rows use one window; independent contrast stretching obscures
    // differences in magnitude between the two reconstruction operators.
    const float common_scale = std::min(mechanical_display_scale, displayScale(dlt));
    const auto selected_slices = selectAxialSlices(
        mechanical_scores, axialScores(dlt));
    std::vector<float> mechanical_slices;
    if (!readSelectedSlices(kMechanicalOutput, selected_slices,
            mechanical_slices)) return false;
    std::vector<YK::TestImage::GrayPanel> panels;
    for (int z = 0; z < 3; ++z)
        panels.push_back({&mechanical_slices, kNx, kNy, 3, z,
            common_scale, 0.f, 1.f});
    for (int z : selected_slices)
        panels.push_back({&dlt, kNx, kNy, kNz, z, common_scale, 0.f, 1.f});
    if (!YK::TestImage::writeGrayMontageBmp(
            kOutput / "mechanical-top_dlt-bottom-auto-axial.bmp",
            panels, 3, 6, 1)) return false;
    std::ofstream report(kOutput / "comparison.txt");
    report << "input_views=360\ninput_detector=1024x1024\ninput_binning=none\n"
        << "input_frame_skipping=none\nvolume=768x768x600\n"
        << "projection_storage=mapped_pinned_host\n"
        << "voxel_mm=0.30,0.30,0.30\nvolume_center_mm=0,0,0\n"
        << "volume_extent_mm=230.4,230.4,180.0\n"
        << "auto_axial_slice_indices=" << selected_slices[0] << ","
        << selected_slices[1] << "," << selected_slices[2] << "\n"
        << "auto_axial_slice_z_mm=" << (selected_slices[0] - 299.5) * 0.3 << ","
        << (selected_slices[1] - 299.5) * 0.3 << ","
        << (selected_slices[2] - 299.5) * 0.3 << "\n"
        << "algorithm=SART\niterations=" << kIterations
        << "\nrelaxation=" << kRelaxation << "\n"
        << "phantom_pose_used=false\n"
        << "mechanical_seconds=" << mechanical_seconds << "\n"
        << "dlt_seconds=" << dlt_seconds << "\n"
        << "mechanical_min=" << mechanical_stats.minimum << "\n"
        << "mechanical_max=" << mechanical_stats.maximum << "\n"
        << "mechanical_mean=" << mechanical_stats.mean << "\n"
        << "mechanical_l2=" << mechanical_stats.l2 << "\n"
        << "dlt_min=" << dlt_stats.minimum << "\n"
        << "dlt_max=" << dlt_stats.maximum << "\n"
        << "dlt_mean=" << dlt_stats.mean << "\n"
        << "dlt_l2=" << dlt_stats.l2 << "\n"
        << "center_z300_384x384_correlation=" <<
            centerSliceCorrelation(mechanical_center_slice, dlt) << "\n"
        << "shared_visual_window_max=" << 1.f / common_scale << "\n"
        << "mechanical_operator=Joseph_FP+Joseph_v3_BP\n"
        << "mechanical_normalization=tigre_approx\n"
        << "dlt_operator=Siddon_FP+Siddon_voxel_v2_BP\n"
        << "dlt_iterative_backend=AlgebraicReconstructorEx_adapter\n"
        << "dlt_normalization=tigre_approx\n"
        << "dlt_coordinate_normalization=source_trajectory_only\n";
    return static_cast<bool>(report);
}

} // namespace

int main_yang_sart_full_resolution_comparison()
{
    if (!cudaOk(cudaSetDeviceFlags(cudaDeviceMapHost), "enable mapped host memory"))
        return 1;
    MappedProjection projection;
    std::vector<YK::DltFpBp::SDltProjectionMatrix> matrices;
    if (!projection.load() || !readDltMatrices(matrices)) {
        std::fprintf(stderr, "Missing comparison input: %s or %s\n",
            kProjectionFile.string().c_str(), kMatrixFile.string().c_str());
        return 1;
    }
    const auto params = makeMechanicalParameters();
    const auto volume_geometry = makeVolumeGeometry();
    std::vector<YK::DltFpBp::SDltRayGeometry> dlt_geometry;
    if (!YK::DltFpBp::buildRayGeometry(matrices, kChannels, kRows,
            volume_geometry.center, dlt_geometry)) return 1;
    std::vector<YK::SConeProjGeomVec> mechanical_geometry;
    YK::detail::buildCircularViews(params, mechanical_geometry);
    if (mechanical_geometry.size() != kViews) return 1;

    cudaStream_t stream = nullptr;
    if (!cudaOk(cudaStreamCreate(&stream), "create stream")) return 1;
    YK::Mem::MemoryController memory;
    auto d_volume = memory.allocateDevice3D<float>(kNx, kNy, kNz, 0, false);
    const size_t volume_count = static_cast<size_t>(kNx) * kNy * kNz;
    bool ok = d_volume &&
        cudaOk(cudaMemsetAsync(d_volume.data(), 0, volume_count * sizeof(float), stream),
            "clear mechanical volume") &&
        cudaOk(cudaStreamSynchronize(stream), "finish input upload");
    std::vector<float> host_volume(volume_count);
    VolumeStats mechanical_stats{};
    std::vector<double> mechanical_scores;
    std::vector<float> mechanical_center_slice;
    float mechanical_display_scale = 1.f;

    double mechanical_seconds = 0.0;
    if (ok) {
        YK::Iter::AlgebraicReconstructionConfig config{};
        config.method = YK::Iter::EAlgebraicMethod::Sart;
        config.weight_model = YK::Iter::EAlgebraicWeightModel::TigreApprox;
        config.iterations = kIterations;
        config.subset_count = kViews;
        config.relaxation = kRelaxation;
        config.use_min = true;
        config.min_constraint = 0.f;
        const auto begin = Clock::now();
        YK::Iter::AlgebraicReconstructorEx reconstructor;
        ok = reconstructor.prepare(params, mechanical_geometry, config, stream) &&
            reconstructor.reconstruct(projection.device, d_volume.data()) &&
            cudaOk(cudaStreamSynchronize(stream), "mechanical SART");
        mechanical_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        ok = ok && cudaOk(cudaMemcpy(host_volume.data(), d_volume.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost),
            "download mechanical volume");
        if (ok) {
            mechanical_stats = statistics(host_volume);
            mechanical_scores = axialScores(host_volume);
            mechanical_display_scale = displayScale(host_volume);
            const size_t slice_count = static_cast<size_t>(kNx) * kNy;
            mechanical_center_slice.assign(
                host_volume.begin() + (kNz / 2) * slice_count,
                host_volume.begin() + (kNz / 2 + 1) * slice_count);
            ok = writeRaw(kMechanicalOutput, host_volume);
        }
    }

    double dlt_seconds = 0.0;
    if (ok) {
        ok = cudaOk(cudaMemsetAsync(d_volume.data(), 0, volume_count * sizeof(float), stream),
            "clear DLT volume") && cudaOk(cudaStreamSynchronize(stream), "finish DLT clear");
        YK::DltFpBp::SDltAlgebraicConfig config{};
        config.method = YK::DltFpBp::EDltAlgebraicMethod::Sart;
        config.weight_model =
            YK::DltFpBp::EDltAlgebraicWeightModel::TigreApprox;
        config.relaxation = kRelaxation;
        config.nonnegative = true;
        const auto begin = Clock::now();
        YK::DltFpBp::DltAlgebraicReconstructor reconstructor;
        ok = ok && reconstructor.prepare(dlt_geometry, volume_geometry,
                kChannels, kRows, config, stream) &&
            reconstructor.iterate(projection.device, d_volume.data(), kIterations, stream) &&
            cudaOk(cudaStreamSynchronize(stream), "DLT SART");
        dlt_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        ok = ok && cudaOk(cudaMemcpy(host_volume.data(), d_volume.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost),
            "download DLT volume");
    }
    if (stream) cudaStreamDestroy(stream);
    if (!ok) return 1;

    const auto dlt_stats = statistics(host_volume);
    ok = std::isfinite(mechanical_stats.maximum) && mechanical_stats.maximum > 0.0 &&
        std::isfinite(dlt_stats.maximum) && dlt_stats.maximum > 0.0 &&
        saveComparison(host_volume, mechanical_scores, mechanical_center_slice,
            mechanical_display_scale, mechanical_seconds, dlt_seconds,
            mechanical_stats, dlt_stats);
    std::printf("Full-resolution Yang SART: mechanical %.3f s, DLT %.3f s, "
        "max %.6g / %.6g, output=%s\n", mechanical_seconds, dlt_seconds,
        mechanical_stats.maximum, dlt_stats.maximum, kOutput.string().c_str());
    return ok ? 0 : 1;
}
