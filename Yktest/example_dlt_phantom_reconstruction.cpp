#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "DLTFpBp/YkDltAlgebraicReconstructor.hpp"
#include "DLTFpBp/YkDltSiddonProjector.hpp"
#include "YkTestImage.hpp"

namespace {

namespace fs = std::filesystem;
using YK::DltFpBp::SDltProjectionMatrix;

// This integration sample can be selected by name from the IDE test runner.
// No physical SID/SDD/detector offsets are supplied to the reconstruction.
constexpr int kOriginalChannels = 1024;
constexpr int kOriginalRows = 1024;
constexpr int kOriginalViews = 360;
constexpr int kDetectorBin = 4;
constexpr int kViewStride = 2;
constexpr int kChannels = kOriginalChannels / kDetectorBin;
constexpr int kRows = kOriginalRows / kDetectorBin;
constexpr int kViews = kOriginalViews / kViewStride;
constexpr int kGrid = 128;
constexpr int kIterations = 8;

const fs::path kRoot = fs::path(YKCBCT_TEST_SOURCE_DIR);
const fs::path kProjectionFile = kRoot /
    "example/multispectrum_sim/outputs/quality/"
    "double-ring-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw";
const fs::path kMatrixFile = kRoot /
    "out/cbct_calibration/cho_reference/double_ring_px10_py15_pz20/"
    "projection_matrices_360x3x4.f32";
const fs::path kOutput = kRoot /
    "out/cbct_calibration/dlt_phantom_reconstruction";

bool readMatrices(std::vector<SDltProjectionMatrix>& matrices)
{
    if (!fs::exists(kMatrixFile) ||
        fs::file_size(kMatrixFile) != kOriginalViews * 12 * sizeof(float))
        return false;
    std::ifstream input(kMatrixFile, std::ios::binary);
    if (!input) return false;
    std::vector<float> raw(kOriginalViews * 12);
    input.read(reinterpret_cast<char*>(raw.data()),
        static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!input) return false;
    matrices.reserve(kViews);
    for (int view = 0; view < kOriginalViews; view += kViewStride) {
        SDltProjectionMatrix matrix{};
        for (int i = 0; i < 12; ++i)
            matrix.value[i] = raw[static_cast<size_t>(view) * 12 + i];
        // The binned pixel at u' is centered at original u=4*u'+1.5.
        // P' = H P, with u'=(u-1.5)/4 and v'=(v-1.5)/4.
        const double block_center = (kDetectorBin - 1) * 0.5;
        for (int axis = 0; axis < 4; ++axis) {
            matrix(0, axis) = (matrix(0, axis) - block_center * matrix(2, axis)) /
                kDetectorBin;
            matrix(1, axis) = (matrix(1, axis) - block_center * matrix(2, axis)) /
                kDetectorBin;
        }
        matrices.push_back(matrix);
    }
    return true;
}

bool readBinnedProjections(std::vector<float>& projections)
{
    if (!fs::exists(kProjectionFile) || fs::file_size(kProjectionFile) !=
        static_cast<uintmax_t>(kOriginalViews) * kOriginalRows *
            kOriginalChannels * sizeof(float)) return false;
    std::ifstream input(kProjectionFile, std::ios::binary);
    if (!input) return false;
    const size_t original_view_size = static_cast<size_t>(kOriginalRows) *
        kOriginalChannels;
    std::vector<float> source(original_view_size);
    projections.resize(static_cast<size_t>(kViews) * kRows * kChannels);
    for (int view = 0; view < kViews; ++view) {
        input.seekg(static_cast<std::streamoff>(view * kViewStride) *
            original_view_size * sizeof(float));
        input.read(reinterpret_cast<char*>(source.data()),
            static_cast<std::streamsize>(original_view_size * sizeof(float)));
        if (!input) return false;
        for (int row = 0; row < kRows; ++row) {
            for (int channel = 0; channel < kChannels; ++channel) {
                double sum = 0.0;
                for (int dv = 0; dv < kDetectorBin; ++dv)
                    for (int du = 0; du < kDetectorBin; ++du)
                        sum += source[static_cast<size_t>(row * kDetectorBin + dv) *
                            kOriginalChannels + channel * kDetectorBin + du];
                projections[(static_cast<size_t>(view) * kRows + row) *
                    kChannels + channel] = static_cast<float>(sum /
                        (kDetectorBin * kDetectorBin));
            }
        }
    }
    return true;
}

bool saveVolumeAndSlices(const std::vector<float>& volume)
{
    fs::create_directories(kOutput);
    const auto raw_file = kOutput / "dlt-ossart-128x128x128-1mm-f32.raw";
    std::ofstream raw(raw_file, std::ios::binary);
    if (!raw) return false;
    raw.write(reinterpret_cast<const char*>(volume.data()),
        static_cast<std::streamsize>(volume.size() * sizeof(float)));
    if (!raw) return false;

    // The two axial planes pass through the two bead rings.  The third panel
    // is an XZ maximum-intensity projection so both rings are visible together.
    std::vector<float> xz(static_cast<size_t>(kGrid) * kGrid, 0.f);
    for (int z = 0; z < kGrid; ++z)
        for (int x = 0; x < kGrid; ++x)
            for (int y = 0; y < kGrid; ++y)
                xz[static_cast<size_t>(z) * kGrid + x] = std::max(
                    xz[static_cast<size_t>(z) * kGrid + x],
                    volume[(static_cast<size_t>(z) * kGrid + y) * kGrid + x]);
    std::vector<float> xy_mip(static_cast<size_t>(kGrid) * kGrid, 0.f);
    for (int z = 0; z < kGrid; ++z)
        for (int y = 0; y < kGrid; ++y)
            for (int x = 0; x < kGrid; ++x)
                xy_mip[static_cast<size_t>(y) * kGrid + x] = std::max(
                    xy_mip[static_cast<size_t>(y) * kGrid + x],
                    volume[(static_cast<size_t>(z) * kGrid + y) * kGrid + x]);

    const float maximum = *std::max_element(volume.begin(), volume.end());
    const float window_max = std::max(1.e-6f, maximum * 0.65f);
    std::vector<YK::TestImage::GrayPanel> panels{
        {&volume, kGrid, kGrid, kGrid, 14, 1.f, 0.f, window_max},
        {&volume, kGrid, kGrid, kGrid, 114, 1.f, 0.f, window_max},
        {&xz, kGrid, kGrid, 1, 0, 1.f, 0.f, window_max},
        {&xy_mip, kGrid, kGrid, 1, 0, 1.f, 0.f, window_max}
    };
    return YK::TestImage::writeGrayMontageBmp(
        kOutput / "dlt-ossart-slices.bmp", panels, 2, 6, 3);
}

} // namespace

int main_dlt_phantom_reconstruction()
{
    std::vector<SDltProjectionMatrix> matrices;
    std::vector<float> measured;
    if (!readMatrices(matrices) || !readBinnedProjections(measured)) {
        std::fprintf(stderr, "DLT phantom input missing or invalid: %s / %s\n",
            kMatrixFile.string().c_str(), kProjectionFile.string().c_str());
        return 1;
    }

    auto grid = YK::SVolGeom::make_centered(kGrid, kGrid, kGrid, 1.f);
    std::vector<YK::DltFpBp::SDltRayGeometry> geometry;
    if (!YK::DltFpBp::buildRayGeometry(matrices, kChannels, kRows,
            grid.center, geometry)) return 1;

    YK::DltFpBp::SDltAlgebraicConfig config{};
    config.method = YK::DltFpBp::EDltAlgebraicMethod::Ossart;
    config.ossart_subset_size = 9;
    config.relaxation = 0.8f;
    YK::DltFpBp::DltAlgebraicReconstructor reconstructor;
    if (!reconstructor.prepare(geometry, grid, kChannels, kRows, config))
        return 1;

    YK::Mem::MemoryController memory;
    auto d_projection = memory.allocateDevice3D<float>(
        kChannels, kRows, kViews, 0, false);
    auto d_volume = memory.allocateDevice3D<float>(kGrid, kGrid, kGrid, 0, false);
    auto d_check = memory.allocateDevice3D<float>(
        kChannels, kRows, kViews, 0, false);
    const size_t volume_count = static_cast<size_t>(kGrid) * kGrid * kGrid;
    if (cudaMemset(d_volume.data(), 0, volume_count * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(d_projection.data(), measured.data(),
            measured.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        !reconstructor.iterate(d_projection.data(), d_volume.data(), kIterations) ||
        cudaDeviceSynchronize() != cudaSuccess) return 1;

    std::vector<float> volume(volume_count);
    if (cudaMemcpy(volume.data(), d_volume.data(), volume.size() * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        !saveVolumeAndSlices(volume)) return 1;

    YK::DltFpBp::DltSiddonProjector projector;
    if (!projector.prepare(geometry, grid, kChannels, kRows) ||
        !projector.forward(d_volume.data(), d_check.data(), false, nullptr) ||
        cudaDeviceSynchronize() != cudaSuccess) return 1;
    std::vector<float> predicted(measured.size());
    if (cudaMemcpy(predicted.data(), d_check.data(),
            predicted.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 1;
    double residual_squared = 0.0, measured_squared = 0.0;
    for (size_t i = 0; i < measured.size(); ++i) {
        const double difference = measured[i] - predicted[i];
        residual_squared += difference * difference;
        measured_squared += static_cast<double>(measured[i]) * measured[i];
    }
    const float maximum = *std::max_element(volume.begin(), volume.end());
    std::printf("DLT-OS-SART views=%d subset_size=%d subsets=%d detector=%dx%d "
        "volume=%d^3 sweeps=%d "
        "max=%.6g residual=%.5f output=%s\n", kViews,
        reconstructor.subsetSize(), reconstructor.subsetCount(), kChannels,
        kRows, kGrid,
        kIterations, maximum,
        std::sqrt(residual_squared / measured_squared), kOutput.string().c_str());
    return maximum > 0.f && std::isfinite(maximum) ? 0 : 1;
}
