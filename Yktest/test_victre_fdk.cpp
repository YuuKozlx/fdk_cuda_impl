#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "FDK/YkFdkPipeline.hpp"
#include "YkTestImage.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;
namespace fs = std::filesystem;

constexpr int kDetectorU = 1024;
constexpr int kDetectorV = 1024;
constexpr int kViews = 360;
constexpr int kBatchViews = 8;

const fs::path kExampleDirectory = fs::path(YKCBCT_TEST_SOURCE_DIR) / "example";
const fs::path kProjectionPath = kExampleDirectory /
    "victre_breast_cbct_x360_1024_frame0_1024x1024pixels_360proj.raw";
const fs::path kAirPath = kExampleDirectory /
    "victre_breast_cbct_x360_1024_air_frame0_1024x1024pixels_1proj.raw";
const fs::path kGeometryPath = kExampleDirectory /
    "victre_breast_cbct_x360_1024_geometry.txt";

bool hasExpectedBytes(const fs::path& path, std::uintmax_t expected)
{
    std::error_code error;
    const auto bytes = fs::file_size(path, error);
    if (error || bytes != expected) {
        YK_LOGE("文件尺寸不正确：{}，期望 {} bytes，实际 {} bytes",
            path.string(), expected, error ? 0 : bytes);
        return false;
    }
    return true;
}

bool readFloats(std::ifstream& input, float* destination, size_t count)
{
    input.read(reinterpret_cast<char*>(destination),
        static_cast<std::streamsize>(count * sizeof(float)));
    return input.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

SCBCTParams makeVictreParams()
{
    SCBCTParams p{};
    p.iPU = kDetectorU;
    p.iPV = kDetectorV;
    p.iPAng = kViews;
    p.iPAngTotal = kViews;
    p.du_mm = 0.2490234375f;
    p.dv_mm = 0.2490234375f;
    p.SID = 600.f;
    p.SDD = 1000.f;
    p.bShortScan = false;
    p.scan_start_angle_rad = 0.f;
    p.scan_range_rad = 2.f * CUDA_PI;
    p.nDirSign = 1;

    // 原数据绕世界 X 轴旋转。固定变换
    //   internal(x,y,z) = (-worldY, -worldZ, worldX)
    // 将其精确映射为库内绕 Z 轴的圆轨迹：0 度时源在
    // (0,-SID,0)，探测器 U/V 分别沿 +X/+Z。因此不需要翻转投影像素。
    // 体模原始尺寸 64 x 97.5 x 47 mm (X,Y,Z)，映射后为
    // 97.5 x 47 x 64 mm。使用 0.25 mm 等方体素覆盖完整乳腺。
    p.iVX = 390;
    p.iVY = 188;
    p.iVZ = 256;
    p.vox_x_mm = 0.25f;
    p.vox_y_mm = 0.25f;
    p.vox_z_mm = 0.25f;
    p.desc = SFilterKernelDesc::RamLak(EWeightsBuildSource::DiscreteRLFFT, 1.f);

    p.angle_list.resize(kViews);
    for (int i = 0; i < kViews; ++i)
        p.angle_list[i] = static_cast<float>(i) * CUDA_PI / 180.f;
    return p;
}

bool applyAirCorrection(const std::vector<float>& air,
    float* batch, size_t view_elements, int views,
    double& minimum, double& maximum, double& sum)
{
    constexpr float kMinimumRatio = 1.0e-6f;
    for (int view = 0; view < views; ++view) {
        float* projection = batch + static_cast<size_t>(view) * view_elements;
        for (size_t pixel = 0; pixel < view_elements; ++pixel) {
            const float flat = air[pixel];
            const float measured = projection[pixel];
            if (!(flat > 0.f) || !std::isfinite(flat) ||
                !(measured >= 0.f) || !std::isfinite(measured)) {
                YK_LOGE("空气校正遇到非法像素：view={}, pixel={}, I={}, I0={}",
                    view, pixel, measured, flat);
                return false;
            }
            // MC-GPU total 图是能量积分强度。重建前转换为线积分。
            // 统计噪声可能使 I 略大于 I0，此时按零衰减处理。
            const float ratio = std::clamp(measured / flat, kMinimumRatio, 1.f);
            const float line_integral = -std::log(ratio);
            projection[pixel] = line_integral;
            minimum = std::min(minimum, static_cast<double>(line_integral));
            maximum = std::max(maximum, static_cast<double>(line_integral));
            sum += line_integral;
        }
    }
    return true;
}

bool writeRaw(const fs::path& path, const std::vector<float>& values)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (error) return false;
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

} // namespace

// 使用 example 中的 VICTRE 乳腺 CBCT 数据做真实数据 FDK 闭环。
// 测试不把 1.41 GiB 投影全部读入内存，而是按 batch 顺序读取、
// 空气校正后立即提交给在线 FDK pipeline。
int main_fdk_victre_breast()
{
    const size_t view_elements = static_cast<size_t>(kDetectorU) * kDetectorV;
    const std::uintmax_t view_bytes = view_elements * sizeof(float);
    std::error_code geometry_error;
    if (!fs::is_regular_file(kGeometryPath, geometry_error) ||
        !hasExpectedBytes(kAirPath, view_bytes) ||
        !hasExpectedBytes(kProjectionPath, view_bytes * kViews))
        return 1;

    std::ifstream air_input(kAirPath, std::ios::binary);
    std::ifstream projection_input(kProjectionPath, std::ios::binary);
    std::vector<float> air(view_elements);
    if (!air_input || !projection_input ||
        !readFloats(air_input, air.data(), air.size())) {
        YK_LOGE("无法读取 VICTRE 空气场或投影文件");
        return 1;
    }

    const SCBCTParams params = makeVictreParams();
    const size_t volume_elements = static_cast<size_t>(params.iVX) *
        params.iVY * params.iVZ;
    Mem::MemoryController memory;
    std::array<Mem::HostPinnedBuffer3D<float>, 2> projection_batches{
        memory.allocatePinnedCpu3D<float>(kDetectorU, kDetectorV, kBatchViews),
        memory.allocatePinnedCpu3D<float>(kDetectorU, kDetectorV, kBatchViews)
    };
    std::array<FdkBatchFence, 2> batch_fences;
    auto device_volume = memory.allocateDevice3D<float>(
        params.iVX, params.iVY, params.iVZ, 0, false);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        YK_LOGE("创建 VICTRE FDK CUDA stream 失败");
        return 1;
    }

    FdkPipeline pipeline;
    bool ok = pipeline.prepareWithAngles(params, params.angle_list,
        kBatchViews, stream);
    double projection_minimum = std::numeric_limits<double>::infinity();
    double projection_maximum = -std::numeric_limits<double>::infinity();
    double projection_sum = 0.0;
    const auto start = std::chrono::steady_clock::now();

    for (int base = 0; ok && base < kViews; base += kBatchViews) {
        const int batch_index = base / kBatchViews;
        const size_t slot = static_cast<size_t>(batch_index % projection_batches.size());
        // 双 pinned buffer 轮换：GPU 处理 slot 0 时，CPU 可读取/校正 slot 1。
        // 只在第三批开始复用同一 slot 时等待其 fence，不阻塞另一块缓冲。
        if (!batch_fences[slot].wait()) {
            ok = false;
            break;
        }
        const int count = std::min(kBatchViews, kViews - base);
        const size_t elements = view_elements * static_cast<size_t>(count);
        auto& projection_batch = projection_batches[slot];
        ok = readFloats(projection_input, projection_batch.data(), elements) &&
            applyAirCorrection(air, projection_batch.data(), view_elements, count,
                projection_minimum, projection_maximum, projection_sum);
        if (!ok) {
            YK_LOGE("读取或校正 VICTRE 投影失败：base={}, count={}", base, count);
            break;
        }
        const FdkProjectionBatch batch{
            projection_batch.cdata(), nullptr, nullptr, count };
        ok = pipeline.enqueueBatch(batch, device_volume.data(), base == 0,
            &batch_fences[slot]);
        if (!ok) YK_LOGE("VICTRE FDK 分包失败：base={}, count={}", base, count);
    }
    ok = ok && pipeline.complete();
    // 两个 fence 分别证明两块主机缓冲已不再被 CUDA 使用。
    // 由于所有批次位于同一 stream，等待最后一个事件也保证整体完成；
    // 两个都 wait 使每块输入的生命期在测试中显式闭环。
    for (const auto& fence : batch_fences) ok = fence.wait() && ok;
    const auto stop = std::chrono::steady_clock::now();

    std::vector<float> reconstruction(volume_elements);
    if (ok) {
        const cudaError_t status = cudaMemcpy(reconstruction.data(), device_volume.data(),
            volume_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            YK_LOGE("VICTRE FDK 体数据下载失败：{}", cudaGetErrorString(status));
            ok = false;
        }
    }

    double volume_minimum = std::numeric_limits<double>::infinity();
    double volume_maximum = -std::numeric_limits<double>::infinity();
    double volume_sum = 0.0;
    size_t invalid_values = 0;
    for (float value : reconstruction) {
        if (!std::isfinite(value)) {
            ++invalid_values;
            continue;
        }
        volume_minimum = std::min(volume_minimum, static_cast<double>(value));
        volume_maximum = std::max(volume_maximum, static_cast<double>(value));
        volume_sum += value;
    }
    ok = ok && invalid_values == 0 && volume_maximum > volume_minimum &&
        std::max(std::fabs(volume_minimum), std::fabs(volume_maximum)) > 1.0e-8;

    const fs::path artifact_directory = fs::path(YKCBCT_TEST_SOURCE_DIR) /
        "out" / "test-artifacts" / "victre-breast-fdk";
    const fs::path raw_path = artifact_directory / "reconstruction_390x188x256_f32.raw";
    const fs::path image_path = artifact_directory / "central_world_x_slice.bmp";
    const bool raw_written = ok && writeRaw(raw_path, reconstruction);
    const float display_maximum = static_cast<float>(std::max(
        std::fabs(volume_minimum), std::fabs(volume_maximum)));
    const std::vector<TestImage::GrayPanel> panels{
        { &reconstruction, params.iVX, params.iVY, params.iVZ,
          params.iVZ / 2, 1.f, 0.f, display_maximum, false }
    };
    const bool image_written = ok && TestImage::writeGrayMontageBmp(
        image_path, panels, 1, 0, 2);
    ok = ok && raw_written && image_written;

    const double elapsed_seconds = std::chrono::duration<double>(stop - start).count();
    const double projection_count = static_cast<double>(view_elements) * kViews;
    YK_LOGI("VICTRE 线积分：min={:.6e}, max={:.6e}, mean={:.6e}",
        projection_minimum, projection_maximum, projection_sum / projection_count);
    YK_LOGI("VICTRE FDK 体：{}x{}x{}, min={:.6e}, max={:.6e}, mean={:.6e}, "
        "invalid={}, elapsed={:.3f}s", params.iVX, params.iVY, params.iVZ,
        volume_minimum, volume_maximum, volume_sum / volume_elements,
        invalid_values, elapsed_seconds);
    YK_LOGI("VICTRE FDK 输出：{} ; {} ; {}", raw_path.string(),
        image_path.string(), ok ? "PASS" : "FAIL");

    pipeline.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
