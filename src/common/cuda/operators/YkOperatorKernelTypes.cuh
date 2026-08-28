#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"

namespace YK::CudaOp {

// Kernel 层统一使用语义化写入模式，避免不同 launch 对 bool accumulate
// 作出相反解释。兼容入口可以用 writeMode(bool) 完成一次转换。
enum class EWriteMode {
    Overwrite,
    Accumulate
};

inline EWriteMode writeMode(bool accumulate)
{
    return accumulate ? EWriteMode::Accumulate : EWriteMode::Overwrite;
}

YK_HD inline bool accumulates(EWriteMode mode)
{
    return mode == EWriteMode::Accumulate;
}

inline void clearIfOverwrite(float* output, size_t count, EWriteMode mode,
    cudaStream_t stream)
{
    if (mode == EWriteMode::Overwrite)
        YK_CUDA_CHECK(cudaMemsetAsync(output, 0, count * sizeof(float), stream));
}

struct SProjectionShape {
    int channels = 0;
    int rows = 0;
    int views = 0;

    size_t elements() const
    {
        return static_cast<size_t>(channels) * rows * views;
    }
};

struct SViewRange {
    int first = 0;
    int count = 0;

    int end() const { return first + count; }
};

enum class EMainAxis { X, Y, Z };

inline EMainAxis mainAxis(const float4& direction)
{
    const float ax = fabsf(direction.x);
    const float ay = fabsf(direction.y);
    const float az = fabsf(direction.z);
    if (ax >= ay && ax >= az) return EMainAxis::X;
    if (ay >= ax && ay >= az) return EMainAxis::Y;
    return EMainAxis::Z;
}

inline EMainAxis mainAxis(const float3& direction)
{
    return mainAxis(make_float4(direction.x, direction.y, direction.z, 0.f));
}

inline SConeProjGeomVec normalizeToVoxel(const SConeProjGeomVec& view,
    const SVolGeom& volume)
{
    const float ix = 1.f / volume.vox_x;
    const float iy = 1.f / volume.vox_y;
    const float iz = 1.f / volume.vox_z;
    SConeProjGeomVec result = view;
    result.src.x = (view.src.x - volume.center.x) * ix;
    result.src.y = (view.src.y - volume.center.y) * iy;
    result.src.z = (view.src.z - volume.center.z) * iz;
    result.detS.x = (view.detS.x - volume.center.x) * ix;
    result.detS.y = (view.detS.y - volume.center.y) * iy;
    result.detS.z = (view.detS.z - volume.center.z) * iz;
    result.detU.x = view.detU.x * ix;
    result.detU.y = view.detU.y * iy;
    result.detU.z = view.detU.z * iz;
    result.detV.x = view.detV.x * ix;
    result.detV.y = view.detV.y * iy;
    result.detV.z = view.detV.z * iz;
    return result;
}

inline std::vector<SConeProjGeomVec> normalizeToVoxelBatch(
    const std::vector<SConeProjGeomVec>& views, const SVolGeom& volume)
{
    std::vector<SConeProjGeomVec> result;
    result.reserve(views.size());
    for (const auto& view : views) result.push_back(normalizeToVoxel(view, volume));
    return result;
}

template<typename Callback>
bool forEachAxisRun(const std::vector<SConeProjGeomVec>& views, int count,
    int detectorPixelsU, int detectorPixelsV, Callback&& callback)
{
    if (count < 0 || detectorPixelsU <= 0 || detectorPixelsV <= 0 ||
        static_cast<size_t>(count) > views.size())
        return false;

    // 先验证全部视图，再启动任何 kernel。否则后面的退化 geometry 会让调用方
    // 得到只计算了一部分角度的结果，而且这种错误很难从 CUDA 状态中发现。
    std::vector<EMainAxis> axes(static_cast<size_t>(count));
    for (int index = 0; index < count; ++index) {
        SProjectionFrame frame{};
        if (!deriveProjectionFrame(views[index], detectorPixelsU,
            detectorPixelsV, frame))
            return false;
        axes[static_cast<size_t>(index)] = mainAxis(frame.centerRay);
    }

    int first = 0;
    while (first < count) {
        const EMainAxis axis = axes[static_cast<size_t>(first)];
        int end = first + 1;
        while (end < count) {
            if (axes[static_cast<size_t>(end)] != axis) break;
            ++end;
        }
        callback(axis, SViewRange{ first, end - first });
        first = end;
    }
    return true;
}

// Flat Joseph FP/BP 共用的坐标排列策略。把方向相关代码集中后，算法文件
// 只负责积分或散射，不再各自维护一套 X/Y/Z 定义。
struct AxisX {
    YK_HD static float c0(float x, float, float) { return x; }
    YK_HD static float c1(float, float y, float) { return y; }
    YK_HD static float c2(float, float, float z) { return z; }
    YK_HD static int nSlices(int nx, int, int) { return nx; }
    YK_HD static int nDim1(int, int ny, int) { return ny; }
    YK_HD static int nDim2(int, int, int nz) { return nz; }
    YK_HD static float vox0(float vx, float, float) { return vx; }
    YK_HD static float vox1(float, float vy, float) { return vy; }
    YK_HD static float vox2(float, float, float vz) { return vz; }
    YK_HD static float voxSize(const SVolGeom& g) { return g.vox_x; }
#ifdef __CUDACC__
    __device__ static float sample(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return tex3D<float>(texture, f0, f1, f2); }
    __device__ static float sampleVol(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return sample(texture, f0, f1, f2); }
#endif
    YK_HD static int3 toVoxel(int s, int d1, int d2)
    { return make_int3(s, d1, d2); }
};

struct AxisY {
    YK_HD static float c0(float, float y, float) { return y; }
    YK_HD static float c1(float x, float, float) { return x; }
    YK_HD static float c2(float, float, float z) { return z; }
    YK_HD static int nSlices(int, int ny, int) { return ny; }
    YK_HD static int nDim1(int nx, int, int) { return nx; }
    YK_HD static int nDim2(int, int, int nz) { return nz; }
    YK_HD static float vox0(float, float vy, float) { return vy; }
    YK_HD static float vox1(float vx, float, float) { return vx; }
    YK_HD static float vox2(float, float, float vz) { return vz; }
    YK_HD static float voxSize(const SVolGeom& g) { return g.vox_y; }
#ifdef __CUDACC__
    __device__ static float sample(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return tex3D<float>(texture, f1, f0, f2); }
    __device__ static float sampleVol(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return sample(texture, f0, f1, f2); }
#endif
    YK_HD static int3 toVoxel(int s, int d1, int d2)
    { return make_int3(d1, s, d2); }
};

struct AxisZ {
    YK_HD static float c0(float, float, float z) { return z; }
    YK_HD static float c1(float x, float, float) { return x; }
    YK_HD static float c2(float, float y, float) { return y; }
    YK_HD static int nSlices(int, int, int nz) { return nz; }
    YK_HD static int nDim1(int nx, int, int) { return nx; }
    YK_HD static int nDim2(int, int ny, int) { return ny; }
    YK_HD static float vox0(float, float, float vz) { return vz; }
    YK_HD static float vox1(float vx, float, float) { return vx; }
    YK_HD static float vox2(float, float vy, float) { return vy; }
    YK_HD static float voxSize(const SVolGeom& g) { return g.vox_z; }
#ifdef __CUDACC__
    __device__ static float sample(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return tex3D<float>(texture, f1, f2, f0); }
    __device__ static float sampleVol(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return sample(texture, f0, f1, f2); }
#endif
    YK_HD static int3 toVoxel(int s, int d1, int d2)
    { return make_int3(d1, d2, s); }
};

} // namespace YK::CudaOp
