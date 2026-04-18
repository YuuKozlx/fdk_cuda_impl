#pragma once
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "YkBuffer.h"

// ============================================================
// [YK] CUDA 内存传输规范
// ============================================================
//
// 一、同步 vs 异步传输
// ─────────────────────────────────────────────────────────────
// 【同步传输】cudaMemcpy / cudaMemcpy3D
//   - host 端内存可以是任意类型（std::vector、裸指针、栈变量）
//   - 函数返回时传输已完成
//   - 适用：一次性初始化、调试 dump、非热路径传输
//
// 【异步传输】cudaMemcpyAsync / cudaMemcpy3DAsync
//   - host 端内存【必须】是 pinned memory（HostPinnedBuffer）
//   - 函数立即返回，传输在 stream 上异步执行
//   - 调用方负责在合适位置 sync(stream)
//   - 适用：热路径、pipeline 中 CPU/GPU 重叠执行
//
// 二、Pinned Memory
// ─────────────────────────────────────────────────────────────
//   分配：cudaMallocHost  释放：cudaFreeHost
//   满足以下三个条件才值得用 pinned：
//     1. 生命周期覆盖整个异步传输过程
//     2. 传输完成前数据不会被修改或释放
//     3. 有实际的异步重叠收益
//
// 三、Stream 规范
// ─────────────────────────────────────────────────────────────
//   - 所有接受 stream 的函数不提供默认值，调用方必须显式传入
//   - 同一 pipeline 内所有操作提交到同一 stream
//   - process() 内部只提交操作，不调用 sync，由外层统一管理
//   - 带 *Sync 后缀的函数内部封装了 sync()
//
// 四、接口命名约定
// ─────────────────────────────────────────────────────────────
//   upload / download         同步，普通 CPU buffer
//   uploadAsync / downloadAsync    异步，必须传 pinned buffer
//   uploadAsyncSync / downloadAsyncSync  异步 + 内部 sync
//   sync(stream)              统一同步点
//
// 五、常见错误
// ─────────────────────────────────────────────────────────────
//   [错误] std::vector + cudaMemcpyAsync  → UB
//   [错误] 局部变量 + cudaMemcpyToSymbolAsync → DMA 读野指针
//   [错误] host 端直接赋值 __constant__ 变量 → device 不更新
//   [错误] process() 内部 cudaStreamSynchronize → 破坏流水线
// ============================================================

namespace YK {
namespace Mem {

class MemoryController {
public:

    // ----------------------------------------------------------------
    // 同步点
    // ----------------------------------------------------------------
    void sync(cudaStream_t stream) const
    {
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ================================================================
    // CPU Buffer
    // ================================================================

    template<typename T>
    CpuBuffer3D<T> allocateCpu3D(int nx, int ny, int nz, bool zero = false) const
    {
        return CpuBuffer3D<T>(nx, ny, nz, zero);
    }

    // ================================================================
    // Pinned CPU Buffer
    // ================================================================

    template<typename T>
    HostPinnedBuffer3D<T> allocatePinnedCpu3D(int nx, int ny, int nz) const
    {
        HostPinnedBuffer3D<T> buf;
        buf.alloc(nx, ny, nz);
        return buf;
    }

    // ================================================================
    // DeviceLinearBuffer3D（vol buffer，无 padding）
    // 底层 cudaMalloc，pitch = nx*sizeof(T)
    // ================================================================

    template<typename T>
    DeviceLinearBuffer3D<T> allocateDevice3D(
        int nx, int ny, int nz,
        int deviceId,
        bool zero = false) const
    {
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::invalid_argument("allocateDevice3D: nx/ny/nz > 0");

        YK_CUDA_CHECK(cudaSetDevice(deviceId));
        T* ptr = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&ptr, size_t(nx) * ny * nz * sizeof(T)));

        if (zero)
            YK_CUDA_CHECK(cudaMemset(ptr, 0, size_t(nx) * ny * nz * sizeof(T)));

        return DeviceLinearBuffer3D<T>::make_owning(ptr, { nx, ny, nz }, deviceId);
    }

    // ================================================================
    // DevicePitchedBuffer3D（投影纹理 buffer，行对齐）
    // 底层 cudaMalloc3D，pitch 由驱动决定
    // ================================================================

    template<typename T>
    DevicePitchedBuffer3D<T> allocateDevice3DPitched(
        int nx, int ny, int nz,
        int deviceId,
        bool zero = false) const
    {
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::invalid_argument("allocateDevice3DPitched: nx/ny/nz > 0");

        YK_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaExtent extent = make_cudaExtent(nx * sizeof(T), ny, nz);
        cudaPitchedPtr dptr;
        YK_CUDA_CHECK(cudaMalloc3D(&dptr, extent));

        if (zero)
            YK_CUDA_CHECK(cudaMemset3D(dptr, 0, extent));

        return DevicePitchedBuffer3D<T>::make_owning(
            (T*)dptr.ptr, { nx, ny, nz }, dptr.pitch, deviceId);
    }

    // ================================================================
    // Borrow（非拥有，指向外部内存）
    // ================================================================

    template<typename T>
    DeviceLinearBuffer3DBorrowed<T> borrowDevice3D(
        T* ptr, int nx, int ny, int nz, int deviceId = 0) const
    {
        return DeviceLinearBuffer3DBorrowed<T>(ptr, nx, ny, nz, deviceId);
    }

    template<typename T>
    DevicePitchedBuffer3DBorrowed<T> borrowDevice3DPitched(
        T* ptr, int nx, int ny, int nz,
        size_t pitchBytes, int deviceId = 0) const
    {
        return DevicePitchedBuffer3DBorrowed<T>(ptr, nx, ny, nz, pitchBytes, deviceId);
    }

    template<typename T>
    CpuBuffer3DBorrowed<T> borrowCpu3D(T* ptr, int nx, int ny, int nz) const
    {
        return CpuBuffer3DBorrowed<T>(ptr, nx, ny, nz);
    }

    // ================================================================
    // Upload 同步（普通 CPU buffer → Device，无需 stream）
    // ================================================================

    // CpuBuffer3D → DeviceLinearBuffer3D
    template<typename T>
    void upload3D(const DeviceLinearBuffer3D<T>& dst,
                  const CpuBuffer3D<T>& src) const
    {
        if (!dst || !src)
            throw std::invalid_argument("upload3D: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(dst.deviceId()));
        YK_CUDA_CHECK(cudaMemcpy(
            dst.data(), src.cdata(),
            src.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    // CpuBuffer3D → DevicePitchedBuffer3D
    template<typename T>
    void upload3D(const DevicePitchedBuffer3D<T>& dst,
                  const CpuBuffer3D<T>& src) const
    {
        if (!dst || !src)
            throw std::invalid_argument("upload3D: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(dst.deviceId()));
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr(
            (void*)src.cdata(),
            src.shape().nx * sizeof(T),
            src.shape().nx, src.shape().ny);
        p.dstPtr = make_cudaPitchedPtr(
            (void*)dst.data(),
            dst.pitch(),
            dst.shape().nx, dst.shape().ny);
        p.extent = make_cudaExtent(
            src.shape().nx * sizeof(T),
            src.shape().ny, src.shape().nz);
        p.kind = cudaMemcpyHostToDevice;
        YK_CUDA_CHECK(cudaMemcpy3D(&p));
    }

    // ================================================================
    // Upload 异步（HostPinnedBuffer3D → Device，需要 stream）
    // ================================================================

    template<typename T>
    void upload3DAsync(const DeviceLinearBuffer3D<T>& dst,
                       const HostPinnedBuffer3D<T>& src,
                       cudaStream_t stream) const
    {
        if (!dst || !src)
            throw std::invalid_argument("upload3DAsync: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(dst.deviceId()));
        YK_CUDA_CHECK(cudaMemcpyAsync(
            dst.data(), src.data(),
            src.size() * sizeof(T),
            cudaMemcpyHostToDevice, stream));
    }

    template<typename T>
    void upload3DAsync(const DevicePitchedBuffer3D<T>& dst,
                       const HostPinnedBuffer3D<T>& src,
                       cudaStream_t stream) const
    {
        if (!dst || !src)
            throw std::invalid_argument("upload3DAsync: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(dst.deviceId()));
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr(
            (void*)src.data(),
            src.shape().nx * sizeof(T),
            src.shape().nx, src.shape().ny);
        p.dstPtr = make_cudaPitchedPtr(
            (void*)dst.data(),
            dst.pitch(),
            dst.shape().nx, dst.shape().ny);
        p.extent = make_cudaExtent(
            src.shape().nx * sizeof(T),
            src.shape().ny, src.shape().nz);
        p.kind = cudaMemcpyHostToDevice;
        YK_CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    }

    template<typename T>
    void upload3DAsyncSync(const DeviceLinearBuffer3D<T>& dst,
                           const HostPinnedBuffer3D<T>& src,
                           cudaStream_t stream) const
    {
        upload3DAsync(dst, src, stream);
        sync(stream);
    }

    template<typename T>
    void upload3DAsyncSync(const DevicePitchedBuffer3D<T>& dst,
                           const HostPinnedBuffer3D<T>& src,
                           cudaStream_t stream) const
    {
        upload3DAsync(dst, src, stream);
        sync(stream);
    }

    // ================================================================
    // Download 同步（Device → CpuBuffer3D，无需 stream）
    // ================================================================

    // DeviceLinearBuffer3D → CpuBuffer3D
    template<typename T>
    void download3D(CpuBuffer3D<T>& dst,
                    const DeviceLinearBuffer3D<T>& src) const
    {
        if (!dst || !src)
            throw std::invalid_argument("download3D: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(src.deviceId()));
        YK_CUDA_CHECK(cudaMemcpy(
            dst.data(), src.cdata(),
            src.size() * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    // DevicePitchedBuffer3D → CpuBuffer3D
    template<typename T>
    void download3D(CpuBuffer3D<T>& dst,
                    const DevicePitchedBuffer3D<T>& src) const
    {
        if (!dst || !src)
            throw std::invalid_argument("download3D: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(src.deviceId()));
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr(
            (void*)src.cdata(),
            src.pitch(),
            src.shape().nx, src.shape().ny);
        p.dstPtr = make_cudaPitchedPtr(
            (void*)dst.data(),
            dst.shape().nx * sizeof(T),
            dst.shape().nx, dst.shape().ny);
        p.extent = make_cudaExtent(
            dst.shape().nx * sizeof(T),
            dst.shape().ny, dst.shape().nz);
        p.kind = cudaMemcpyDeviceToHost;
        YK_CUDA_CHECK(cudaMemcpy3D(&p));
    }

    // ================================================================
    // Download 异步（Device → HostPinnedBuffer3D，需要 stream）
    // ================================================================

    template<typename T>
    void download3DAsync(HostPinnedBuffer3D<T>& dst,
                         const DeviceLinearBuffer3D<T>& src,
                         cudaStream_t stream) const
    {
        if (!dst || !src)
            throw std::invalid_argument("download3DAsync: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(src.deviceId()));
        YK_CUDA_CHECK(cudaMemcpyAsync(
            dst.data(), src.cdata(),
            src.size() * sizeof(T),
            cudaMemcpyDeviceToHost, stream));
    }

    template<typename T>
    void download3DAsync(HostPinnedBuffer3D<T>& dst,
                         const DevicePitchedBuffer3D<T>& src,
                         cudaStream_t stream) const
    {
        if (!dst || !src)
            throw std::invalid_argument("download3DAsync: null ptr");

        YK_CUDA_CHECK(cudaSetDevice(src.deviceId()));
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr(
            (void*)src.cdata(),
            src.pitch(),
            src.shape().nx, src.shape().ny);
        p.dstPtr = make_cudaPitchedPtr(
            (void*)dst.data(),
            dst.shape().nx * sizeof(T),
            dst.shape().nx, dst.shape().ny);
        p.extent = make_cudaExtent(
            dst.shape().nx * sizeof(T),
            dst.shape().ny, dst.shape().nz);
        p.kind = cudaMemcpyDeviceToHost;
        YK_CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    }

    template<typename T>
    void download3DAsyncSync(HostPinnedBuffer3D<T>& dst,
                             const DeviceLinearBuffer3D<T>& src,
                             cudaStream_t stream) const
    {
        download3DAsync(dst, src, stream);
        sync(stream);
    }

    template<typename T>
    void download3DAsyncSync(HostPinnedBuffer3D<T>& dst,
                             const DevicePitchedBuffer3D<T>& src,
                             cudaStream_t stream) const
    {
        download3DAsync(dst, src, stream);
        sync(stream);
    }
};

// ================================================================
// PodDataController
// 专供 1D POD / 结构体数组（geo参数、LUT、校正表等）
// ================================================================
class PodDataController {
public:

    template<typename T>
    DeviceLinearBuffer<T> allocate(int n, int deviceId = 0) const
    {
        DeviceLinearBuffer<T> buf;
        buf.alloc(n, deviceId);
        return buf;
    }

    // 便捷：alloc + 同步上传一步完成
    template<typename T>
    DeviceLinearBuffer<T> allocateAndUpload(
        const std::vector<T>& src,
        int deviceId = 0) const
    {
        auto buf = allocate<T>((int)src.size(), deviceId);
        upload(buf, src);
        return buf;
    }

    // ----------------------------------------------------------------
    // 同步上传：src 是普通内存，安全
    // ----------------------------------------------------------------
    template<typename T>
    void upload(const DeviceLinearBuffer<T>& dst,
                const std::vector<T>& src) const
    {
        if (src.empty() || !dst)
            throw std::invalid_argument("PodDataController::upload: null or empty");
        YK_CUDA_CHECK(cudaMemcpy(
            dst.data(), src.data(),
            src.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    template<typename T>
    void upload(const DeviceLinearBuffer<T>& dst,
                const T* src, int n) const
    {
        if (!src || !dst)
            throw std::invalid_argument("PodDataController::upload: null ptr");
        YK_CUDA_CHECK(cudaMemcpy(
            dst.data(), src,
            n * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    // ----------------------------------------------------------------
    // 异步上传：src 必须是 HostPinnedBuffer
    // ----------------------------------------------------------------
    template<typename T>
    void uploadAsync(const DeviceLinearBuffer<T>& dst,
                     const HostPinnedBuffer<T>& src,
                     int n,
                     cudaStream_t stream) const
    {
        if (!src || !dst)
            throw std::invalid_argument("PodDataController::uploadAsync: null ptr");
        if (n > src.count())
            throw std::invalid_argument("PodDataController::uploadAsync: n > src.count()");
        YK_CUDA_CHECK(cudaMemcpyAsync(
            dst.data(), src.data(),
            n * sizeof(T),
            cudaMemcpyHostToDevice, stream));
    }

    // ----------------------------------------------------------------
    // 同步下载
    // ----------------------------------------------------------------
    template<typename T>
    void download(std::vector<T>& dst,
                  const DeviceLinearBuffer<T>& src) const
    {
        if (!src)
            throw std::invalid_argument("PodDataController::download: null src");
        dst.resize(src.count());
        YK_CUDA_CHECK(cudaMemcpy(
            dst.data(), src.data(),
            src.count() * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    // ----------------------------------------------------------------
    // 异步下载：dst 必须是 HostPinnedBuffer，调用方负责 sync
    // ----------------------------------------------------------------
    template<typename T>
    void downloadAsync(HostPinnedBuffer<T>& dst,
                       const DeviceLinearBuffer<T>& src,
                       cudaStream_t stream) const
    {
        if (!src || !dst)
            throw std::invalid_argument("PodDataController::downloadAsync: null ptr");
        if (src.count() > dst.count())
            throw std::invalid_argument("PodDataController::downloadAsync: dst too small");
        YK_CUDA_CHECK(cudaMemcpyAsync(
            dst.data(), src.data(),
            src.count() * sizeof(T),
            cudaMemcpyDeviceToHost, stream));
    }

    void sync(cudaStream_t stream) const
    {
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

} // namespace Mem
} // namespace YK
