#pragma once
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "YkMemTypes.h"
#include "YkGlobals.h"   // YK_CUDA_CHECK

namespace YK {
namespace Mem {

// ============================================================
// CpuBuffer3D
// 拥有所有权的 CPU 3D 缓冲，线性布局，无 padding
// ============================================================
template<typename T>
class CpuBuffer3D {
public:
    using ValueType = T;

    CpuBuffer3D() = default;

    CpuBuffer3D(int nx, int ny, int nz, bool zero = false)
        : sh_{ nx, ny, nz }, sliceStride_(size_t(nx) * ny)
    {
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::invalid_argument("CpuBuffer3D: nx/ny/nz must > 0");
        ptr_ = std::make_unique<T[]>(size_t(nx) * ny * nz);
        if (zero)
            std::memset(ptr_.get(), 0, sizeof(T) * size_t(nx) * ny * nz);
    }

    CpuBuffer3D(const CpuBuffer3D&)            = delete;
    CpuBuffer3D& operator=(const CpuBuffer3D&) = delete;

    CpuBuffer3D(CpuBuffer3D&& o) noexcept
        : ptr_(std::move(o.ptr_)), sh_(o.sh_), sliceStride_(o.sliceStride_)
    { o.sh_ = {}; o.sliceStride_ = 0; }

    CpuBuffer3D& operator=(CpuBuffer3D&& o) noexcept {
        if (this != &o) {
            ptr_         = std::move(o.ptr_);
            sh_          = o.sh_;
            sliceStride_ = o.sliceStride_;
            o.sh_ = {}; o.sliceStride_ = 0;
        }
        return *this;
    }

    ~CpuBuffer3D() = default;

    CpuView3D<T>       view()  noexcept       { return { ptr_.get(), sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }
    CpuView3D<const T> cview() const noexcept { return { ptr_.get(), sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }

    Shape3D      shape()  const noexcept { return sh_; }
    T*           data()         noexcept { return ptr_.get(); }
    const T*     cdata()  const noexcept { return ptr_.get(); }
    uint64_t     size()   const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    std::unique_ptr<T[]> ptr_;
    Shape3D sh_{};
    size_t  sliceStride_ = 0;
};

// ============================================================
// CpuBuffer3DBorrowed
// 非拥有 CPU 3D 缓冲，指向外部内存
// ============================================================
template<typename T>
class CpuBuffer3DBorrowed {
public:
    using ValueType = T;

    CpuBuffer3DBorrowed(T* ptr, int nx, int ny, int nz)
        : ptr_(ptr), sh_{ nx, ny, nz }, sliceStride_(size_t(nx) * ny)
    {
        if (!ptr)
            throw std::invalid_argument("CpuBuffer3DBorrowed: ptr is null");
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::invalid_argument("CpuBuffer3DBorrowed: nx/ny/nz must > 0");
    }

    CpuBuffer3DBorrowed(const CpuBuffer3DBorrowed&)            = delete;
    CpuBuffer3DBorrowed& operator=(const CpuBuffer3DBorrowed&) = delete;

    CpuBuffer3DBorrowed(CpuBuffer3DBorrowed&& o) noexcept
        : ptr_(o.ptr_), sh_(o.sh_), sliceStride_(o.sliceStride_)
    { o.ptr_ = nullptr; o.sh_ = {}; o.sliceStride_ = 0; }

    CpuBuffer3DBorrowed& operator=(CpuBuffer3DBorrowed&& o) noexcept {
        if (this != &o) {
            ptr_         = o.ptr_;        o.ptr_ = nullptr;
            sh_          = o.sh_;         o.sh_ = {};
            sliceStride_ = o.sliceStride_; o.sliceStride_ = 0;
        }
        return *this;
    }

    ~CpuBuffer3DBorrowed() = default;   // 不释放，外部管理

    CpuView3D<T>       view()  noexcept       { return { ptr_, sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }
    CpuView3D<const T> cview() const noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }

    Shape3D      shape()  const noexcept { return sh_; }
    T*           data()         noexcept { return ptr_; }
    const T*     cdata()  const noexcept { return ptr_; }
    uint64_t     size()   const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*      ptr_         = nullptr;
    Shape3D sh_{};
    size_t  sliceStride_ = 0;
};

// ============================================================
// HostPinnedBuffer
// 1D pinned memory，专供 1D POD 数组异步传输使用
// ============================================================
template<typename T>
class HostPinnedBuffer {
public:
    HostPinnedBuffer() = default;
    ~HostPinnedBuffer() { reset(); }

    HostPinnedBuffer(const HostPinnedBuffer&)            = delete;
    HostPinnedBuffer& operator=(const HostPinnedBuffer&) = delete;

    HostPinnedBuffer(HostPinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), n_(o.n_)
    { o.ptr_ = nullptr; o.n_ = 0; }

    HostPinnedBuffer& operator=(HostPinnedBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr_ = o.ptr_; o.ptr_ = nullptr; n_ = o.n_; o.n_ = 0; }
        return *this;
    }

    void alloc(int n) {
        reset();
        n_ = n;
        YK_CUDA_CHECK(cudaMallocHost(&ptr_, n * sizeof(T)));
    }

    void reset() {
        if (ptr_) { cudaFreeHost(ptr_); ptr_ = nullptr; }
        n_ = 0;
    }

    void copyFrom(const std::vector<T>& src) {
        if ((int)src.size() > n_)
            throw std::invalid_argument("HostPinnedBuffer::copyFrom: src too large");
        std::memcpy(ptr_, src.data(), src.size() * sizeof(T));
    }

    T*       data()  const noexcept { return ptr_; }
    int      count() const noexcept { return n_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*  ptr_ = nullptr;
    int n_   = 0;
};

// ============================================================
// HostPinnedBuffer3D
// 3D pinned memory，专供 3D buffer 异步传输使用
// ============================================================
template<typename T>
class HostPinnedBuffer3D {
public:
    HostPinnedBuffer3D() = default;
    ~HostPinnedBuffer3D() { reset(); }

    HostPinnedBuffer3D(const HostPinnedBuffer3D&)            = delete;
    HostPinnedBuffer3D& operator=(const HostPinnedBuffer3D&) = delete;

    HostPinnedBuffer3D(HostPinnedBuffer3D&& o) noexcept
        : ptr_(o.ptr_), sh_(o.sh_)
    { o.ptr_ = nullptr; o.sh_ = {}; }

    HostPinnedBuffer3D& operator=(HostPinnedBuffer3D&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_; o.ptr_ = nullptr;
            sh_  = o.sh_;  o.sh_  = {};
        }
        return *this;
    }

    void alloc(int nx, int ny, int nz) {
        reset();
        sh_ = { nx, ny, nz };
        YK_CUDA_CHECK(cudaMallocHost(&ptr_, uint64_t(nx) * ny * nz * sizeof(T)));
    }

    void reset() {
        if (ptr_) { cudaFreeHost(ptr_); ptr_ = nullptr; }
        sh_ = {};
    }

    void copyFrom(const CpuBuffer3D<T>& src) {
        if (src.size() > uint64_t(sh_.nx) * sh_.ny * sh_.nz)
            throw std::invalid_argument("HostPinnedBuffer3D::copyFrom: src too large");
        std::memcpy(ptr_, src.cdata(), src.size() * sizeof(T));
    }

    T*       data()  const noexcept { return ptr_; }
    Shape3D  shape() const noexcept { return sh_; }
    uint64_t size()  const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*      ptr_ = nullptr;
    Shape3D sh_{};
};

// ============================================================
// DeviceLinearBuffer
// 1D 线性设备缓冲，专供结构体 / POD 数组使用（geo参数、LUT等）
// 底层：cudaMalloc
// ============================================================
template<typename T>
class DeviceLinearBuffer {
public:
    DeviceLinearBuffer() = default;
    ~DeviceLinearBuffer() { reset(); }

    DeviceLinearBuffer(const DeviceLinearBuffer&)            = delete;
    DeviceLinearBuffer& operator=(const DeviceLinearBuffer&) = delete;

    DeviceLinearBuffer(DeviceLinearBuffer&& o) noexcept
        : ptr_(o.ptr_), n_(o.n_), deviceId_(o.deviceId_)
    { o.ptr_ = nullptr; o.n_ = 0; o.deviceId_ = 0; }

    DeviceLinearBuffer& operator=(DeviceLinearBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_      = o.ptr_;      o.ptr_      = nullptr;
            n_        = o.n_;        o.n_        = 0;
            deviceId_ = o.deviceId_; o.deviceId_ = 0;
        }
        return *this;
    }

    void alloc(int n, int deviceId = 0) {
        reset();
        deviceId_ = deviceId;
        n_        = n;
        YK_CUDA_CHECK(cudaSetDevice(deviceId_));
        YK_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }

    void reset() {
        if (ptr_) {
            cudaSetDevice(deviceId_);
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        n_ = 0;
    }

    T*       data()     const noexcept { return ptr_; }
    int      count()    const noexcept { return n_; }
    int      deviceId() const noexcept { return deviceId_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*  ptr_      = nullptr;
    int n_        = 0;
    int deviceId_ = 0;
};

// ============================================================
// DeviceLinearBuffer3D
// 3D 线性设备缓冲，pitch = nx*sizeof(T)，无 padding
// 用于 vol buffer，kernel 内可直接用 Nx 计算 idx
// 底层：cudaMalloc
// ============================================================
template<typename T>
class DeviceLinearBuffer3D {
public:
    using ValueType = T;

    DeviceLinearBuffer3D() = default;
    ~DeviceLinearBuffer3D() { reset_noexcept(); }

    DeviceLinearBuffer3D(const DeviceLinearBuffer3D&)            = delete;
    DeviceLinearBuffer3D& operator=(const DeviceLinearBuffer3D&) = delete;

    DeviceLinearBuffer3D(DeviceLinearBuffer3D&& o) noexcept { move_from(o); }
    DeviceLinearBuffer3D& operator=(DeviceLinearBuffer3D&& o) noexcept {
        if (this != &o) { reset_noexcept(); move_from(o); }
        return *this;
    }

    // pitch 固定为 nx*sizeof(T)，对外透明
    YK_NODISCARD DeviceView3D<T> view() const noexcept
    {
        const size_t pb = size_t(sh_.nx) * sizeof(T);
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pb, pb * sh_.ny };
    }

    YK_NODISCARD DeviceView3D<const T> cview() const noexcept
    {
        const size_t pb = size_t(sh_.nx) * sizeof(T);
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pb, pb * sh_.ny };
    }

    Shape3D  shape()    const noexcept { return sh_; }
    size_t   pitch()    const noexcept { return size_t(sh_.nx) * sizeof(T); }
    uint64_t size()     const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    int      deviceId() const noexcept { return deviceId_; }
    T*       data()     const noexcept { return ptr_; }
    const T* cdata()    const noexcept { return ptr_; }
    void     reset()    noexcept       { reset_noexcept(); }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    friend class MemoryController;

    static DeviceLinearBuffer3D make_owning(T* ptr, Shape3D sh, int deviceId) {
        DeviceLinearBuffer3D b;
        b.ptr_      = ptr;
        b.sh_       = sh;
        b.deviceId_ = deviceId;
        return b;
    }

    void reset_noexcept() noexcept {
        if (ptr_) {
            YK_CUDA_CHECK(cudaSetDevice(deviceId_));
            YK_CUDA_CHECK(cudaFree(ptr_));
            ptr_ = nullptr;
        }
        sh_ = {}; deviceId_ = 0;
    }

    void move_from(DeviceLinearBuffer3D& o) noexcept {
        ptr_      = o.ptr_;      o.ptr_      = nullptr;
        sh_       = o.sh_;       o.sh_       = {};
        deviceId_ = o.deviceId_; o.deviceId_ = 0;
    }

    T*      ptr_      = nullptr;
    Shape3D sh_{};
    int     deviceId_ = 0;
};

// ============================================================
// DeviceLinearBuffer3DBorrowed
// 非拥有，指向外部线性 GPU 内存（第三方传入的 vol 指针）
// ============================================================
template<typename T>
class DeviceLinearBuffer3DBorrowed {
public:
    using ValueType = T;

    DeviceLinearBuffer3DBorrowed(T* ptr, int nx, int ny, int nz, int deviceId = 0)
        : ptr_(ptr), sh_{ nx, ny, nz }, deviceId_(deviceId)
    {
        if (!ptr)
            throw std::invalid_argument("DeviceLinearBuffer3DBorrowed: ptr is null");
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::invalid_argument("DeviceLinearBuffer3DBorrowed: nx/ny/nz > 0");
    }

    DeviceLinearBuffer3DBorrowed(const DeviceLinearBuffer3DBorrowed&)            = delete;
    DeviceLinearBuffer3DBorrowed& operator=(const DeviceLinearBuffer3DBorrowed&) = delete;

    DeviceLinearBuffer3DBorrowed(DeviceLinearBuffer3DBorrowed&& o) noexcept
        : ptr_(o.ptr_), sh_(o.sh_), deviceId_(o.deviceId_)
    { o.ptr_ = nullptr; o.sh_ = {}; }

    DeviceLinearBuffer3DBorrowed& operator=(DeviceLinearBuffer3DBorrowed&& o) noexcept {
        if (this != &o) {
            ptr_      = o.ptr_;      o.ptr_      = nullptr;
            sh_       = o.sh_;       o.sh_       = {};
            deviceId_ = o.deviceId_; o.deviceId_ = 0;
        }
        return *this;
    }

    ~DeviceLinearBuffer3DBorrowed() = default;  // 不释放

    YK_NODISCARD DeviceView3D<T> view() const noexcept
    {
        const size_t pb = size_t(sh_.nx) * sizeof(T);
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pb, pb * sh_.ny };
    }

    YK_NODISCARD DeviceView3D<const T> cview() const noexcept
    {
        const size_t pb = size_t(sh_.nx) * sizeof(T);
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pb, pb * sh_.ny };
    }

    Shape3D  shape()    const noexcept { return sh_; }
    size_t   pitch()    const noexcept { return size_t(sh_.nx) * sizeof(T); }
    uint64_t size()     const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    int      deviceId() const noexcept { return deviceId_; }
    T*       data()     const noexcept { return ptr_; }
    const T* cdata()    const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*      ptr_      = nullptr;
    Shape3D sh_{};
    int     deviceId_ = 0;
};

// ============================================================
// DevicePitchedBuffer3D
// 3D pitched 设备缓冲，每行对齐到 256 字节
// 用于投影纹理 buffer，绑定纹理对象时使用 pitch()
// 底层：cudaMalloc3D
// ============================================================
template<typename T>
class DevicePitchedBuffer3D {
public:
    using ValueType = T;

    DevicePitchedBuffer3D() = default;
    ~DevicePitchedBuffer3D() { reset_noexcept(); }

    DevicePitchedBuffer3D(const DevicePitchedBuffer3D&)            = delete;
    DevicePitchedBuffer3D& operator=(const DevicePitchedBuffer3D&) = delete;

    DevicePitchedBuffer3D(DevicePitchedBuffer3D&& o) noexcept { move_from(o); }
    DevicePitchedBuffer3D& operator=(DevicePitchedBuffer3D&& o) noexcept {
        if (this != &o) { reset_noexcept(); move_from(o); }
        return *this;
    }

    YK_NODISCARD DeviceView3D<T> view() const noexcept
    {
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, pitchBytes_ * sh_.ny };
    }

    YK_NODISCARD DeviceView3D<const T> cview() const noexcept
    {
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, pitchBytes_ * sh_.ny };
    }

    Shape3D  shape()      const noexcept { return sh_; }
    size_t   pitch()      const noexcept { return pitchBytes_; }
    uint64_t size()       const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    int      deviceId()   const noexcept { return deviceId_; }
    T*       data()       const noexcept { return ptr_; }
    const T* cdata()      const noexcept { return ptr_; }
    void     reset()      noexcept       { reset_noexcept(); }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    friend class MemoryController;

    static DevicePitchedBuffer3D make_owning(T* ptr, Shape3D sh, size_t pitchBytes, int deviceId) {
        DevicePitchedBuffer3D b;
        b.ptr_        = ptr;
        b.sh_         = sh;
        b.pitchBytes_ = pitchBytes;
        b.deviceId_   = deviceId;
        return b;
    }

    void reset_noexcept() noexcept {
        if (ptr_) {
            YK_CUDA_CHECK(cudaSetDevice(deviceId_));
            YK_CUDA_CHECK(cudaFree(ptr_));
            ptr_ = nullptr;
        }
        sh_ = {}; pitchBytes_ = 0; deviceId_ = 0;
    }

    void move_from(DevicePitchedBuffer3D& o) noexcept {
        ptr_        = o.ptr_;        o.ptr_        = nullptr;
        sh_         = o.sh_;         o.sh_         = {};
        pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
        deviceId_   = o.deviceId_;   o.deviceId_   = 0;
    }

    T*      ptr_        = nullptr;
    Shape3D sh_{};
    size_t  pitchBytes_ = 0;
    int     deviceId_   = 0;
};

// ============================================================
// DevicePitchedBuffer3DBorrowed
// 非拥有，指向外部 pitched GPU 内存（第三方传入的投影指针）
// ============================================================
template<typename T>
class DevicePitchedBuffer3DBorrowed {
public:
    using ValueType = T;

    DevicePitchedBuffer3DBorrowed(T* ptr, int nx, int ny, int nz,
                                  size_t pitchBytes, int deviceId = 0)
        : ptr_(ptr), sh_{ nx, ny, nz }, pitchBytes_(pitchBytes), deviceId_(deviceId)
    {
        if (!ptr)
            throw std::invalid_argument("DevicePitchedBuffer3DBorrowed: ptr is null");
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::invalid_argument("DevicePitchedBuffer3DBorrowed: nx/ny/nz > 0");
        if (pitchBytes < size_t(nx) * sizeof(T))
            throw std::invalid_argument("DevicePitchedBuffer3DBorrowed: pitchBytes < nx*sizeof(T)");
    }

    DevicePitchedBuffer3DBorrowed(const DevicePitchedBuffer3DBorrowed&)            = delete;
    DevicePitchedBuffer3DBorrowed& operator=(const DevicePitchedBuffer3DBorrowed&) = delete;

    DevicePitchedBuffer3DBorrowed(DevicePitchedBuffer3DBorrowed&& o) noexcept
        : ptr_(o.ptr_), sh_(o.sh_), pitchBytes_(o.pitchBytes_), deviceId_(o.deviceId_)
    { o.ptr_ = nullptr; o.sh_ = {}; o.pitchBytes_ = 0; }

    DevicePitchedBuffer3DBorrowed& operator=(DevicePitchedBuffer3DBorrowed&& o) noexcept {
        if (this != &o) {
            ptr_        = o.ptr_;        o.ptr_        = nullptr;
            sh_         = o.sh_;         o.sh_         = {};
            pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
            deviceId_   = o.deviceId_;   o.deviceId_   = 0;
        }
        return *this;
    }

    ~DevicePitchedBuffer3DBorrowed() = default;  // 不释放

    YK_NODISCARD DeviceView3D<T> view() const noexcept
    {
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, pitchBytes_ * sh_.ny };
    }

    YK_NODISCARD DeviceView3D<const T> cview() const noexcept
    {
        return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, pitchBytes_ * sh_.ny };
    }

    Shape3D  shape()    const noexcept { return sh_; }
    size_t   pitch()    const noexcept { return pitchBytes_; }
    uint64_t size()     const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
    int      deviceId() const noexcept { return deviceId_; }
    T*       data()     const noexcept { return ptr_; }
    const T* cdata()    const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*      ptr_        = nullptr;
    Shape3D sh_{};
    size_t  pitchBytes_ = 0;
    int     deviceId_   = 0;
};

} // namespace Mem
} // namespace YK
