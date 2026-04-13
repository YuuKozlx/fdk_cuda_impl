#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "YkGlobals.h"  // YK_CUDA_CHECK

// ============================================================
// 条件编译 C++14 / C++20
// ============================================================
#if __cplusplus >= 202002L
#define YK_NODISCARD [[nodiscard]]
#define YK_BYTE      std::byte
#else
#define YK_NODISCARD
#define YK_BYTE      unsigned char
#endif

#ifdef __CUDACC__
#define YK_HD __host__ __device__
#else
#define YK_HD
#endif

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
//   - host 端内存【必须】是 pinned memory（HostPinnedBuffer 系列）
//   - 函数立即返回，传输在 stream 上异步执行
//   - 调用方负责在合适位置 sync(stream)
//   - 适用：热路径、pipeline 中 CPU/GPU 重叠执行
//
// 二、Pinned Memory
// ─────────────────────────────────────────────────────────────
//   分配：cudaMallocHost   释放：cudaFreeHost
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
// 四、Buffer 类型与分配方式对照
// ─────────────────────────────────────────────────────────────
//
//   DeviceLinearBuffer          1D POD/结构体数组  cudaMalloc
//   DeviceLinearBuffer3D        3D vol buffer      cudaMalloc       pitch=nx*sizeof(T)
//   DevicePitchedBuffer3D       3D 投影纹理buffer  cudaMalloc3D     pitch=驱动对齐值
//   DeviceLinearBuffer3DBorrowed     借用外部线性3D指针，构造时验证pitch
//   DevicePitchedBuffer3DBorrowed    借用外部pitched3D指针，需显式传pitch
//
// 五、接口命名约定
// ─────────────────────────────────────────────────────────────
//   upload / download              同步，普通 CPU buffer
//   uploadAsync / downloadAsync    异步，必须传 pinned buffer，调用方负责sync
//   uploadAsyncSync / downloadAsyncSync  异步 + 内部 sync
//   sync(stream)                   统一同步点，不直接散用cudaStreamSynchronize
//
// 六、常见错误
// ─────────────────────────────────────────────────────────────
//   [错误] std::vector + cudaMemcpyAsync          → UB
//   [错误] 局部变量 + cudaMemcpyToSymbolAsync     → DMA 读野指针
//   [错误] host 端直接赋值 __constant__ 变量      → device 不更新
//   [错误] process() 内部 cudaStreamSynchronize   → 破坏流水线
//   [错误] 外部pitched指针用DeviceLinearBuffer3DBorrowed接收 → 构造时抛异常
// ============================================================

namespace YK {
    namespace Mem {

        // ============================================================
        // 基础类型
        // ============================================================

        struct Shape3D { int nx = 0, ny = 0, nz = 0; };
        struct Region3D { int ox = 0, oy = 0, oz = 0; int sx = 0, sy = 0, sz = 0; };
        enum class OwnerTag : uint8_t { Owning, Borrowed };

        // ============================================================
        // CpuView3D
        // 轻量级非拥有视图，指向已分配的 CPU 内存（线性布局）
        // ============================================================
        template<typename T>
        struct CpuView3D {
            T* ptr = nullptr;
            int    nx = 0, ny = 0, nz = 0;
            size_t sliceStride = 0;     // 每个z层的元素跨度 = nx*ny

            explicit operator bool() const noexcept { return ptr != nullptr; }

            T* at(int x, int y, int z) const noexcept
            {
                return ptr + size_t(z) * sliceStride + size_t(y) * nx + x;
            }

#if __cplusplus >= 202002L
            auto operator()(int x, int y, int z) const noexcept -> decltype(*at(x, y, z))
            {
                return *at(x, y, z);
            }
#else
            typename std::conditional<std::is_const<T>::value, const T&, T&>::type
                operator()(int x, int y, int z) const noexcept { return *at(x, y, z); }
#endif
        };

        // ============================================================
        // DeviceView3D
        // 轻量级非拥有视图，指向已分配的 GPU 内存
        // 线性时 pitchBytes = nx*sizeof(T)，pitched时 pitchBytes >= nx*sizeof(T)
        // ============================================================
        template<typename T>
        struct DeviceView3D {
            T* ptr = nullptr;
            int    nx = 0, ny = 0, nz = 0;
            size_t pitchBytes = 0;
            size_t sliceBytes = 0;

            explicit operator bool() const noexcept { return ptr != nullptr; }

            YK_HD T* at(int x, int y, int z) const noexcept
            {
                auto byteOffset = size_t(z) * sliceBytes + size_t(y) * pitchBytes;
                return reinterpret_cast<T*>(
                    reinterpret_cast<YK_BYTE*>(ptr) + byteOffset) + x;
            }

#if __cplusplus >= 202002L
            YK_HD auto operator()(int x, int y, int z) const noexcept -> decltype(*at(x, y, z))
            {
                return *at(x, y, z);
            }
#else
            YK_HD typename std::conditional<std::is_const<T>::value, const T&, T&>::type
                operator()(int x, int y, int z) const noexcept { return *at(x, y, z); }
#endif
        };

        // ============================================================
        // CpuBuffer3D  —  拥有所有权的 CPU 3D 缓冲，线性布局
        // ============================================================
        template<typename T>
        class CpuBuffer3D {
        public:
            using ValueType = T;

            CpuBuffer3D() = default;

            CpuBuffer3D(int nx, int ny, int nz, bool zero = false)
                : sh_{ nx, ny, nz }, sliceStride_(size_t(nx)* ny)
            {
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("CpuBuffer3D: nx/ny/nz must > 0");
                ptr_ = std::make_unique<T[]>(size_t(nx) * ny * nz);
                if (zero)
                    std::memset(ptr_.get(), 0, sizeof(T) * size_t(nx) * ny * nz);
            }

            CpuBuffer3D(const CpuBuffer3D&) = delete;
            CpuBuffer3D& operator=(const CpuBuffer3D&) = delete;

            CpuBuffer3D(CpuBuffer3D&& o) noexcept
                : ptr_(std::move(o.ptr_)), sh_(o.sh_), sliceStride_(o.sliceStride_)
            {
                o.sh_ = {}; o.sliceStride_ = 0;
            }

            CpuBuffer3D& operator=(CpuBuffer3D&& o) noexcept {
                if (this != &o) {
                    ptr_ = std::move(o.ptr_);
                    sh_ = o.sh_;
                    sliceStride_ = o.sliceStride_;
                    o.sh_ = {}; o.sliceStride_ = 0;
                }
                return *this;
            }

            ~CpuBuffer3D() = default;

            CpuView3D<T>       view()  noexcept { return { ptr_.get(), sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }
            CpuView3D<const T> cview() const noexcept { return { ptr_.get(), sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }

            Shape3D      shape()  const noexcept { return sh_; }
            T* data()         noexcept { return ptr_.get(); }
            const T* cdata()  const noexcept { return ptr_.get(); }
            uint64_t     size()   const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            std::unique_ptr<T[]> ptr_;
            Shape3D sh_{};
            size_t  sliceStride_ = 0;
        };

        // ============================================================
        // CpuBuffer3DBorrowed  —  非拥有 CPU 3D 缓冲，指向外部内存
        // ============================================================
        template<typename T>
        class CpuBuffer3DBorrowed {
        public:
            using ValueType = T;

            CpuBuffer3DBorrowed(T* ptr, int nx, int ny, int nz)
                : ptr_(ptr), sh_{ nx, ny, nz }, sliceStride_(size_t(nx)* ny)
            {
                if (!ptr)
                    throw std::invalid_argument("CpuBuffer3DBorrowed: ptr is null");
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("CpuBuffer3DBorrowed: nx/ny/nz must > 0");
            }

            CpuBuffer3DBorrowed(const CpuBuffer3DBorrowed&) = delete;
            CpuBuffer3DBorrowed& operator=(const CpuBuffer3DBorrowed&) = delete;

            CpuBuffer3DBorrowed(CpuBuffer3DBorrowed&& o) noexcept
                : ptr_(o.ptr_), sh_(o.sh_), sliceStride_(o.sliceStride_)
            {
                o.ptr_ = nullptr; o.sh_ = {}; o.sliceStride_ = 0;
            }

            CpuBuffer3DBorrowed& operator=(CpuBuffer3DBorrowed&& o) noexcept {
                if (this != &o) {
                    ptr_ = o.ptr_;         o.ptr_ = nullptr;
                    sh_ = o.sh_;          o.sh_ = {};
                    sliceStride_ = o.sliceStride_; o.sliceStride_ = 0;
                }
                return *this;
            }

            ~CpuBuffer3DBorrowed() = default;   // 不释放，外部管理

            CpuView3D<T>       view()  noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }
            CpuView3D<const T> cview() const noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }

            Shape3D      shape()  const noexcept { return sh_; }
            T* data()         noexcept { return ptr_; }
            const T* cdata()  const noexcept { return ptr_; }
            uint64_t     size()   const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            Shape3D sh_{};
            size_t  sliceStride_ = 0;
        };

        // ============================================================
        // HostPinnedBuffer  —  1D pinned memory，供 1D POD 异步传输使用
        // ============================================================
        template<typename T>
        class HostPinnedBuffer {
        public:
            HostPinnedBuffer() = default;
            ~HostPinnedBuffer() { reset(); }

            HostPinnedBuffer(const HostPinnedBuffer&) = delete;
            HostPinnedBuffer& operator=(const HostPinnedBuffer&) = delete;

            HostPinnedBuffer(HostPinnedBuffer&& o) noexcept
                : ptr_(o.ptr_), n_(o.n_)
            {
                o.ptr_ = nullptr; o.n_ = 0;
            }

            HostPinnedBuffer& operator=(HostPinnedBuffer&& o) noexcept {
                if (this != &o) {
                    reset();
                    ptr_ = o.ptr_; o.ptr_ = nullptr;
                    n_ = o.n_;   o.n_ = 0;
                }
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

            T* data()  const noexcept { return ptr_; }
            int      count() const noexcept { return n_; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            int n_ = 0;
        };

        // ============================================================
        // HostPinnedBuffer3D  —  3D pinned memory，供 3D buffer 异步传输使用
        // ============================================================
        template<typename T>
        class HostPinnedBuffer3D {
        public:
            HostPinnedBuffer3D() = default;
            ~HostPinnedBuffer3D() { reset(); }

            HostPinnedBuffer3D(const HostPinnedBuffer3D&) = delete;
            HostPinnedBuffer3D& operator=(const HostPinnedBuffer3D&) = delete;

            HostPinnedBuffer3D(HostPinnedBuffer3D&& o) noexcept
                : ptr_(o.ptr_), sh_(o.sh_)
            {
                o.ptr_ = nullptr; o.sh_ = {};
            }

            HostPinnedBuffer3D& operator=(HostPinnedBuffer3D&& o) noexcept {
                if (this != &o) {
                    reset();
                    ptr_ = o.ptr_; o.ptr_ = nullptr;
                    sh_ = o.sh_;  o.sh_ = {};
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

            T* data()  const noexcept { return ptr_; }
            Shape3D  shape() const noexcept { return sh_; }
            uint64_t size()  const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            Shape3D sh_{};
        };

        // ============================================================
        // DeviceLinearBuffer  —  1D 线性设备缓冲
        // 用于结构体/POD数组（geo参数、LUT、校正表等）
        // 底层：cudaMalloc
        // ============================================================
        template<typename T>
        class DeviceLinearBuffer {
        public:
            DeviceLinearBuffer() = default;
            ~DeviceLinearBuffer() { reset(); }

            DeviceLinearBuffer(const DeviceLinearBuffer&) = delete;
            DeviceLinearBuffer& operator=(const DeviceLinearBuffer&) = delete;

            DeviceLinearBuffer(DeviceLinearBuffer&& o) noexcept
                : ptr_(o.ptr_), n_(o.n_), deviceId_(o.deviceId_)
            {
                o.ptr_ = nullptr; o.n_ = 0; o.deviceId_ = 0;
            }

            DeviceLinearBuffer& operator=(DeviceLinearBuffer&& o) noexcept {
                if (this != &o) {
                    reset();
                    ptr_ = o.ptr_;      o.ptr_ = nullptr;
                    n_ = o.n_;        o.n_ = 0;
                    deviceId_ = o.deviceId_; o.deviceId_ = 0;
                }
                return *this;
            }

            void alloc(int n, int deviceId = 0) {
                reset();
                deviceId_ = deviceId;
                n_ = n;
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

            T* data()     const noexcept { return ptr_; }
            int      count()    const noexcept { return n_; }
            int      deviceId() const noexcept { return deviceId_; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            int n_ = 0;
            int deviceId_ = 0;
        };

        // ============================================================
        // DeviceLinearBuffer3D  —  3D 线性设备缓冲
        // 用于 vol buffer，pitch = nx*sizeof(T)，无 padding
        // kernel 内可直接用 Nx 计算 idx，无需传 pitch
        // 底层：cudaMalloc
        // ============================================================
        template<typename T>
        class DeviceLinearBuffer3D {
        public:
            using ValueType = T;

            DeviceLinearBuffer3D() = default;
            ~DeviceLinearBuffer3D() { reset_noexcept(); }

            DeviceLinearBuffer3D(const DeviceLinearBuffer3D&) = delete;
            DeviceLinearBuffer3D& operator=(const DeviceLinearBuffer3D&) = delete;

            DeviceLinearBuffer3D(DeviceLinearBuffer3D&& o) noexcept { move_from(o); }
            DeviceLinearBuffer3D& operator=(DeviceLinearBuffer3D&& o) noexcept {
                if (this != &o) { reset_noexcept(); move_from(o); }
                return *this;
            }

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
            T* data()     const noexcept { return ptr_; }
            const T* cdata()    const noexcept { return ptr_; }
            void     reset()    noexcept { reset_noexcept(); }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            friend class MemoryController;

            static DeviceLinearBuffer3D make_owning(T* ptr, Shape3D sh, int deviceId) {
                DeviceLinearBuffer3D b;
                b.ptr_ = ptr;
                b.sh_ = sh;
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
                ptr_ = o.ptr_;      o.ptr_ = nullptr;
                sh_ = o.sh_;       o.sh_ = {};
                deviceId_ = o.deviceId_; o.deviceId_ = 0;
            }

            T* ptr_ = nullptr;
            Shape3D sh_{};
            int     deviceId_ = 0;
        };

        // ============================================================
        // DeviceLinearBuffer3DBorrowed  —  借用外部线性 GPU 内存
        // 构造时验证 pitchBytes == nx*sizeof(T)，传错类型立即报错
        // ============================================================
        template<typename T>
        class DeviceLinearBuffer3DBorrowed {
        public:
            using ValueType = T;

            DeviceLinearBuffer3DBorrowed(T* ptr, int nx, int ny, int nz,
                size_t pitchBytes, int deviceId = 0)
                : ptr_(ptr), sh_{ nx, ny, nz }, deviceId_(deviceId)
            {
                if (!ptr)
                    throw std::invalid_argument("DeviceLinearBuffer3DBorrowed: ptr is null");
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("DeviceLinearBuffer3DBorrowed: nx/ny/nz > 0");
                if (pitchBytes != size_t(nx) * sizeof(T))
                    throw std::invalid_argument(
                        "DeviceLinearBuffer3DBorrowed: pitchBytes != nx*sizeof(T), "
                        "use DevicePitchedBuffer3DBorrowed for pitched memory");
            }

            DeviceLinearBuffer3DBorrowed(const DeviceLinearBuffer3DBorrowed&) = delete;
            DeviceLinearBuffer3DBorrowed& operator=(const DeviceLinearBuffer3DBorrowed&) = delete;

            DeviceLinearBuffer3DBorrowed(DeviceLinearBuffer3DBorrowed&& o) noexcept
                : ptr_(o.ptr_), sh_(o.sh_), deviceId_(o.deviceId_)
            {
                o.ptr_ = nullptr; o.sh_ = {};
            }

            DeviceLinearBuffer3DBorrowed& operator=(DeviceLinearBuffer3DBorrowed&& o) noexcept {
                if (this != &o) {
                    ptr_ = o.ptr_;      o.ptr_ = nullptr;
                    sh_ = o.sh_;       o.sh_ = {};
                    deviceId_ = o.deviceId_; o.deviceId_ = 0;
                }
                return *this;
            }

            ~DeviceLinearBuffer3DBorrowed() = default;  // 不释放，外部管理

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
            T* data()     const noexcept { return ptr_; }
            const T* cdata()    const noexcept { return ptr_; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            Shape3D sh_{};
            int     deviceId_ = 0;
        };

        // ============================================================
        // DevicePitchedBuffer3D  —  3D pitched 设备缓冲
        // 用于投影纹理 buffer，行对齐到 512 字节边界
        // 绑定纹理对象时用 pitch() 填 pitchInBytes
        // 底层：cudaMalloc3D
        // ============================================================
        template<typename T>
        class DevicePitchedBuffer3D {
        public:
            using ValueType = T;

            DevicePitchedBuffer3D() = default;
            ~DevicePitchedBuffer3D() { reset_noexcept(); }

            DevicePitchedBuffer3D(const DevicePitchedBuffer3D&) = delete;
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

            Shape3D  shape()    const noexcept { return sh_; }
            size_t   pitch()    const noexcept { return pitchBytes_; }
            uint64_t size()     const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            int      deviceId() const noexcept { return deviceId_; }
            T* data()     const noexcept { return ptr_; }
            const T* cdata()    const noexcept { return ptr_; }
            void     reset()    noexcept { reset_noexcept(); }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            friend class MemoryController;

            static DevicePitchedBuffer3D make_owning(T* ptr, Shape3D sh,
                size_t pitchBytes, int deviceId)
            {
                DevicePitchedBuffer3D b;
                b.ptr_ = ptr;
                b.sh_ = sh;
                b.pitchBytes_ = pitchBytes;
                b.deviceId_ = deviceId;
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
                ptr_ = o.ptr_;        o.ptr_ = nullptr;
                sh_ = o.sh_;         o.sh_ = {};
                pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
                deviceId_ = o.deviceId_;   o.deviceId_ = 0;
            }

            T* ptr_ = nullptr;
            Shape3D sh_{};
            size_t  pitchBytes_ = 0;
            int     deviceId_ = 0;
        };

        // ============================================================
        // DevicePitchedBuffer3DBorrowed  —  借用外部 pitched GPU 内存
        // 调用方必须显式传入正确的 pitchBytes
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
                    throw std::invalid_argument(
                        "DevicePitchedBuffer3DBorrowed: pitchBytes < nx*sizeof(T)");
            }

            DevicePitchedBuffer3DBorrowed(const DevicePitchedBuffer3DBorrowed&) = delete;
            DevicePitchedBuffer3DBorrowed& operator=(const DevicePitchedBuffer3DBorrowed&) = delete;

            DevicePitchedBuffer3DBorrowed(DevicePitchedBuffer3DBorrowed&& o) noexcept
                : ptr_(o.ptr_), sh_(o.sh_), pitchBytes_(o.pitchBytes_), deviceId_(o.deviceId_)
            {
                o.ptr_ = nullptr; o.sh_ = {}; o.pitchBytes_ = 0;
            }

            DevicePitchedBuffer3DBorrowed& operator=(DevicePitchedBuffer3DBorrowed&& o) noexcept {
                if (this != &o) {
                    ptr_ = o.ptr_;        o.ptr_ = nullptr;
                    sh_ = o.sh_;         o.sh_ = {};
                    pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
                    deviceId_ = o.deviceId_;   o.deviceId_ = 0;
                }
                return *this;
            }

            ~DevicePitchedBuffer3DBorrowed() = default;  // 不释放，外部管理

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
            T* data()     const noexcept { return ptr_; }
            const T* cdata()    const noexcept { return ptr_; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            Shape3D sh_{};
            size_t  pitchBytes_ = 0;
            int     deviceId_ = 0;
        };

        // ============================================================
        // MemoryController  —  3D buffer 分配、传输、借用
        // ============================================================
        class MemoryController {
        public:

            void setDevice(int id) const { YK_CUDA_CHECK(cudaSetDevice(id)); }

            void sync(cudaStream_t stream) const
            {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // ----------------------------------------------------------------
            // CPU / Pinned 分配
            // ----------------------------------------------------------------

            template<typename T>
            CpuBuffer3D<T> allocateCpu3D(int nx, int ny, int nz, bool zero = false) const
            {
                return CpuBuffer3D<T>(nx, ny, nz, zero);
            }

            template<typename T>
            HostPinnedBuffer3D<T> allocatePinnedCpu3D(int nx, int ny, int nz) const
            {
                HostPinnedBuffer3D<T> buf;
                buf.alloc(nx, ny, nz);
                return buf;
            }

            // ----------------------------------------------------------------
            // GPU 分配
            // allocateDevice3D        线性，vol buffer，pitch=nx*sizeof(T)
            // allocateDevice3DPitched pitched，投影纹理buffer，pitch=驱动对齐值
            // ----------------------------------------------------------------

            template<typename T>
            DeviceLinearBuffer3D<T> allocateDevice3D(
                int nx, int ny, int nz, int deviceId, bool zero = false) const
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

            template<typename T>
            DevicePitchedBuffer3D<T> allocateDevice3DPitched(
                int nx, int ny, int nz, int deviceId, bool zero = false) const
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

            // ----------------------------------------------------------------
            // Borrow
            // ----------------------------------------------------------------

            template<typename T>
            DeviceLinearBuffer3DBorrowed<T> borrowDevice3D(
                T* ptr, int nx, int ny, int nz,
                size_t pitchBytes, int deviceId = 0) const
            {
                return DeviceLinearBuffer3DBorrowed<T>(ptr, nx, ny, nz, pitchBytes, deviceId);
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


            // ----------------------------------------------------------------
            // upload3D 同步
            // ----------------------------------------------------------------
            template<typename T, typename SrcBuf>
            void upload3D(const DeviceLinearBuffer3D<T>& dst, const SrcBuf& src) const
            {
                if (!dst || !src)
                    throw std::invalid_argument("upload3D: null ptr");
                YK_CUDA_CHECK(cudaSetDevice(dst.deviceId()));
                YK_CUDA_CHECK(cudaMemcpy(
                    dst.data(), src.cdata(),
                    src.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
            }

            template<typename T, typename SrcBuf>
            void upload3D(const DevicePitchedBuffer3D<T>& dst, const SrcBuf& src) const
            {
                if (!dst || !src)
                    throw std::invalid_argument("upload3D: null ptr");
                YK_CUDA_CHECK(cudaSetDevice(dst.deviceId()));
                cudaMemcpy3DParms p = {};
                p.srcPtr = make_cudaPitchedPtr((void*)src.cdata(),
                    src.shape().nx * sizeof(T), src.shape().nx, src.shape().ny);
                p.dstPtr = make_cudaPitchedPtr((void*)dst.data(),
                    dst.pitch(), dst.shape().nx, dst.shape().ny);
                p.extent = make_cudaExtent(
                    src.shape().nx * sizeof(T), src.shape().ny, src.shape().nz);
                p.kind = cudaMemcpyHostToDevice;
                YK_CUDA_CHECK(cudaMemcpy3D(&p));
            }

            // ----------------------------------------------------------------
            // Upload 异步（HostPinnedBuffer3D → Device）
            // ----------------------------------------------------------------

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
                p.srcPtr = make_cudaPitchedPtr((void*)src.data(),
                    src.shape().nx * sizeof(T), src.shape().nx, src.shape().ny);
                p.dstPtr = make_cudaPitchedPtr((void*)dst.data(),
                    dst.pitch(), dst.shape().nx, dst.shape().ny);
                p.extent = make_cudaExtent(
                    src.shape().nx * sizeof(T), src.shape().ny, src.shape().nz);
                p.kind = cudaMemcpyHostToDevice;
                YK_CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
            }

            template<typename T>
            void upload3DAsyncSync(const DeviceLinearBuffer3D<T>& dst,
                const HostPinnedBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                upload3DAsync(dst, src, stream); sync(stream);
            }

            template<typename T>
            void upload3DAsyncSync(const DevicePitchedBuffer3D<T>& dst,
                const HostPinnedBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                upload3DAsync(dst, src, stream); sync(stream);
            }


            // ----------------------------------------------------------------
            // download3D 同步
            // ----------------------------------------------------------------
            template<typename T, typename DstBuf>
            void download3D(DstBuf& dst, const DeviceLinearBuffer3D<T>& src) const
            {
                if (!dst || !src)
                    throw std::invalid_argument("download3D: null ptr");
                YK_CUDA_CHECK(cudaSetDevice(src.deviceId()));
                YK_CUDA_CHECK(cudaMemcpy(
                    dst.data(), src.cdata(),
                    src.size() * sizeof(T),
                    cudaMemcpyDeviceToHost));
            }

            template<typename T, typename DstBuf>
            void download3D(DstBuf& dst, const DevicePitchedBuffer3D<T>& src) const
            {
                if (!dst || !src)
                    throw std::invalid_argument("download3D: null ptr");
                YK_CUDA_CHECK(cudaSetDevice(src.deviceId()));
                cudaMemcpy3DParms p = {};
                p.srcPtr = make_cudaPitchedPtr((void*)src.cdata(),
                    src.pitch(), src.shape().nx, src.shape().ny);
                p.dstPtr = make_cudaPitchedPtr((void*)dst.data(),
                    dst.shape().nx * sizeof(T), dst.shape().nx, dst.shape().ny);
                p.extent = make_cudaExtent(
                    dst.shape().nx * sizeof(T), dst.shape().ny, dst.shape().nz);
                p.kind = cudaMemcpyDeviceToHost;
                YK_CUDA_CHECK(cudaMemcpy3D(&p));
            }
            // ----------------------------------------------------------------
            // Download 异步（Device → HostPinnedBuffer3D）
            // ----------------------------------------------------------------

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
                p.srcPtr = make_cudaPitchedPtr((void*)src.cdata(),
                    src.pitch(), src.shape().nx, src.shape().ny);
                p.dstPtr = make_cudaPitchedPtr((void*)dst.data(),
                    dst.shape().nx * sizeof(T), dst.shape().nx, dst.shape().ny);
                p.extent = make_cudaExtent(
                    dst.shape().nx * sizeof(T), dst.shape().ny, dst.shape().nz);
                p.kind = cudaMemcpyDeviceToHost;
                YK_CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
            }

            template<typename T>
            void download3DAsyncSync(HostPinnedBuffer3D<T>& dst,
                const DeviceLinearBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                download3DAsync(dst, src, stream); sync(stream);
            }

            template<typename T>
            void download3DAsyncSync(HostPinnedBuffer3D<T>& dst,
                const DevicePitchedBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                download3DAsync(dst, src, stream); sync(stream);
            }
        };

        // ============================================================
        // PodDataController  —  1D POD/结构体数组分配与传输
        // 用于 geo参数、LUT、校正表等小数组
        // ============================================================
        class PodDataController {
        public:

            template<typename T>
            DeviceLinearBuffer<T> allocate(int n, int deviceId = 0) const
            {
                DeviceLinearBuffer<T> buf;
                buf.alloc(n, deviceId);
                return buf;
            }

            template<typename T>
            DeviceLinearBuffer<T> allocateAndUpload(
                const std::vector<T>& src, int deviceId = 0) const
            {
                auto buf = allocate<T>((int)src.size(), deviceId);
                upload(buf, src);
                return buf;
            }

            // 同步上传
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

            // 异步上传，src 必须是 HostPinnedBuffer
            template<typename T>
            void uploadAsync(const DeviceLinearBuffer<T>& dst,
                const HostPinnedBuffer<T>& src,
                int n, cudaStream_t stream) const
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

            // 同步下载
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

            // 异步下载，dst 必须是 HostPinnedBuffer，调用方负责 sync
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