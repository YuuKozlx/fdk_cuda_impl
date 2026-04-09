#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include "YkGlobals.h" // 包含 YK_CUDA_CHECK 宏

// -------------------------- 条件编译 C++14 / C++20 --------------------------
#if __cplusplus >= 202002L
#define YK_NODISCARD [[nodiscard]]
#define YK_BYTE std::byte
#else
#define YK_NODISCARD
#define YK_BYTE unsigned char
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
// 一、同步 vs 异步传输的核心规则
// ─────────────────────────────────────────────────────────────
//
// 【同步传输】cudaMemcpy / cudaMemcpy3D / cudaMemcpyToSymbol
//
//   - host 端内存可以是任意类型：std::vector、裸指针、栈变量均可
//   - 函数返回时传输已完成，数据立即可用
//   - 适用场景：
//       · 一次性初始化（几何参数、校正表、LUT）
//       · 调试 dump（本来就要等结果）
//       · 不在热路径上的传输
//
// 【异步传输】cudaMemcpyAsync / cudaMemcpy3DAsync / cudaMemcpyToSymbolAsync
//
//   - host 端内存【必须】是 pinned memory（cudaMallocHost 分配）
//     普通堆内存（new / malloc / std::vector）用于异步传输是 UB：
//     函数立即返回，DMA 在后台进行，OS 随时可能换出物理页，
//     导致 DMA 读到垃圾数据，且不报错、不崩溃，只产生错误结果。
//   - 函数立即返回，传输在指定 stream 上异步执行
//   - 调用方必须保证 host 端内存在传输完成前不被修改或释放
//   - 适用场景：
//       · 热路径上需要与 GPU kernel 重叠执行的传输
//       · pipeline 中 CPU 准备下一帧数据同时 GPU 处理当前帧
//
// ─────────────────────────────────────────────────────────────
// 二、Pinned Memory 注意事项
// ─────────────────────────────────────────────────────────────
//
//   分配：cudaMallocHost(&ptr, size)
//   释放：cudaFreeHost(ptr)
//
//   - 物理页被 OS 锁定，不会被换出，GPU DMA 可直接访问
//   - 传输带宽高于普通内存（省去一次内部 staging 拷贝）
//   - 是稀缺资源，过度分配会压缩系统可用物理内存，影响 OS 调度
//   - 只在需要异步传输的成员变量上使用，临时局部变量不值得用
//
//   判断是否需要 pinned memory 的三个条件（缺一不可）：
//     1. host 端内存生命周期覆盖整个异步传输过程
//     2. 传输完成前 host 端数据不会被修改或释放
//     3. 有实际的异步重叠收益（CPU 和 GPU 真正并行）
//   三个条件不全满足时，直接用同步传输。
//
// ─────────────────────────────────────────────────────────────
// 三、Stream 使用规范
// ─────────────────────────────────────────────────────────────
//
//   【禁止使用默认参数 stream = 0】
//   所有接受 stream 的函数不提供默认值，调用方必须显式传入。
//   原因：忘传 stream 时悄悄退回 default stream，与其他非 0
//   stream 上的操作产生隐式同步 bug，极难排查。
//
//   【Default stream（stream = 0 / nullptr）的特殊语义】
//   - 提交到 default stream 之前，等待所有其他 stream 完成
//   - 其他 stream 提交之前，等待 default stream 完成
//   - 全程只用 default stream 时不存在同步问题，但失去并行性
//   - cudaStreamCreate 之后返回的是非 0 stream，不再是 default stream
//
//   【统一 stream 原则】
//   同一 pipeline 内所有操作（memcpy、kernel、symbol upload）
//   提交到同一个 stream，stream 内操作严格顺序执行，无需额外同步：
//
//     cudaMemcpyToSymbolAsync(sym, h_pinned, size, 0,
//                             cudaMemcpyHostToDevice, stream);
//     my_kernel<<<grid, block, 0, stream>>>(...);   // 一定在 symbol 上传后执行
//
//   跨 stream 操作需要显式同步点（cudaStreamSynchronize 或 cudaEvent）。
//
//   【同步点由调用方管理】
//   process() / upload() 等函数内部只负责向 stream 提交操作，
//   不在内部调用 cudaStreamSynchronize，由外层 pipeline 统一决定
//   在何处等待。例外：带 *Sync 后缀的函数在内部封装了 sync()，
//   适用于不需要重叠、但希望接口简单的场景。
//
// ─────────────────────────────────────────────────────────────
// 四、本文件接口命名约定
// ─────────────────────────────────────────────────────────────
//
//   upload / download
//     同步传输，host 端为普通内存（std::vector / CpuBuffer3D）
//     函数返回即完成，无需传入 stream
//
//   uploadAsync / downloadAsync
//     异步传输，host 端必须为 HostPinnedBuffer / HostPinnedBuffer3D
//     函数返回时传输仍在进行，调用方负责在合适位置 sync(stream)
//
//   uploadAsyncSync / downloadAsyncSync
//     异步传输 + 内部 sync，host 端必须为 pinned buffer
//     函数返回时传输已完成，调用方无需手动 sync
//     适用于需要 pinned 带宽但不需要重叠的场景
//
//   sync(stream)
//     所有需要等待 stream 完成的地方统一调用此函数，
//     不直接散用 cudaStreamSynchronize
//
// ─────────────────────────────────────────────────────────────
// 五、常见错误速查
// ─────────────────────────────────────────────────────────────
//
//   [错误] std::vector + cudaMemcpyAsync
//     → UB，改用 cudaMemcpy（同步）或改用 HostPinnedBuffer + Async
//
//   [错误] 局部变量 + cudaMemcpyToSymbolAsync
//     → 函数返回后局部变量析构，DMA 读野指针
//     → 改用 cudaMemcpyToSymbol（同步），或将 buffer 提升为成员变量
//
//   [错误] host 端 for 循环直接赋值 __constant__ 变量
//     → 只写了 host shadow copy，device 端不更新，kernel 读到全 0
//     → 必须通过 cudaMemcpyToSymbol / cudaMemcpyToSymbolAsync 写入
//
//   [错误] cudaMemcpy（default stream）+ kernel（非 0 stream）之间无同步
//     → 两者在不同 stream 上，无隐式顺序保证
//     → 统一到同一 stream，或在两者之间插入 cudaStreamSynchronize
//
//   [错误] process() 内部调用 cudaStreamSynchronize 阻塞整个 pipeline
//     → 破坏流水线并行性
//     → sync 点交给调用方，process() 只提交操作
//
// ============================================================

namespace YK {
    namespace Mem {

        // -------------------------- Shape / Region --------------------------
        struct Shape3D { int nx = 0, ny = 0, nz = 0; };
        struct Region3D { int ox = 0, oy = 0, oz = 0; int sx = 0, sy = 0, sz = 0; };
        enum class OwnerTag : uint8_t { Owning, Borrowed };

        // -------------------------- CPU Buffer --------------------------
        template<typename T>
        struct CpuView3D {
            T* ptr = nullptr;
            int nx = 0, ny = 0, nz = 0;
            size_t sliceStride = 0;

            explicit operator bool() const noexcept { return ptr != nullptr; }
            T* at(int x, int y, int z) const noexcept { return ptr + size_t(z) * sliceStride + size_t(y) * nx + x; }

#if __cplusplus >= 202002L
            // C++20 写法
            auto operator()(int x, int y, int z) const noexcept {
                if constexpr (std::is_const_v<T>)
                    return *at(x, y, z); // 返回 const T&
                else
                    return *at(x, y, z); // 返回 T&
            }
#else
            // C++14 写法
            typename std::conditional<std::is_const<T>::value, const T&, T&>::type
                operator()(int x, int y, int z) const noexcept {
                return *at(x, y, z);
            }
#endif
        };

        template<typename T>
        class CpuBuffer3D {
        public:
            using ValueType = T;
            CpuBuffer3D() = default;

            // Owning 构造函数
            CpuBuffer3D(int nx, int ny, int nz, bool zero = false)
                : sh_{ nx, ny, nz }, sliceStride_(size_t(nx)* ny)
            {
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("CpuBuffer3D: nx/ny/nz must > 0");

                ptr_ = std::make_unique<T[]>(size_t(nx) * ny * nz);
                if (zero)
                    std::memset(ptr_.get(), 0, sizeof(T) * size_t(nx) * ny * nz);
            }

            // 禁用拷贝
            CpuBuffer3D(const CpuBuffer3D&) = delete;
            CpuBuffer3D& operator=(const CpuBuffer3D&) = delete;

            // 支持移动
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

            CpuView3D<T> view() { return { ptr_.get(), sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }
            CpuView3D<const T> cview() const { return { ptr_.get(), sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }

            Shape3D shape() const noexcept { return sh_; }
            T* data() noexcept { return ptr_.get(); }
            uint64_t size() const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            const T* cdata() const noexcept { return ptr_.get(); }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            std::unique_ptr<T[]> ptr_;
            Shape3D sh_{};
            size_t sliceStride_ = 0;
        };

        template<typename T>
        class HostPinnedBuffer3D {
        public:
            HostPinnedBuffer3D() = default;
            ~HostPinnedBuffer3D() { reset(); }

            HostPinnedBuffer3D(const HostPinnedBuffer3D&) = delete;
            HostPinnedBuffer3D& operator=(const HostPinnedBuffer3D&) = delete;
            HostPinnedBuffer3D(HostPinnedBuffer3D&& o) noexcept {
                ptr_ = o.ptr_; o.ptr_ = nullptr;
                sh_ = o.sh_;  o.sh_ = {};
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
            Shape3D sh_ = {};
        };



        template<typename T>
        class CpuBuffer3DBorrowed {
        public:
            using ValueType = T;

            // 构造函数：必须提供外部指针和尺寸
            CpuBuffer3DBorrowed(T* externalPtr, int nx, int ny, int nz)
                : ptr_(externalPtr), sh_{ nx, ny, nz }, sliceStride_(size_t(nx)* ny)
            {
                if (!externalPtr)
                    throw std::invalid_argument("CpuBuffer3DBorrowed: externalPtr is null");
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("CpuBuffer3DBorrowed: nx/ny/nz must > 0");
            }

            // 禁用拷贝
            CpuBuffer3DBorrowed(const CpuBuffer3DBorrowed&) = delete;
            CpuBuffer3DBorrowed& operator=(const CpuBuffer3DBorrowed&) = delete;

            // 支持移动
            CpuBuffer3DBorrowed(CpuBuffer3DBorrowed&& o) noexcept
                : ptr_(o.ptr_), sh_(o.sh_), sliceStride_(o.sliceStride_)
            {
                o.ptr_ = nullptr;
                o.sh_ = {};
                o.sliceStride_ = 0;
            }

            CpuBuffer3DBorrowed& operator=(CpuBuffer3DBorrowed&& o) noexcept {
                if (this != &o) {
                    ptr_ = o.ptr_;
                    sh_ = o.sh_;
                    sliceStride_ = o.sliceStride_;
                    o.ptr_ = nullptr;
                    o.sh_ = {};
                    o.sliceStride_ = 0;
                }
                return *this;
            }

            // view / cview
            CpuView3D<T> view() { return { ptr_, sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }
            CpuView3D<const T> cview() const { return { ptr_, sh_.nx, sh_.ny, sh_.nz, sliceStride_ }; }

            // 数据访问
            T* data() noexcept { return ptr_; }
            const T* cdata() const noexcept { return ptr_; }

            // 形状
            Shape3D shape() const noexcept { return sh_; }
            uint64_t size() const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }

            // 转 bool
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_;             // 外部内存指针
            Shape3D sh_;         // 尺寸
            size_t sliceStride_; // 每个 z 层的跨度
        };

        // -------------------------- Device Buffer --------------------------
        template<typename T>
        struct DeviceView3D {
            T* ptr = nullptr;
            int nx = 0, ny = 0, nz = 0;
            size_t pitchBytes = 0, sliceBytes = 0;

            explicit operator bool() const noexcept { return ptr != nullptr; }

            YK_HD T* at(int x, int y, int z) const noexcept {
                auto byteOffset = size_t(z) * sliceBytes + size_t(y) * pitchBytes;
                return reinterpret_cast<T*>(reinterpret_cast<YK_BYTE*>(ptr) + byteOffset) + x;
            }

#if __cplusplus >= 202002L
            YK_HD auto operator()(int x, int y, int z) const noexcept {
                if constexpr (std::is_const_v<T>)
                    return *at(x, y, z);
                else
                    return *at(x, y, z);
            }
#else
            YK_HD typename std::conditional<std::is_const<T>::value, const T&, T&>::type
                operator()(int x, int y, int z) const noexcept {
                return *at(x, y, z);
            }
#endif
        };


        template<typename T>
        class DeviceBuffer3D {
        public:
            using ValueType = T;
            DeviceBuffer3D() = default;
            ~DeviceBuffer3D() { reset_noexcept(); }
            DeviceBuffer3D(const DeviceBuffer3D&) = delete;
            DeviceBuffer3D& operator=(const DeviceBuffer3D&) = delete;
            DeviceBuffer3D(DeviceBuffer3D&& o) noexcept { move_from(o); }
            DeviceBuffer3D& operator=(DeviceBuffer3D&& o) noexcept { if (this != &o) { reset_noexcept(); move_from(o); } return *this; }

            YK_NODISCARD DeviceView3D<T> view() const noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, sliceBytes_ }; }
            YK_NODISCARD DeviceView3D<const T> cview() const noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, sliceBytes_ }; }

            Shape3D shape() const noexcept { return sh_; }
            size_t pitch() const noexcept { return pitchBytes_; }
            size_t slice() const noexcept { return sliceBytes_; }
            uint64_t size() const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            OwnerTag owner() const noexcept { return owner_; }
            int deviceId() const noexcept { return deviceId_; }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }
            T* data() const noexcept { return ptr_; }
            const T* cdata() const { return ptr_; }

            void reset() noexcept { reset_noexcept(); }

        private:
            friend class MemoryController;

            static DeviceBuffer3D make_owning(T* ptr, Shape3D sh, size_t pitchBytes, int deviceId) {
                return DeviceBuffer3D{ ptr, sh, pitchBytes, pitchBytes * sh.ny, OwnerTag::Owning, deviceId };
            }

            // 新增：借用外部指针，不持有所有权
            static DeviceBuffer3D make_borrowed(T* ptr, Shape3D sh, size_t pitchBytes, int deviceId) {
                return DeviceBuffer3D{ ptr, sh, pitchBytes, pitchBytes * sh.ny, OwnerTag::Borrowed, deviceId };
            }

            void reset_noexcept() noexcept {
                if (owner_ == OwnerTag::Owning && ptr_) {
                    YK_CUDA_CHECK(cudaSetDevice(deviceId_));
                    YK_CUDA_CHECK(cudaFree(ptr_));
                }
                ptr_ = nullptr; sh_ = {}; pitchBytes_ = sliceBytes_ = 0; owner_ = OwnerTag::Borrowed; deviceId_ = 0;
            }

            void move_from(DeviceBuffer3D& o) noexcept {
                ptr_ = o.ptr_; o.ptr_ = nullptr;
                sh_ = o.sh_; o.sh_ = {};
                pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
                sliceBytes_ = o.sliceBytes_; o.sliceBytes_ = 0;
                owner_ = o.owner_; o.owner_ = OwnerTag::Borrowed;
                deviceId_ = o.deviceId_; o.deviceId_ = 0;
            }

            DeviceBuffer3D(T* ptr, Shape3D sh, size_t pitchBytes, size_t sliceBytes, OwnerTag owner, int deviceId)
                : ptr_(ptr), sh_(sh), pitchBytes_(pitchBytes), sliceBytes_(sliceBytes), owner_(owner), deviceId_(deviceId) {
            }

            T* ptr_ = nullptr;
            Shape3D sh_{};
            size_t pitchBytes_ = 0, sliceBytes_ = 0;
            OwnerTag owner_ = OwnerTag::Borrowed;
            int deviceId_ = 0;
        };


        template<typename T>
        class DeviceBuffer3DBorrowed {
        public:
            using ValueType = T;

            DeviceBuffer3DBorrowed(T* ptr, int nx, int ny, int nz,
                size_t pitchBytes, int deviceId = 0)
                : ptr_(ptr), sh_{ nx, ny, nz }
                , pitchBytes_(pitchBytes)
                , sliceBytes_(pitchBytes* ny)
                , deviceId_(deviceId)
            {
                if (!ptr)
                    throw std::invalid_argument("DeviceBuffer3DBorrowed: ptr is null");
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("DeviceBuffer3DBorrowed: nx/ny/nz>0");
            }

            // 线性指针便捷构造（pitch = nx*sizeof(T)）
            static DeviceBuffer3DBorrowed linear(T* ptr, int nx, int ny, int nz, int deviceId = 0) {
                return DeviceBuffer3DBorrowed(ptr, nx, ny, nz, nx * sizeof(T), deviceId);
            }

            // 禁用拷贝
            DeviceBuffer3DBorrowed(const DeviceBuffer3DBorrowed&) = delete;
            DeviceBuffer3DBorrowed& operator=(const DeviceBuffer3DBorrowed&) = delete;

            // 支持移动
            DeviceBuffer3DBorrowed(DeviceBuffer3DBorrowed&& o) noexcept
                : ptr_(o.ptr_), sh_(o.sh_)
                , pitchBytes_(o.pitchBytes_), sliceBytes_(o.sliceBytes_)
                , deviceId_(o.deviceId_)
            {
                o.ptr_ = nullptr; o.sh_ = {};
                o.pitchBytes_ = o.sliceBytes_ = 0;
            }

            DeviceBuffer3DBorrowed& operator=(DeviceBuffer3DBorrowed&& o) noexcept {
                if (this != &o) {
                    ptr_ = o.ptr_;        o.ptr_ = nullptr;
                    sh_ = o.sh_;         o.sh_ = {};
                    pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
                    sliceBytes_ = o.sliceBytes_; o.sliceBytes_ = 0;
                    deviceId_ = o.deviceId_;   o.deviceId_ = 0;
                }
                return *this;
            }

            ~DeviceBuffer3DBorrowed() = default;   // 不 cudaFree，外部管理

            DeviceView3D<T>       view()  const noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, sliceBytes_ }; }
            DeviceView3D<const T> cview() const noexcept { return { ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, sliceBytes_ }; }

            Shape3D      shape()    const noexcept { return sh_; }
            uint64_t size() const noexcept { return uint64_t(sh_.nx) * sh_.ny * sh_.nz; }
            size_t       pitch()    const noexcept { return pitchBytes_; }
            size_t       slice()    const noexcept { return sliceBytes_; }
            int          deviceId() const noexcept { return deviceId_; }
            T* data()     const noexcept { return ptr_; }
            const T* cdata()    const noexcept { return ptr_; }
            explicit operator bool()const noexcept { return ptr_ != nullptr; }

        private:
            T* ptr_ = nullptr;
            Shape3D sh_ = {};
            size_t  pitchBytes_ = 0;
            size_t  sliceBytes_ = 0;
            int     deviceId_ = 0;
        };

        // -------------------------- MemoryController --------------------------
        class MemoryController {
        public:
            void setDevice(int id) const { YK_CUDA_CHECK(cudaSetDevice(id)); }

            // ----------------------------------------------------------------
            // 底层 sync，统一调用
            // ----------------------------------------------------------------
            void sync(cudaStream_t stream) const
            {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // ---------------- CPU Buffers ----------------

            template<typename T>
            CpuBuffer3D<T> allocateCpu3D(int nx, int ny, int nz, bool zero = false) const
            {
                return CpuBuffer3D<T>(nx, ny, nz, zero);
            }

            // ---------------- Pinned CPU Buffers ----------------

            template<typename T>
            HostPinnedBuffer3D<T> allocatePinnedCpu3D(int nx, int ny, int nz) const
            {
                HostPinnedBuffer3D<T> buf;
                buf.alloc(nx, ny, nz);
                return buf;
            }

            // ---------------- GPU Buffers ----------------

            template<typename T>
            DeviceBuffer3D<T> allocateDevice3D(
                int nx, int ny, int nz,
                int deviceId,
                bool zero = false) const
            {
                if (nx <= 0 || ny <= 0 || nz <= 0)
                    throw std::invalid_argument("allocateDevice3D: nx/ny/nz>0");

                cudaSetDevice(deviceId);
                cudaExtent extent = make_cudaExtent(nx * sizeof(T), ny, nz);
                cudaPitchedPtr dptr;
                YK_CUDA_CHECK(cudaMalloc3D(&dptr, extent));

                if (zero) {
                    YK_CUDA_CHECK(cudaMemset3D(dptr, 0, extent));
                }

                return DeviceBuffer3D<T>::make_owning(
                    (T*)dptr.ptr, { nx, ny, nz }, dptr.pitch, deviceId);
            }

            // ---------------- Upload（同步，普通 CPU buffer）----------------

            template<typename T, typename Buf>
            void upload3D(const DeviceBuffer3D<T>& dst,
                const Buf& src) const         // 无 stream，同步
            {
                if (!dst || !src)
                    throw std::invalid_argument("upload3D: null ptr");

                cudaSetDevice(dst.deviceId());
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

                YK_CUDA_CHECK(cudaMemcpy3D(&p));    // 同步版本
            }

            // ---------------- Upload（异步，需要 pinned CPU buffer）----------------

            template<typename T>
            void upload3DAsync(const DeviceBuffer3D<T>& dst,
                const HostPinnedBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                if (!dst || !src)
                    throw std::invalid_argument("upload3DAsync: null ptr");

                cudaSetDevice(dst.deviceId());
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
            void upload3DAsyncSync(const DeviceBuffer3D<T>& dst,
                const HostPinnedBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                upload3DAsync(dst, src, stream);
                sync(stream);
            }

            // ---------------- Download（同步，普通 CPU buffer）----------------

            template<typename T, typename Buf>
            void download3D(Buf& dst,
                const DeviceBuffer3D<T>& src) const    // 无 stream，同步
            {
                if (!dst || !src)
                    throw std::invalid_argument("download3D: null ptr");

                cudaSetDevice(src.deviceId());
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

                YK_CUDA_CHECK(cudaMemcpy3D(&p));    // 同步版本
            }

            // ---------------- Download（异步，需要 pinned CPU buffer）----------------

            template<typename T>
            void download3DAsync(HostPinnedBuffer3D<T>& dst,
                const DeviceBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                if (!dst || !src)
                    throw std::invalid_argument("download3DAsync: null ptr");

                cudaSetDevice(src.deviceId());
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
                const DeviceBuffer3D<T>& src,
                cudaStream_t stream) const
            {
                download3DAsync(dst, src, stream);
                sync(stream);
            }

            // ---------------- Borrow ----------------

            template<typename T>
            DeviceBuffer3DBorrowed<T> borrowDevice3D(
                T* ptr, int nx, int ny, int nz,
                size_t pitchBytes, int deviceId) const
            {
                return DeviceBuffer3DBorrowed<T>(ptr, nx, ny, nz, pitchBytes, deviceId);
            }

            template<typename T>
            DeviceBuffer3DBorrowed<T> borrowDevice3DLinear(
                T* ptr, int nx, int ny, int nz, int deviceId) const
            {
                return DeviceBuffer3DBorrowed<T>::linear(ptr, nx, ny, nz, deviceId);
            }

            template<typename T>
            CpuBuffer3DBorrowed<T> borrowCpu3D(
                T* ptr, int nx, int ny, int nz) const
            {
                return CpuBuffer3DBorrowed<T>(ptr, nx, ny, nz);
            }

            //template<typename T>
            //DeviceBuffer3D<T> borrowDevice3D(
            //    T* ptr, int nx, int ny, int nz,
            //    size_t pitchBytes, int deviceId = 0) const
            //{
            //    if (!ptr) throw std::invalid_argument("borrowDevice3D: ptr is null");
            //    if (nx <= 0 || ny <= 0 || nz <= 0)
            //        throw std::invalid_argument("borrowDevice3D: nx/ny/nz>0");
            //    return DeviceBuffer3D<T>::make_borrowed(ptr, { nx, ny, nz }, pitchBytes, deviceId);
            //}

            //// 借用外部线性指针（cudaMalloc 分配的，pitch = nx*sizeof(T)）
            //template<typename T>
            //DeviceBuffer3D<T> borrowDevice3DLinear(
            //    T* ptr, int nx, int ny, int nz,
            //    int deviceId = 0) const
            //{
            //    if (!ptr) throw std::invalid_argument("borrowDevice3DLinear: ptr is null");
            //    if (nx <= 0 || ny <= 0 || nz <= 0)
            //        throw std::invalid_argument("borrowDevice3DLinear: nx/ny/nz>0");
            //    const size_t pitchBytes = nx * sizeof(T);
            //    return DeviceBuffer3D<T>::make_borrowed(ptr, { nx, ny, nz }, pitchBytes, deviceId);
            //}
        };



    } // namespace Util

    namespace Mem {
        // 线性设备缓冲，专门给结构体/POD 数组用
        template<typename T>
        class DeviceLinearBuffer {
        public:
            DeviceLinearBuffer() = default;
            ~DeviceLinearBuffer() { reset(); }

            DeviceLinearBuffer(const DeviceLinearBuffer&) = delete;
            DeviceLinearBuffer& operator=(const DeviceLinearBuffer&) = delete;
            DeviceLinearBuffer(DeviceLinearBuffer&& o) noexcept {
                ptr_ = o.ptr_; o.ptr_ = nullptr;
                n_ = o.n_;   o.n_ = 0;
                deviceId_ = o.deviceId_; o.deviceId_ = 0;
            }
            DeviceLinearBuffer& operator=(DeviceLinearBuffer&& o) noexcept {
                if (this != &o) {
                    reset(); ptr_ = o.ptr_; o.ptr_ = nullptr;
                    n_ = o.n_; o.n_ = 0; deviceId_ = o.deviceId_;
                }
                return *this;
            }

            void alloc(int n, int deviceId = 0) {
                reset();
                deviceId_ = deviceId;
                n_ = n;
                cudaSetDevice(deviceId_);
                YK_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            }

            void reset() {
                if (ptr_) { cudaSetDevice(deviceId_); cudaFree(ptr_); ptr_ = nullptr; }
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
    // HostPinnedBuffer：pinned memory，专供异步传输使用
    // ============================================================
        template<typename T>
        class HostPinnedBuffer {
        public:
            HostPinnedBuffer() = default;
            ~HostPinnedBuffer() { reset(); }

            HostPinnedBuffer(const HostPinnedBuffer&) = delete;
            HostPinnedBuffer& operator=(const HostPinnedBuffer&) = delete;
            HostPinnedBuffer(HostPinnedBuffer&& o) noexcept {
                ptr_ = o.ptr_; o.ptr_ = nullptr;
                n_ = o.n_;   o.n_ = 0;
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

            // 从 std::vector 拷贝到 pinned buffer
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
            // PodDataController
            // ============================================================
        class PodDataController {
        public:

            template<typename T>
            DeviceLinearBuffer<T> allocate(int n, int deviceId = 0) const {
                DeviceLinearBuffer<T> buf;
                buf.alloc(n, deviceId);
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
                    throw std::invalid_argument("upload: null or empty");
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
                    throw std::invalid_argument("upload: null ptr");
                YK_CUDA_CHECK(cudaMemcpy(
                    dst.data(), src,
                    n * sizeof(T),
                    cudaMemcpyHostToDevice));
            }

            // ----------------------------------------------------------------
            // 异步上传：src 必须是 HostPinnedBuffer，调用方管理生命周期
            // ----------------------------------------------------------------
            template<typename T>
            void uploadAsync(const DeviceLinearBuffer<T>& dst,
                const HostPinnedBuffer<T>& src,
                int n,
                cudaStream_t stream) const
            {
                if (!src || !dst)
                    throw std::invalid_argument("uploadAsync: null ptr");
                if (n > src.count())
                    throw std::invalid_argument("uploadAsync: n > src.count()");
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    dst.data(), src.data(),
                    n * sizeof(T),
                    cudaMemcpyHostToDevice, stream));
            }

            // ----------------------------------------------------------------
            // 同步下载：dst 是普通 vector，安全
            // ----------------------------------------------------------------
            template<typename T>
            void download(std::vector<T>& dst,
                const DeviceLinearBuffer<T>& src) const
            {
                if (!src)
                    throw std::invalid_argument("download: null src");
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
                    throw std::invalid_argument("downloadAsync: null ptr");
                if (src.count() > dst.count())
                    throw std::invalid_argument("downloadAsync: dst too small");
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    dst.data(), src.data(),
                    src.count() * sizeof(T),
                    cudaMemcpyDeviceToHost, stream));
                // 不在这里 sync，调用方在合适的地方 cudaStreamSynchronize
            }

            // ----------------------------------------------------------------
            // 便捷函数：alloc + 同步上传一步完成
            // ----------------------------------------------------------------
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
            // 底层 sync，所有需要同步的地方统一调用这里
            // ----------------------------------------------------------------
            void sync(cudaStream_t stream) const
            {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

        };
    }
} // namespace YK