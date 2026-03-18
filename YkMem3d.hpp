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
            const T* cdata() const noexcept { return ptr_.get(); }
            explicit operator bool() const noexcept { return ptr_ != nullptr; }

        private:
            std::unique_ptr<T[]> ptr_;
            Shape3D sh_{};
            size_t sliceStride_ = 0;
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

        // -------------------------- MemoryController --------------------------
        class MemoryController {
        public:
            void setDevice(int id) const { YK_CUDA_CHECK(cudaSetDevice(id)); }

            // ---------------- CPU Buffers ----------------

            // allocate owning CPU buffer
            template<typename T>
            CpuBuffer3D<T> allocateCpu3D(int nx, int ny, int nz, bool zero = false) const {
                return CpuBuffer3D<T>(nx, ny, nz, zero);
            }

            // ---------------- GPU Buffers ----------------

            template<typename T>
            DeviceBuffer3D<T> allocateDevice3D(int nx, int ny, int nz, int deviceId = 0, bool zero = false, cudaStream_t stream = 0) const {
                if (nx <= 0 || ny <= 0 || nz <= 0) throw std::invalid_argument("allocateDevice3D: nx/ny/nz>0");

                cudaSetDevice(deviceId);
                cudaExtent extent = make_cudaExtent(nx * sizeof(T), ny, nz);
                cudaPitchedPtr dptr;
                YK_CUDA_CHECK(cudaMalloc3D(&dptr, extent));

                if (zero) {
                    YK_CUDA_CHECK(cudaMemset3DAsync(dptr, 0, extent, stream));
                }

                return DeviceBuffer3D<T>::make_owning((T*)dptr.ptr, { nx, ny, nz }, dptr.pitch, deviceId);
            }

            // ---------------- Upload / Download ----------------

            template<typename T, typename Buf>
            void upload3D(const DeviceBuffer3D<T>& dst, const Buf& src, cudaStream_t stream = 0) const {
                if (!dst || !src) throw std::invalid_argument("upload3D: null ptr");

                cudaSetDevice(dst.deviceId());
                cudaMemcpy3DParms copyParams = {};
                copyParams.srcPtr = make_cudaPitchedPtr((void*)src.cdata(), src.shape().nx * sizeof(T), src.shape().nx, src.shape().ny);
                copyParams.dstPtr = make_cudaPitchedPtr((void*)dst.data(), dst.pitch(), dst.shape().nx, dst.shape().ny);
                copyParams.extent = make_cudaExtent(src.shape().nx * sizeof(T), src.shape().ny, src.shape().nz);
                copyParams.kind = cudaMemcpyHostToDevice;

                YK_CUDA_CHECK(cudaMemcpy3DAsync(&copyParams, stream));
            }

            template<typename T, typename Buf>
            void download3D(Buf& dst, const DeviceBuffer3D<T>& src, cudaStream_t stream = 0) const {
                if (!dst || !src) throw std::invalid_argument("download3D: null ptr");

                cudaSetDevice(src.deviceId());
                cudaMemcpy3DParms copyParams = {};
                copyParams.srcPtr = make_cudaPitchedPtr((void*)src.cdata(), src.pitch(), src.shape().nx, src.shape().ny);
                copyParams.dstPtr = make_cudaPitchedPtr((void*)dst.data(), dst.shape().nx * sizeof(T), dst.shape().nx, dst.shape().ny);
                copyParams.extent = make_cudaExtent(dst.shape().nx * sizeof(T), dst.shape().ny, dst.shape().nz);
                copyParams.kind = cudaMemcpyDeviceToHost;

                YK_CUDA_CHECK(cudaMemcpy3DAsync(&copyParams, stream));
            }

            void sync(cudaStream_t stream = 0) const { YK_CUDA_CHECK(cudaStreamSynchronize(stream)); }
        };

    } // namespace Util
} // namespace YK