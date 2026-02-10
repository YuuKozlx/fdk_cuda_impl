#pragma once
// ============================================================
// YK::Util - Single-header CUDA 3D buffer + view + controller
// - GpuView3D<T>        : POD view for kernels (non-owning)
// - GpuBuffer3D<T>      : stores pointer + shape + pitch, RAII (owning or borrowed)
// - CudaMemoryController: alloc/wrap/upload/download/zero helpers
//
// Design notes (minimal & practical):
// - "3D" is stored as a pitched 2D allocation with height = ny * nz
//   so that: sliceBytes = pitchBytes * ny, and z-step = sliceBytes.
// - Upload/Download assume host data is contiguous in x-fastest order,
//   laid out as [z][y][x] with contiguous (nx*ny*nz) elements.
// - C++14 compatible.
// ============================================================

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include "YkGlobals.h"

namespace YK {
    namespace Util {


        // ---------------------------
        // Basic shape/region
        // ---------------------------
        struct Shape3D {
            int nx = 0;
            int ny = 0;
            int nz = 0;
        };

        // Optional ROI structure (not used by upload/download full, but handy to extend later)
        struct Region3D {
            int ox = 0, oy = 0, oz = 0;
            int sx = 0, sy = 0, sz = 0;
        };

        enum class OwnerTag : uint8_t {
            Owning,
            Borrowed
        };

        // ---------------------------
        // GpuView3D<T> - POD view (kernel-friendly)
        // ---------------------------
        template<typename T>
        struct GpuView3D {
            T* ptr = nullptr;
            int nx = 0, ny = 0, nz = 0;
            size_t pitchBytes = 0; // bytes per row (y-step)
            size_t sliceBytes = 0; // bytes per slice (z-step) = pitchBytes * ny

            __host__ __device__ explicit operator bool() const { return ptr != nullptr; }

            // x-fastest, then y by pitch, then z by sliceBytes
            __host__ __device__ T* at(int x, int y, int z) const {
                char* base = reinterpret_cast<char*>(ptr) + static_cast<size_t>(z) * sliceBytes + static_cast<size_t>(y) * pitchBytes;
                return reinterpret_cast<T*>(base) + x;
            }
        };

        // ---------------------------
        // GpuBuffer3D<T> - owns (or borrows) a pitched "3D" buffer
        // ---------------------------
        template<typename T>
        class GpuBuffer3D {
        public:
            GpuBuffer3D() = default;
            ~GpuBuffer3D() { reset_noexcept(); }

            GpuBuffer3D(const GpuBuffer3D&) = delete;
            GpuBuffer3D& operator=(const GpuBuffer3D&) = delete;

            GpuBuffer3D(GpuBuffer3D&& o) noexcept { move_from(o); }
            GpuBuffer3D& operator=(GpuBuffer3D&& o) noexcept {
                if (this != &o) { reset_noexcept(); move_from(o); }
                return *this;
            }

            // View for kernels / algorithms
            GpuView3D<T> view() const {
                return GpuView3D<T>{ ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, sliceBytes_ };
            }
            GpuView3D<const T> cview() const {
                return GpuView3D<const T>{ ptr_, sh_.nx, sh_.ny, sh_.nz, pitchBytes_, sliceBytes_ };
            }

            // Introspection
            Shape3D shape() const { return sh_; }
            size_t pitch_bytes() const { return pitchBytes_; }
            size_t slice_bytes() const { return sliceBytes_; }
            OwnerTag owner() const { return owner_; }
            explicit operator bool() const { return ptr_ != nullptr; }
            T* data() const { return ptr_; }

            // Manual reset (optional; RAII already frees for owning)
            void reset() { reset_noexcept(); }

        private:
            // Only controller can construct filled buffers
            friend class CudaMemoryController;

            static GpuBuffer3D make_owning(T* ptr, Shape3D sh, size_t pitchBytes) {
                GpuBuffer3D b;
                b.ptr_ = ptr;
                b.sh_ = sh;
                b.pitchBytes_ = pitchBytes;
                b.sliceBytes_ = pitchBytes * static_cast<size_t>(sh.ny);
                b.owner_ = OwnerTag::Owning;
                return b;
            }
            static GpuBuffer3D make_borrowed(T* ptr, Shape3D sh, size_t pitchBytes) {
                GpuBuffer3D b;
                b.ptr_ = ptr;
                b.sh_ = sh;
                b.pitchBytes_ = pitchBytes;
                b.sliceBytes_ = pitchBytes * static_cast<size_t>(sh.ny);
                b.owner_ = OwnerTag::Borrowed;
                return b;
            }

            void reset_noexcept() noexcept {
                if (owner_ == OwnerTag::Owning && ptr_) {
                    // cannot throw in noexcept; ignore error
                    cudaFree(ptr_);
                }
                ptr_ = nullptr;
                sh_ = {};
                pitchBytes_ = 0;
                sliceBytes_ = 0;
                owner_ = OwnerTag::Borrowed;
            }

            void move_from(GpuBuffer3D& o) noexcept {
                ptr_ = o.ptr_; o.ptr_ = nullptr;
                sh_ = o.sh_; o.sh_ = {};
                pitchBytes_ = o.pitchBytes_; o.pitchBytes_ = 0;
                sliceBytes_ = o.sliceBytes_; o.sliceBytes_ = 0;
                owner_ = o.owner_; o.owner_ = OwnerTag::Borrowed;
            }

        private:
            T* ptr_ = nullptr;
            Shape3D sh_{};
            size_t pitchBytes_ = 0;
            size_t sliceBytes_ = 0;
            OwnerTag owner_ = OwnerTag::Borrowed;
        };

        // ---------------------------
        // CudaMemoryController - alloc/wrap/copy helpers
        // ---------------------------
        class CudaMemoryController {
        public:
            // Select device (optional helper)
            void setDevice(int index) const {
                YK_CUDA_CHECK(cudaSetDevice(index));
            }

            // Allocate a "3D" pitched buffer:
            // We allocate a pitched 2D surface of size: width = nx*sizeof(T), height = ny*nz.
            // Layout: z slices are stacked vertically, each slice has ny rows.
            template<typename T>
            GpuBuffer3D<T> allocatePitched3D(int nx, int ny, int nz, bool zero = false, cudaStream_t stream = 0) const {
                if (nx <= 0 || ny <= 0 || nz <= 0) {
                    throw std::invalid_argument("allocatePitched3D: nx/ny/nz must be > 0");
                }

                T* dptr = nullptr;
                size_t pitch = 0;

                const size_t widthBytes = static_cast<size_t>(nx) * sizeof(T);
                const size_t heightRows = static_cast<size_t>(ny) * static_cast<size_t>(nz); // stacked slices

                YK_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&dptr), &pitch, widthBytes, heightRows));

                auto buf = GpuBuffer3D<T>::make_owning(dptr, Shape3D{ nx, ny, nz }, pitch);

                if (zero) {
                    // total bytes = pitch * heightRows
                    const size_t bytes = pitch * heightRows;
                    YK_UTIL_CUDA_CHECK(cudaMemsetAsync(dptr, 0, bytes, stream));
                }
                return buf;
            }

            // Wrap an external pitched pointer (borrowed; will NOT cudaFree)
            template<typename T>
            GpuBuffer3D<T> wrapPitched3D(T* dptr, int nx, int ny, int nz, size_t pitchBytes) const {
                if (!dptr) throw std::invalid_argument("wrapPitched3D: dptr is null");
                if (nx <= 0 || ny <= 0 || nz <= 0) throw std::invalid_argument("wrapPitched3D: nx/ny/nz must be > 0");
                if (pitchBytes < static_cast<size_t>(nx) * sizeof(T)) {
                    throw std::invalid_argument("wrapPitched3D: pitchBytes < nx*sizeof(T)");
                }
                return GpuBuffer3D<T>::make_borrowed(dptr, Shape3D{ nx, ny, nz }, pitchBytes);
            }

            // Zero entire buffer (async)
            template<typename T>
            void zero(const GpuBuffer3D<T>& buf, cudaStream_t stream = 0) const {
                auto sh = buf.shape();
                if (!buf) return;
                const size_t heightRows = static_cast<size_t>(sh.ny) * static_cast<size_t>(sh.nz);
                const size_t bytes = buf.pitch_bytes() * heightRows;
                YK_UTIL_CUDA_CHECK(cudaMemsetAsync(buf.data(), 0, bytes, stream));
            }

            // Upload contiguous host data (H->D), full buffer
            // Host layout: contiguous nx*ny*nz elements, x-fastest.
            template<typename T>
            void upload(const GpuBuffer3D<T>& dst, const T* hsrc, cudaStream_t stream = 0) const {
                if (!dst) throw std::invalid_argument("upload: dst is null");
                if (!hsrc) throw std::invalid_argument("upload: hsrc is null");

                const auto sh = dst.shape();
                const size_t widthBytes = static_cast<size_t>(sh.nx) * sizeof(T);
                const size_t heightRows = static_cast<size_t>(sh.ny) * static_cast<size_t>(sh.nz);

                // host is contiguous: host pitch = widthBytes
                YK_UTIL_CUDA_CHECK(cudaMemcpy2DAsync(
                    dst.data(), dst.pitch_bytes(),
                    hsrc, widthBytes,
                    widthBytes, heightRows,
                    cudaMemcpyHostToDevice, stream));
            }

            // Download contiguous host data (D->H), full buffer
            template<typename T>
            void download(T* hdst, const GpuBuffer3D<T>& src, cudaStream_t stream = 0) const {
                if (!src) throw std::invalid_argument("download: src is null");
                if (!hdst) throw std::invalid_argument("download: hdst is null");

                const auto sh = src.shape();
                const size_t widthBytes = static_cast<size_t>(sh.nx) * sizeof(T);
                const size_t heightRows = static_cast<size_t>(sh.ny) * static_cast<size_t>(sh.nz);

                YK_UTIL_CUDA_CHECK(cudaMemcpy2DAsync(
                    hdst, widthBytes,
                    src.data(), src.pitch_bytes(),
                    widthBytes, heightRows,
                    cudaMemcpyDeviceToHost, stream));
            }

            // (Optional) synchronize helper
            void sync(cudaStream_t stream = 0) const {
                YK_UTIL_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        };

    } // namespace Util
} // namespace YK

