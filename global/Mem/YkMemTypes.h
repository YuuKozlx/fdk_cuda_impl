#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// -------------------------- 条件编译 C++14 / C++20 --------------------------
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

namespace YK {
namespace Mem {

// ============================================================
// 基础类型
// ============================================================

struct Shape3D  { int nx = 0, ny = 0, nz = 0; };
struct Region3D { int ox = 0, oy = 0, oz = 0; int sx = 0, sy = 0, sz = 0; };
enum class OwnerTag : uint8_t { Owning, Borrowed };

// ============================================================
// CpuView3D
// 轻量级非拥有视图，指向已分配的 CPU 内存
// sliceStride = nx * ny（无 padding，线性布局）
// ============================================================
template<typename T>
struct CpuView3D {
    T*     ptr         = nullptr;
    int    nx = 0, ny = 0, nz = 0;
    size_t sliceStride = 0;           // 每个 z 层的元素跨度

    explicit operator bool() const noexcept { return ptr != nullptr; }

    T* at(int x, int y, int z) const noexcept
    {
        return ptr + size_t(z) * sliceStride + size_t(y) * nx + x;
    }

#if __cplusplus >= 202002L
    auto operator()(int x, int y, int z) const noexcept -> decltype(*at(x,y,z))
    {
        return *at(x, y, z);
    }
#else
    typename std::conditional<std::is_const<T>::value, const T&, T&>::type
    operator()(int x, int y, int z) const noexcept
    {
        return *at(x, y, z);
    }
#endif
};

// ============================================================
// DeviceView3D
// 轻量级非拥有视图，指向已分配的 GPU 内存（支持 pitched 和线性）
// 线性布局时 pitchBytes = nx * sizeof(T)
// ============================================================
template<typename T>
struct DeviceView3D {
    T*     ptr        = nullptr;
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
    YK_HD auto operator()(int x, int y, int z) const noexcept -> decltype(*at(x,y,z))
    {
        return *at(x, y, z);
    }
#else
    YK_HD typename std::conditional<std::is_const<T>::value, const T&, T&>::type
    operator()(int x, int y, int z) const noexcept
    {
        return *at(x, y, z);
    }
#endif
};

} // namespace Mem
} // namespace YK
