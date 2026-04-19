#pragma once
#include <cuda_runtime.h>

#define YK_INLINE inline

// ============================================================
// 1. CUDA / Host / Device 修饰符
// ============================================================

#ifdef __CUDACC__
#  define YK_HD            __host__ __device__
#  define YK_DEVICE        __device__
#  define YK_FORCE_INLINE  __forceinline__
#else
#  define YK_HD
#  define YK_DEVICE
#  define YK_FORCE_INLINE  inline
#endif

#ifndef HD
#  define HD     __host__ __device__
#endif
#ifndef DEVICE
#  define DEVICE __device__
#endif
#ifndef HOST
#  define HOST   __host__
#endif

// ============================================================
// 2. 错误检查宏
// ============================================================

#ifndef YK_CUDA_CHECK
#  define YK_CUDA_CHECK(x)                                                      \
     do {                                                                        \
         cudaError_t err__ = (x);                                               \
         if (err__ != cudaSuccess) {                                             \
             std::printf("[CUDA] %s at %s:%d\n",                                \
                 cudaGetErrorString(err__), __FILE__, __LINE__);                 \
             std::abort();                                                       \
         }                                                                       \
     } while (0)
#endif

#ifndef YK_CUFFT_CHECK
#  include <cufft.h>
#  define YK_CUFFT_CHECK(x)                                                     \
     do {                                                                        \
         cufftResult err__ = (x);                                               \
         if (err__ != CUFFT_SUCCESS) {                                           \
             std::printf("[CUFFT] error %d at %s:%d\n",                         \
                 (int)err__, __FILE__, __LINE__);                                \
             std::abort();                                                       \
         }                                                                       \
     } while (0)
#endif

// ============================================================
// 3. Kernel 启动后错误检查
//    YK_DEBUG 模式：额外做 DeviceSync（精确定位 kernel 错误）
// ============================================================

//#define YK_DEBUG

#ifndef YK_CUDA_KERNEL_CHECK
#  ifdef YK_DEBUG
#    define YK_CUDA_KERNEL_CHECK()                                              \
       do {                                                                      \
           auto err = cudaPeekAtLastError();                                     \
           if (err != cudaSuccess) { YK_CUDA_CHECK(err); }                      \
           YK_CUDA_CHECK(cudaDeviceSynchronize());                               \
       } while (0)
#  else
#    define YK_CUDA_KERNEL_CHECK()                                              \
       do {                                                                      \
           YK_CUDA_CHECK(cudaPeekAtLastError());                                 \
       } while (0)
#  endif
#endif

// ============================================================
// 4. Assert（仅 Debug 构建生效）
// ============================================================

#ifndef YK_ASSERT
#  include <cstdio>
#  include <cstdlib>
#  if defined(_DEBUG) || defined(DEBUG)
#    define YK_ASSERT(cond)                                                     \
       do {                                                                      \
           if (!(cond)) {                                                        \
               std::fprintf(stderr, "[YK_ASSERT] %s:%d: %s\n",                  \
                   __FILE__, __LINE__, #cond);                                   \
               std::abort();                                                     \
           }                                                                     \
       } while (0)
#  else
#    define YK_ASSERT(cond) ((void)0)
#  endif
#endif

// ============================================================
// 5. 数学常量与角度转换
// ============================================================

#ifndef CUDA_PI
#  define CUDA_PI 3.14159265358979323846f
#endif
#ifndef DEG2RAD
#  define DEG2RAD(x) ((x) * CUDA_PI / 180.0f)
#endif
#ifndef RAD2DEG
#  define RAD2DEG(x) ((x) * 180.0f / CUDA_PI)
#endif

// ============================================================
// 6. 内存分配 / 释放
// ============================================================

// Device
#define YK_CUDA_MALLOC(ptr, bytes)                                              \
    YK_CUDA_CHECK(cudaMalloc((void**)&(ptr), (bytes)))

#define YK_CUDA_FREE(ptr)                                                       \
    do {                                                                        \
        if ((ptr) != nullptr) {                                                 \
            YK_CUDA_CHECK(cudaFree(ptr));                                       \
            (ptr) = nullptr;                                                    \
        }                                                                       \
    } while (0)

// Pinned host（析构/reset 路径请勿使用 YK_CUDA_CHECK 包 cudaFreeHost）
#define YK_CUDA_MALLOC_HOST(ptr, bytes)                                         \
    YK_CUDA_CHECK(cudaMallocHost((void**)&(ptr), (bytes)))

#define YK_CUDA_FREE_HOST(ptr)                                                  \
    do {                                                                        \
        if ((ptr) != nullptr) {                                                 \
            cudaFreeHost(ptr);   /* 析构路径，不抛异常，不 abort */              \
            (ptr) = nullptr;                                                    \
        }                                                                       \
    } while (0)

// ============================================================
// 7. 内存拷贝（同步）
// ============================================================

#define YK_CUDA_MEMCPY_H2D(dst, src, bytes)                                     \
    YK_CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice))

#define YK_CUDA_MEMCPY_D2H(dst, src, bytes)                                     \
    YK_CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost))

#define YK_CUDA_MEMCPY_D2D(dst, src, bytes)                                     \
    YK_CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToDevice))

// ============================================================
// 8. 内存拷贝（异步，src/dst 须为 pinned 或 device 内存）
// ============================================================

#define YK_CUDA_MEMCPY_H2D_ASYNC(dst, src, bytes, stream)                       \
    YK_CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes),                        \
        cudaMemcpyHostToDevice, (stream)))

#define YK_CUDA_MEMCPY_D2H_ASYNC(dst, src, bytes, stream)                       \
    YK_CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes),                        \
        cudaMemcpyDeviceToHost, (stream)))

#define YK_CUDA_MEMCPY_D2D_ASYNC(dst, src, bytes, stream)                       \
    YK_CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes),                        \
        cudaMemcpyDeviceToDevice, (stream)))

// ============================================================
// 9. Memset
// ============================================================

#define YK_CUDA_MEMSET(ptr, value, bytes)                                       \
    YK_CUDA_CHECK(cudaMemset((ptr), (value), (bytes)))

#define YK_CUDA_MEMSET_ASYNC(ptr, value, bytes, stream)                         \
    YK_CUDA_CHECK(cudaMemsetAsync((ptr), (value), (bytes), (stream)))

// ============================================================
// 10. Stream
// ============================================================

#define YK_CUDA_STREAM_CREATE(stream)                                           \
    YK_CUDA_CHECK(cudaStreamCreate(&(stream)))

#define YK_CUDA_STREAM_DESTROY(stream)                                          \
    YK_CUDA_CHECK(cudaStreamDestroy((stream)))

#define YK_CUDA_SYNC_STREAM(stream)                                             \
    YK_CUDA_CHECK(cudaStreamSynchronize((stream)))

#define YK_CUDA_SYNC_DEVICE()                                                   \
    YK_CUDA_CHECK(cudaDeviceSynchronize())

// ============================================================
// 11. Grid / Block 计算工具
// ============================================================

#ifndef YK_CUDA_DIV_UP
#  define YK_CUDA_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#endif

// ============================================================
// 12. Kernel 启动宏
//     无后缀：默认 stream（0）
//     _S 后缀：显式传入 stream（推荐）
// ============================================================

// --- 1D ---
#define YK_CUDA_LAUNCH_1D(kernel, n, block, ...)                                \
    do {                                                                        \
        dim3 _block(block);                                                     \
        dim3 _grid(YK_CUDA_DIV_UP((n), _block.x));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                                 \
    } while (0)

#define YK_CUDA_LAUNCH_1D_S(kernel, n, block, stream, ...)                      \
    do {                                                                        \
        dim3 _block(block);                                                     \
        dim3 _grid(YK_CUDA_DIV_UP((n), _block.x));                             \
        kernel<<<_grid, _block, 0, (stream)>>>(__VA_ARGS__);                    \
    } while (0)

// --- 2D ---
#define YK_CUDA_LAUNCH_2D(kernel, nx, ny, bx, by, ...)                          \
    do {                                                                        \
        dim3 _block(bx, by);                                                    \
        dim3 _grid(YK_CUDA_DIV_UP((nx), _block.x),                             \
                   YK_CUDA_DIV_UP((ny), _block.y));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                                 \
    } while (0)

#define YK_CUDA_LAUNCH_2D_S(kernel, nx, ny, bx, by, stream, ...)                \
    do {                                                                        \
        dim3 _block(bx, by);                                                    \
        dim3 _grid(YK_CUDA_DIV_UP((nx), _block.x),                             \
                   YK_CUDA_DIV_UP((ny), _block.y));                             \
        kernel<<<_grid, _block, 0, (stream)>>>(__VA_ARGS__);                    \
    } while (0)

// --- 3D ---
#define YK_CUDA_LAUNCH_3D(kernel, nx, ny, nz, bx, by, bz, ...)                  \
    do {                                                                        \
        dim3 _block(bx, by, bz);                                                \
        dim3 _grid(YK_CUDA_DIV_UP((nx), _block.x),                             \
                   YK_CUDA_DIV_UP((ny), _block.y),                             \
                   YK_CUDA_DIV_UP((nz), _block.z));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                                 \
    } while (0)

#define YK_CUDA_LAUNCH_3D_S(kernel, nx, ny, nz, bx, by, bz, stream, ...)        \
    do {                                                                        \
        dim3 _block(bx, by, bz);                                                \
        dim3 _grid(YK_CUDA_DIV_UP((nx), _block.x),                             \
                   YK_CUDA_DIV_UP((ny), _block.y),                             \
                   YK_CUDA_DIV_UP((nz), _block.z));                             \
        kernel<<<_grid, _block, 0, (stream)>>>(__VA_ARGS__);                    \
    } while (0)

// ============================================================
// 13. Device 初始化工具
// ============================================================

YK_INLINE void cuda_set_device(int device_id)
{
    int count = 0;
    YK_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device_id < 0 || device_id >= count) {
        std::fprintf(stderr, "[CUDA ERROR] Invalid device id %d\n", device_id);
        std::abort();
    }
    YK_CUDA_CHECK(cudaSetDevice(device_id));
}

// ============================================================
// 14. 日志宏（.cu 编译单元专用，fprintf/printf，无外部依赖）
//     .cpp 编译单元请使用 yk_log.h 中的 YK_LOGI 等宏
// ============================================================

#ifdef __CUDACC__

#  ifndef YK_LOG_TAG
#    define YK_LOG_TAG "YK"
#  endif

// 内部实现
#  define _YK_CU_LOG(level_str_, stream_, fmt_, ...)                            \
     do {                                                                        \
         std::fprintf((stream_), "[" level_str_ "][" YK_LOG_TAG "] "            \
             fmt_ "\n", ##__VA_ARGS__);                                         \
     } while (0)

#  define _YK_CU_LOG_LOC(level_str_, stream_, fmt_, ...)                        \
     _YK_CU_LOG(level_str_, stream_, "[%s:%d] " fmt_,                           \
         __FILE__, __LINE__, ##__VA_ARGS__)

// Host side of .cu
#  define YK_LOGT(fmt, ...) _YK_CU_LOG("T", stdout, fmt, ##__VA_ARGS__)
#  define YK_LOGD(fmt, ...) _YK_CU_LOG("D", stdout, fmt, ##__VA_ARGS__)
#  define YK_LOGI(fmt, ...) _YK_CU_LOG("I", stdout, fmt, ##__VA_ARGS__)
#  define YK_LOGW(fmt, ...) _YK_CU_LOG("W", stderr, fmt, ##__VA_ARGS__)
#  define YK_LOGE(fmt, ...) _YK_CU_LOG("E", stderr, fmt, ##__VA_ARGS__)
#  define YK_LOGC(fmt, ...) _YK_CU_LOG("C", stderr, fmt, ##__VA_ARGS__)

#  define YK_LOGE_LOC(fmt, ...) _YK_CU_LOG_LOC("E", stderr, fmt, ##__VA_ARGS__)
#  define YK_LOGC_LOC(fmt, ...) _YK_CU_LOG_LOC("C", stderr, fmt, ##__VA_ARGS__)

// Device kernel 内部
#  define YK_DEV_LOGD(fmt, ...) printf("[D][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#  define YK_DEV_LOGI(fmt, ...) printf("[I][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#  define YK_DEV_LOGW(fmt, ...) printf("[W][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#  define YK_DEV_LOGE(fmt, ...) printf("[E][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#endif // __CUDACC__