#pragma once
#include <cuda_runtime.h>

#define YK_INLINE inline




#ifdef __CUDACC__
#define YK_HD __host__ __device__
#define YK_DEVICE __device__
#define YK_FORCE_INLINE __forceinline__
#else
#define YK_HD
#define YK_DEVICE
#define YK_FORCE_INLINE inline
#endif


#ifndef YK_CUDA_CHECK
#define YK_CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        std::printf("[CUDA] %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)
#endif

#ifndef YK_CUFFT_CHECK
#define YK_CUFFT_CHECK(x) do { \
    cufftResult err__ = (x); \
    if (err__ != CUFFT_SUCCESS) { \
        std::printf("[CUFFT] error %d at %s:%d\n", (int)err__, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)
#endif

//#define YK_DEBUG

#ifndef YK_CUDA_KERNEL_CHECK
#ifdef YK_DEBUG
#define YK_CUDA_KERNEL_CHECK()                                                  \
    do {                                                                        \
        auto err = cudaPeekAtLastError();                                       \
        if (err != cudaSuccess) {                                               \
            YK_CUDA_CHECK(err);                                                 \
        }                                                                       \
        YK_CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)
#else
#define YK_CUDA_KERNEL_CHECK()                                                  \
    do {                                                                        \
        YK_CUDA_CHECK(cudaPeekAtLastError());                                   \
    } while (0)
#endif
#endif

#ifndef YK_ASSERT
#include <cstdio>
#include <cstdlib>

#if defined(_DEBUG) || defined(DEBUG)
#define YK_ASSERT(cond)                                           \
            do {                                                         \
                if (!(cond)) {                                           \
                    std::fprintf(stderr,                                 \
                        "[YK_ASSERT] %s:%d: %s\n",                       \
                        __FILE__, __LINE__, #cond);                      \
                    std::abort();                                        \
                }                                                        \
            } while (0)
#else
#define YK_ASSERT(cond) ((void)0)
#endif
#endif



//#ifndef YK_LOGE
//#define YK_LOGE(fmt, ...) std::fprintf(stderr, "[YK][FilterProcessor][E] " fmt "\n", ##__VA_ARGS__)
//#endif
//#ifndef YK_LOGW
//#define YK_LOGW(fmt, ...) std::fprintf(stderr, "[YK][FilterProcessor][W] " fmt "\n", ##__VA_ARGS__)
//#endif
//#ifndef YK_LOGI
//#define YK_LOGI(fmt, ...) std::fprintf(stdout, "[YK][FilterProcessor][I] " fmt "\n", ##__VA_ARGS__)
//#endif

// ============================================================
// 2. CUDA math helpers
// ============================================================

#ifndef CUDA_PI
#define CUDA_PI 3.14159265358979323846f
#endif

#ifndef DEG2RAD
#define DEG2RAD(x) ((x) * CUDA_PI / 180.0f)
#endif

#ifndef RAD2DEG
#define RAD2DEG(x) ((x) * 180.0f / CUDA_PI)
#endif

// ============================================================
// 3. Device / host qualifiers
// ============================================================

#ifndef HD
#define HD __host__ __device__
#endif

#ifndef DEVICE
#define DEVICE __device__
#endif

#ifndef HOST
#define HOST __host__
#endif

// ============================================================
// 4. CUDA launch helpers
// ============================================================

#ifndef YK_CUDA_DIV_UP
#define YK_CUDA_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#endif

// 1D launch
#define YK_CUDA_LAUNCH_1D(kernel, n, block, ...)                                \
    do {                                                                     \
        dim3 _block(block);                                                  \
        dim3 _grid(YK_CUDA_DIV_UP((n), _block.x));                              \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                              \
    } while (0)

// 2D launch
#define YK_CUDA_LAUNCH_2D(kernel, nx, ny, blockx, blocky, ...)                  \
    do {                                                                     \
        dim3 _block(blockx, blocky);                                         \
        dim3 _grid(YK_CUDA_DIV_UP((nx), _block.x),                               \
                   YK_CUDA_DIV_UP((ny), _block.y));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                              \
    } while (0)

// 3D launch
#define YK_CUDA_LAUNCH_3D(kernel, nx, ny, nz, bx, by, bz, ...)                  \
    do {                                                                     \
        dim3 _block(bx, by, bz);                                             \
        dim3 _grid(YK_CUDA_DIV_UP((nx), _block.x),                               \
                   YK_CUDA_DIV_UP((ny), _block.y),                               \
                   YK_CUDA_DIV_UP((nz), _block.z));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                              \
    } while (0)

// ============================================================
// 5. Memory helpers
// ============================================================

#define YK_CUDA_MALLOC(ptr, bytes)                                              \
    YK_CUDA_CHECK(cudaMalloc((void**)&(ptr), (bytes)))

#define YK_CUDA_FREE(ptr)                                                       \
    do {                                                                     \
        if ((ptr) != nullptr) {                                              \
            YK_CUDA_CHECK(cudaFree(ptr));                                       \
            (ptr) = nullptr;                                                 \
        }                                                                    \
    } while (0)

#define YK_CUDA_MEMCPY_H2D(dst, src, bytes)                                     \
    YK_CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice))

#define YK_CUDA_MEMCPY_D2H(dst, src, bytes)                                     \
    YK_CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost))

#define YK_CUDA_MEMSET(ptr, value, bytes)                                       \
    YK_CUDA_CHECK(cudaMemset((ptr), (value), (bytes)))

// ============================================================
// 6. CUDA stream helpers
// ============================================================

#define YK_CUDA_STREAM_CREATE(stream)                                           \
    YK_CUDA_CHECK(cudaStreamCreate(&(stream)))

#define CUDA_STREAM_DESTROY(stream)                                          \
    YK_CUDA_CHECK(cudaStreamDestroy((stream)))

#define CUDA_SYNC_STREAM(stream)                                             \
    YK_CUDA_CHECK(cudaStreamSynchronize((stream)))

#define YK_CUDA_SYNC_DEVICE()                                                   \
    YK_CUDA_CHECK(cudaDeviceSynchronize())

// ============================================================
// 7. Device index helper
// ============================================================

YK_INLINE void cuda_set_device(int device_id) {
    int count = 0;
    YK_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device_id < 0 || device_id >= count) {
        fprintf(stderr, "[CUDA ERROR] Invalid device id %d\n", device_id);
        std::abort();
    }
    YK_CUDA_CHECK(cudaSetDevice(device_id));
}

// ============================================================
// 8. Debug helpers
// ============================================================

#ifndef YK_CUDA_DEBUG_PRINT
#define YK_CUDA_DEBUG_PRINT(fmt, ...)                                           \
    printf("[CUDA DEBUG] " fmt "\n", ##__VA_ARGS__)
#endif

