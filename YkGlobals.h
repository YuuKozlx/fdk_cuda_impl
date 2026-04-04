#pragma once
#pragma once

// ============================================================
// CUDA Common Macros & Utilities
// ============================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector_types.h>



struct SDimensions3D {
    unsigned int iVX;
    unsigned int iVY;
    unsigned int iVZ;
    unsigned int iPAng;
    unsigned int iPU; // number of detectors in the U direction
    unsigned int iPV; // number of detectors in the V direction
};


struct SProjDims {
    int Nu = 0;
    int Nv = 0;
    int Ang = 0;

    static SProjDims from(const SDimensions3D& d) {
        return { (int)d.iPU, (int)d.iPV, (int)d.iPAng };
    }
};

struct SVolDims {
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;

    static SVolDims from(const SDimensions3D& d) {
        return { (int)d.iVX, (int)d.iVY, (int)d.iVZ };
    }
};



struct alignas(16) SConeProjectionVec {
    float3 src;      // tube params : source position (world);
    float3 srcCR; // tube params£ºcenter ray direction [unit vector];
    float3 detS;     // detector pixel (0,0) position (world)
    float3 detU;     // per-pixel vector in U direction (world), length = du
    float3 detV;    // per-pixel vector in V direction (world), length = dv
    float3 angle;   // angle.x : gantry rotation angle [rad], measured (encoder)
    // angle.y : reserved (e.g. angle velocity / jitter / 0)
    // angle.z : reserved
};

// ---------------------- launch policy ----------------------
struct SKernelLaunchPolicy {
    int block_threads = 256;   // warp-row: must be multiple of 32
    bool bounds_check = true;  // ÊÇ·ñ¼ì²é a in [0, Ang)
};
//
//struct SFDKGeoParamPerView
//{
//    float du_mm = 1.0f;     // |detU|
//    float dv_mm = 1.0f;     // |detV|
//
//    float offsetU_pix = 0.0f;
//    float offsetV_pix = 0.0f;
//    bool  offset_valid = true;
//
//    float theta = 0.0f;     // unwrapped atan2(src.x, -src.y)
//    float dtheta = 0.0f;    // >= eps
//
//    float SOD_mm = 0.0f;        // |src - isocenter|
//    float SDD_mm = 0.0f;
//};

struct alignas(16) SFDKGeoParamPerView
{
    float theta = 0.0f;
    float dtheta = 0.0f;

    // IMPORTANT: this stores SID by your definition (NOT classic SOD)
    float SOD_mm = 0.0f;   // == SID_mm
    float SDD_mm = 0.0f;

    int   Nu = 0;
    int   Nv = 0;

    float offsetU_pix = 0.0f;
    float offsetV_pix = 0.0f;

    float3 ray_center = { 0,0,0 };// from source to principal point (mm)

    float3 det_n = { 0,0,0 }; // detector plane normal (unit vector)


    float du_mm = 1.0f;
    float dv_mm = 1.0f;

    float UU = 0.0f;
    float VV = 0.0f;
    float UV = 0.0f;
    float invDetUV = 0.0f;
};







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

#ifndef YK_CUDA_KERNEL_CHECK
#define YK_CUDA_KERNEL_CHECK()                                                  \
    do {                                                                     \
        YK_CUDA_CHECK(cudaPeekAtLastError());                                   \
        YK_CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)
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



// ============================================================
// 1. CUDA error checking
// ============================================================

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr,                                                  \
                    "[CUDA ERROR] %s:%d\n  %s\n",                            \
                    __FILE__, __LINE__,                                      \
                    cudaGetErrorString(err));                                \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
#endif

#ifndef CUDA_KERNEL_CHECK
#define CUDA_KERNEL_CHECK()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaPeekAtLastError());                                   \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)
#endif

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

#ifndef CUDA_DIV_UP
#define CUDA_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#endif

// 1D launch
#define CUDA_LAUNCH_1D(kernel, n, block, ...)                                \
    do {                                                                     \
        dim3 _block(block);                                                  \
        dim3 _grid(CUDA_DIV_UP((n), _block.x));                              \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                              \
    } while (0)

// 2D launch
#define CUDA_LAUNCH_2D(kernel, nx, ny, blockx, blocky, ...)                  \
    do {                                                                     \
        dim3 _block(blockx, blocky);                                         \
        dim3 _grid(CUDA_DIV_UP((nx), _block.x),                               \
                   CUDA_DIV_UP((ny), _block.y));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                              \
    } while (0)

// 3D launch
#define CUDA_LAUNCH_3D(kernel, nx, ny, nz, bx, by, bz, ...)                  \
    do {                                                                     \
        dim3 _block(bx, by, bz);                                             \
        dim3 _grid(CUDA_DIV_UP((nx), _block.x),                               \
                   CUDA_DIV_UP((ny), _block.y),                               \
                   CUDA_DIV_UP((nz), _block.z));                             \
        kernel<<<_grid, _block>>>(__VA_ARGS__);                              \
    } while (0)

// ============================================================
// 5. Memory helpers
// ============================================================

#define CUDA_MALLOC(ptr, bytes)                                              \
    CUDA_CHECK(cudaMalloc((void**)&(ptr), (bytes)))

#define CUDA_FREE(ptr)                                                       \
    do {                                                                     \
        if ((ptr) != nullptr) {                                              \
            CUDA_CHECK(cudaFree(ptr));                                       \
            (ptr) = nullptr;                                                 \
        }                                                                    \
    } while (0)

#define CUDA_MEMCPY_H2D(dst, src, bytes)                                     \
    CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice))

#define CUDA_MEMCPY_D2H(dst, src, bytes)                                     \
    CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost))

#define CUDA_MEMSET(ptr, value, bytes)                                       \
    CUDA_CHECK(cudaMemset((ptr), (value), (bytes)))

// ============================================================
// 6. CUDA stream helpers
// ============================================================

#define CUDA_STREAM_CREATE(stream)                                           \
    CUDA_CHECK(cudaStreamCreate(&(stream)))

#define CUDA_STREAM_DESTROY(stream)                                          \
    CUDA_CHECK(cudaStreamDestroy((stream)))

#define CUDA_SYNC_STREAM(stream)                                             \
    CUDA_CHECK(cudaStreamSynchronize((stream)))

#define CUDA_SYNC_DEVICE()                                                   \
    CUDA_CHECK(cudaDeviceSynchronize())

// ============================================================
// 7. Device index helper
// ============================================================

inline void cuda_set_device(int device_id) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device_id < 0 || device_id >= count) {
        fprintf(stderr, "[CUDA ERROR] Invalid device id %d\n", device_id);
        std::abort();
    }
    CUDA_CHECK(cudaSetDevice(device_id));
}

// ============================================================
// 8. Debug helpers
// ============================================================

#ifndef CUDA_DEBUG_PRINT
#define CUDA_DEBUG_PRINT(fmt, ...)                                           \
    printf("[CUDA DEBUG] " fmt "\n", ##__VA_ARGS__)
#endif

