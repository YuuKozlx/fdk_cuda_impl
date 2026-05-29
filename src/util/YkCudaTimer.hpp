#pragma once
#include "YkCpuProfiler.hpp"  // 复用 CpuTimer / ScopeLogger
#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace YK {
    namespace Util {
        struct CudaTimer {
            cudaEvent_t start, stop;
            cudaStream_t stream;
            const char* tag;

            CudaTimer(const char* tag, cudaStream_t stream = 0)
                : tag(tag), stream(stream)
            {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, stream);
            }

            ~CudaTimer()
            {
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, start, stop);
                //printf("[%s] %.3f ms\n", tag, ms);
                YK_LOGD("[CUDA] {}: {:.3f} ms\n", tag, ms);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        };

        // ----------------------------------------------------------------
// CudaTimer：已有，不变
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// CudaMemSnapshot：在构造/析构时各采一次显存，输出差值
// ----------------------------------------------------------------
        struct CudaMemSnapshot {
            const char* tag;
            size_t free_before, total_before;

            CudaMemSnapshot(const char* tag) : tag(tag)
            {
                cudaMemGetInfo(&free_before, &total_before);
                printf("[%s] VRAM before: free=%.2fGB total=%.2fGB\n",
                    tag,
                    free_before / 1e9,
                    total_before / 1e9);
            }

            ~CudaMemSnapshot()
            {
                size_t free_after, total_after;
                cudaMemGetInfo(&free_after, &total_after);
                const long long delta = (long long)free_before - (long long)free_after;
                printf("[%s] VRAM after:  free=%.2fGB  delta=+%.2fMB\n",
                    tag,
                    free_after / 1e9,
                    delta / 1e6);
            }
        };

        // ----------------------------------------------------------------
        // CudaTimerMem：Timer + MemSnapshot 合并版
        // ----------------------------------------------------------------
        struct CudaTimerMem {
            CudaTimer       timer;
            CudaMemSnapshot mem;

            CudaTimerMem(const char* tag, cudaStream_t stream = 0)
                : timer(tag, stream), mem(tag)
            {
            }
        };


        struct CudaStreamGuard {
            cudaStream_t stream = nullptr;

            CudaStreamGuard() {
                YK_CUDA_CHECK(cudaStreamCreate(&stream));
            }

            ~CudaStreamGuard() noexcept {
                if (stream) {
                    cudaError_t err = cudaStreamDestroy(stream);
                    if (err != cudaSuccess)
                        std::fprintf(stderr, "[CUDA] cudaStreamDestroy failed: %s\n",
                            cudaGetErrorString(err));
                    stream = nullptr;
                }
            }

            CudaStreamGuard(const CudaStreamGuard&) = delete;
            CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;
            CudaStreamGuard(CudaStreamGuard&&) = delete;
            CudaStreamGuard& operator=(CudaStreamGuard&&) = delete;

            operator cudaStream_t() const noexcept { return stream; }
        };


        // ============================================================
        // CudaEventTimer  —  GPU 侧精确计时，基于 cudaEvent_t
        // 用法：
        //   CudaEventTimer t;
        //   t.start(stream);
        //   kernel<<<...>>>(stream);
        //   t.stop(stream);
        //   float ms = t.elapsed_ms();  // 隐含 sync
        // usage:
        // CudaEventTimer t;
        // t.start(stream);
        // launchFilterKernel << <grid, block, 0, stream >> > (...);
        // t.stop(stream);
        // YK_LOGI("filter: {:.3f} ms", t.elapsed_ms());

        // t.start(stream);
        // launchBackprojKernel << <grid, block, 0, stream >> > (...);
        // t.print("backproj", stream);  // stop + elapsed 一步完成
        // ============================================================
        struct CudaEventTimer {

            CudaEventTimer() {
                YK_CUDA_CHECK(cudaEventCreate(&start_));
                YK_CUDA_CHECK(cudaEventCreate(&stop_));
            }

            ~CudaEventTimer() noexcept {
                if (start_) {
                    cudaError_t e = cudaEventDestroy(start_);
                    if (e != cudaSuccess)
                        std::fprintf(stderr, "[CUDA] cudaEventDestroy(start) failed: %s\n",
                            cudaGetErrorString(e));
                }
                if (stop_) {
                    cudaError_t e = cudaEventDestroy(stop_);
                    if (e != cudaSuccess)
                        std::fprintf(stderr, "[CUDA] cudaEventDestroy(stop) failed: %s\n",
                            cudaGetErrorString(e));
                }
            }

            CudaEventTimer(const CudaEventTimer&) = delete;
            CudaEventTimer& operator=(const CudaEventTimer&) = delete;
            CudaEventTimer(CudaEventTimer&&) = delete;
            CudaEventTimer& operator=(CudaEventTimer&&) = delete;

            // 在 stream 中插入 start 事件
            void start(cudaStream_t stream = 0) {
                recorded_ = false;
                YK_CUDA_CHECK(cudaEventRecord(start_, stream));
            }

            // 在 stream 中插入 stop 事件
            void stop(cudaStream_t stream = 0) {
                YK_CUDA_CHECK(cudaEventRecord(stop_, stream));
                recorded_ = true;
            }

            // 等待 stop 事件完成，返回 GPU 侧耗时（毫秒）
            // 隐含对 stop_ 的同步，不影响整个 stream
            float elapsed_ms() {
                if (!recorded_)
                    throw std::logic_error("CudaEventTimer: stop() not called");
                YK_CUDA_CHECK(cudaEventSynchronize(stop_));
                float ms = 0.f;
                YK_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
                return ms;
            }

            // 打印到 stdout，方便调试
            void print(const char* tag, cudaStream_t stream = 0) {
                stop(stream);
                std::printf("[Timer] %-30s %.3f ms\n", tag, elapsed_ms());
            }

        private:
            cudaEvent_t start_ = nullptr;
            cudaEvent_t stop_ = nullptr;
            bool        recorded_ = false;
        };





        // ============================================================
        // CudaDeviceInfo  —  设备属性查询
        // usage:
        // CudaDeviceInfo info(0);
        //info.print();  // 启动时打印设备信息

        //// 分配大 buffer 前检查显存
        //size_t needed = size_t(Nx) * Ny * Nz * sizeof(float);
        //if (needed > info.free_memory())
        //    throw std::runtime_error("insufficient GPU memory");

        //// 动态调整 block size
        //int block = std::min(256, info.max_threads_per_block());
        // ============================================================
        struct CudaDeviceInfo {

            explicit CudaDeviceInfo(int deviceId = 0) : deviceId_(deviceId) {
                YK_CUDA_CHECK(cudaSetDevice(deviceId_));
                YK_CUDA_CHECK(cudaGetDeviceProperties(&prop_, deviceId_));
            }

            // SM 数量
            int sm_count() const noexcept { return prop_.multiProcessorCount; }

            // 每个 block 最大线程数
            int max_threads_per_block() const noexcept { return prop_.maxThreadsPerBlock; }

            // 每个 SM 最大线程数
            int max_threads_per_sm() const noexcept { return prop_.maxThreadsPerMultiProcessor; }

            // 理论最大并发线程数
            int max_concurrent_threads() const noexcept {
                return prop_.multiProcessorCount * prop_.maxThreadsPerMultiProcessor;
            }

            // Warp 大小（通常 32，别硬编码）
            int warp_size() const noexcept { return prop_.warpSize; }

            // 共享内存（per block，字节）
            size_t shared_mem_per_block() const noexcept { return prop_.sharedMemPerBlock; }

            // 共享内存（per SM，字节）
            size_t shared_mem_per_sm() const noexcept { return prop_.sharedMemPerMultiprocessor; }

            // L2 缓存大小（字节）
            int l2_cache_size() const noexcept { return prop_.l2CacheSize; }

            // 显存总量（字节）
            size_t total_memory() const noexcept { return prop_.totalGlobalMem; }

            // 当前可用显存（字节），每次调用都重新查询
            size_t free_memory() const {
                size_t free = 0, total = 0;
                YK_CUDA_CHECK(cudaSetDevice(deviceId_));
                YK_CUDA_CHECK(cudaMemGetInfo(&free, &total));
                return free;
            }

            // 已使用显存（字节）
            size_t used_memory() const {
                size_t free = 0, total = 0;
                YK_CUDA_CHECK(cudaSetDevice(deviceId_));
                YK_CUDA_CHECK(cudaMemGetInfo(&free, &total));
                return total - free;
            }

            // Compute capability
            int compute_major() const noexcept { return prop_.major; }
            int compute_minor() const noexcept { return prop_.minor; }
            int compute_capability() const noexcept { return prop_.major * 10 + prop_.minor; }

            // 设备名称
            const char* name() const noexcept { return prop_.name; }

            // 打印常用信息，调试用
            void print() const {
                std::printf("[CudaDeviceInfo] device %d: %s\n", deviceId_, prop_.name);
                std::printf("  compute capability : %d.%d\n", prop_.major, prop_.minor);
                std::printf("  SM count           : %d\n", prop_.multiProcessorCount);
                std::printf("  max threads/block  : %d\n", prop_.maxThreadsPerBlock);
                std::printf("  max threads/SM     : %d\n", prop_.maxThreadsPerMultiProcessor);
                std::printf("  warp size          : %d\n", prop_.warpSize);
                std::printf("  shared mem/block   : %zu KB\n", prop_.sharedMemPerBlock / 1024);
                std::printf("  L2 cache           : %d KB\n", prop_.l2CacheSize / 1024);
                std::printf("  total memory       : %zu MB\n", prop_.totalGlobalMem / 1024 / 1024);
                size_t free = 0, total = 0;
                cudaMemGetInfo(&free, &total);
                std::printf("  free  memory       : %zu MB\n", free / 1024 / 1024);
                std::printf("  used  memory       : %zu MB\n", (total - free) / 1024 / 1024);
            }

        private:
            int                  deviceId_ = 0;
            cudaDeviceProp       prop_{};
        };






    };
};

