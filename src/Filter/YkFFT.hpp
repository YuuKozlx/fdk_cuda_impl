#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>
#include <cstdio>

#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"

namespace YK {

    class CudaFFT {
    public:
        enum class EPlanMode : uint8_t {
            Both,       // R2C + C2R（默认）
            R2COnly,    // 只建 FFT
            C2ROnly     // 只建 IFFT
        };

        CudaFFT() = default;
        ~CudaFFT() { release(false); }

        CudaFFT(const CudaFFT&) = delete;
        CudaFFT& operator=(const CudaFFT&) = delete;

        CudaFFT(CudaFFT&& o) noexcept { move_from(o); }
        CudaFFT& operator=(CudaFFT&& o) noexcept {
            if (this != &o) { release(true); move_from(o); }
            return *this;
        }

        bool init(int N, int batch, cudaStream_t stream = 0) {
            return init(N, batch, EPlanMode::Both, stream);
        }

        bool init(int N, int batch, EPlanMode mode, cudaStream_t stream = 0) {
            if (N <= 0 || batch <= 0) return false;

            // 固定/记录 device（避免外部切 device 导致 plan/workspace 跨设备）
            int curDev = -1;
            YK_CUDA_CHECK(cudaGetDevice(&curDev));

            // 复用条件：参数相同 + 当前 device 相同 + mode 对应的 plan 都存在
            if (N == N_ && batch == batch_ && mode == mode_ && curDev == device_) {
                const bool need_r2c = (mode_ == EPlanMode::Both || mode_ == EPlanMode::R2COnly);
                const bool need_c2r = (mode_ == EPlanMode::Both || mode_ == EPlanMode::C2ROnly);
                if ((!need_r2c || plan_r2c_) && (!need_c2r || plan_c2r_)) {
                    setStream(stream);
                    return true;
                }
            }

            release(true);

            device_ = curDev;
            N_ = N;
            batch_ = batch;
            mode_ = mode;
            stream_ = stream;
            n_complex_ = N_ / 2 + 1;

            const int rank = 1;
            int n[1] = { N_ };

            size_t work_r2c = 0, work_c2r = 0;

            // ---------------- R2C plan ----------------
            if (mode_ == EPlanMode::Both || mode_ == EPlanMode::R2COnly) {
                YK_CUFFT_CHECK(cufftCreate(&plan_r2c_));
                YK_CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));

                // 关键：关闭 auto-allocation，确保我们自己的 work area 生效且不会额外分配
                YK_CUFFT_CHECK(cufftSetAutoAllocation(plan_r2c_, 0));

                int inembed[1] = { N_ };
                int onembed[1] = { n_complex_ };
                int istride = 1, ostride = 1;
                int idist = N_;
                int odist = n_complex_;

                YK_CUFFT_CHECK(cufftMakePlanMany(
                    plan_r2c_, rank, n,
                    inembed, istride, idist,
                    onembed, ostride, odist,
                    CUFFT_R2C, batch_, &work_r2c
                ));
            }

            // ---------------- C2R plan ----------------
            if (mode_ == EPlanMode::Both || mode_ == EPlanMode::C2ROnly) {
                YK_CUFFT_CHECK(cufftCreate(&plan_c2r_));
                YK_CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));

                // 关键：关闭 auto-allocation
                YK_CUFFT_CHECK(cufftSetAutoAllocation(plan_c2r_, 0));

                // CUFFT_C2R 的 n 仍然是实域长度 N_
                int inembed[1] = { n_complex_ };
                int onembed[1] = { N_ };
                int istride = 1, ostride = 1;
                int idist = n_complex_;
                int odist = N_;

                YK_CUFFT_CHECK(cufftMakePlanMany(
                    plan_c2r_, rank, n,
                    inembed, istride, idist,
                    onembed, ostride, odist,
                    CUFFT_C2R, batch_, &work_c2r
                ));
            }

            // ---------------- workspace ----------------
            work_bytes_ = (work_r2c > work_c2r) ? work_r2c : work_c2r;

            // 可选：0 workspace 也允许（某些尺寸/版本可能返回 0）
            if (work_bytes_ > 0) {
                // 确保仍在同 device 上分配
                int dev2 = -1;
                YK_CUDA_CHECK(cudaGetDevice(&dev2));
                YK_ASSERT(dev2 == device_ && "CudaFFT::init: current device changed unexpectedly");

                YK_CUDA_CHECK(cudaMalloc(&d_work_, work_bytes_));

                if (plan_r2c_) YK_CUFFT_CHECK(cufftSetWorkArea(plan_r2c_, d_work_));
                if (plan_c2r_) YK_CUFFT_CHECK(cufftSetWorkArea(plan_c2r_, d_work_));
            }
            else {
                // 明确置空，避免误用
                d_work_ = nullptr;
            }

            return true;
        }

        void setStream(cudaStream_t s) {
            stream_ = s;
            if (plan_r2c_) YK_CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
            if (plan_c2r_) YK_CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
        }

        // 同一 handle 不要并发执行（同/不同 stream 都不建议）
        void fft(float* d_real, cufftComplex* d_complex) const {
            YK_ASSERT(device_ >= 0 && "CudaFFT not initialized");
            int curDev = -1;
            YK_CUDA_CHECK(cudaGetDevice(&curDev));
            YK_ASSERT(curDev == device_ && "CudaFFT::fft called on different CUDA device");

            YK_ASSERT(plan_r2c_ && "CudaFFT::fft called but R2C plan not created");
            YK_CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)d_real, d_complex));
        }

        void ifft(cufftComplex* d_complex, float* d_real) const {
            YK_ASSERT(device_ >= 0 && "CudaFFT not initialized");
            int curDev = -1;
            YK_CUDA_CHECK(cudaGetDevice(&curDev));
            YK_ASSERT(curDev == device_ && "CudaFFT::ifft called on different CUDA device");

            YK_ASSERT(plan_c2r_ && "CudaFFT::ifft called but C2R plan not created");
            YK_CUFFT_CHECK(cufftExecC2R(plan_c2r_, d_complex, (cufftReal*)d_real));
        }

        float invN() const { return (N_ > 0) ? (1.0f / (float)N_) : 1.0f; }

        void release(bool strict = true) {
            auto destroy_plan = [&](cufftHandle& h, const char* name) {
                if (!h) return;
                cufftResult r = cufftDestroy(h);
                if (strict) {
                    YK_CUFFT_CHECK(r);
                }
                else {
                    if (r != CUFFT_SUCCESS) {
                        std::fprintf(stderr,
                            "[CudaFFT] warning: cufftDestroy(%s) failed, code=%d\n",
                            name, (int)r);
                    }
                }
                h = 0;
                };

            destroy_plan(plan_r2c_, "R2C");
            destroy_plan(plan_c2r_, "C2R");

            if (d_work_) {
                cudaError_t e = cudaFree(d_work_);
                if (strict) {
                    YK_CUDA_CHECK(e);
                }
                else {
                    if (e != cudaSuccess) {
                        std::fprintf(stderr,
                            "[CudaFFT] warning: cudaFree(work) failed: %s\n",
                            cudaGetErrorString(e));
                    }
                }
                d_work_ = nullptr;
            }

            N_ = 0;
            batch_ = 0;
            n_complex_ = 0;
            stream_ = 0;
            mode_ = EPlanMode::Both;
            work_bytes_ = 0;
            device_ = -1;
        }

        int N() const { return N_; }
        int batch() const { return batch_; }
        int n_complex() const { return n_complex_; }
        cudaStream_t stream() const { return stream_; }
        EPlanMode mode() const { return mode_; }
        size_t work_bytes() const { return work_bytes_; }
        int device() const { return device_; }

    private:
        void move_from(CudaFFT& o) noexcept {
            plan_r2c_ = o.plan_r2c_; o.plan_r2c_ = 0;
            plan_c2r_ = o.plan_c2r_; o.plan_c2r_ = 0;
            d_work_ = o.d_work_;   o.d_work_ = nullptr;

            work_bytes_ = o.work_bytes_; o.work_bytes_ = 0;

            N_ = o.N_; o.N_ = 0;
            batch_ = o.batch_; o.batch_ = 0;
            n_complex_ = o.n_complex_; o.n_complex_ = 0;
            stream_ = o.stream_; o.stream_ = 0;
            mode_ = o.mode_; o.mode_ = EPlanMode::Both;

            device_ = o.device_; o.device_ = -1;
        }

    private:
        cufftHandle plan_r2c_ = 0;
        cufftHandle plan_c2r_ = 0;

        void* d_work_ = nullptr;
        size_t work_bytes_ = 0;

        int N_ = 0;
        int batch_ = 0;
        int n_complex_ = 0;
        cudaStream_t stream_ = 0;
        EPlanMode mode_ = EPlanMode::Both;

        int device_ = -1; // 新增：记录创建/分配所在 device
    };

} // namespace YK
