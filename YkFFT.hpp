#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

#include "YkGlobals.h"

namespace YK {

    class CudaFFT {
    public:
        enum class EPlanMode : uint8_t {
            Both,       // R2C + C2R（默认，兼容旧代码）
            R2COnly,    // 只建 FFT
            C2ROnly     // 只建 IFFT
        };

        CudaFFT() = default;
        ~CudaFFT() { release(); }

        CudaFFT(const CudaFFT&) = delete;
        CudaFFT& operator=(const CudaFFT&) = delete;

        CudaFFT(CudaFFT&& o) noexcept { move_from(o); }
        CudaFFT& operator=(CudaFFT&& o) noexcept {
            if (this != &o) { release(); move_from(o); }
            return *this;
        }

        // 兼容旧接口：默认 Both
        bool init(int N, int batch, cudaStream_t stream = 0) {
            return init(N, batch, EPlanMode::Both, stream);
        }

        // 新增：模式 init（不改类名，不影响旧代码）
        bool init(int N, int batch, EPlanMode mode, cudaStream_t stream = 0) {
            release();
            if (N <= 0 || batch <= 0) return false;

            N_ = N;
            batch_ = batch;
            stream_ = stream;
            mode_ = mode;
            n_complex_ = N_ / 2 + 1;

            const int rank = 1;
            int n[1] = { N_ };

            size_t work_r2c = 0, work_c2r = 0;

            // R2C plan
            if (mode_ == EPlanMode::Both || mode_ == EPlanMode::R2COnly) {
                YK_CUFFT_CHECK(cufftCreate(&plan_r2c_));
                YK_CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));

                int inembed[1] = { N_ };
                int onembed[1] = { n_complex_ };
                int istride = 1, ostride = 1;
                int idist = N_, odist = n_complex_;

                YK_CUFFT_CHECK(cufftMakePlanMany(
                    plan_r2c_, rank, n,
                    inembed, istride, idist,
                    onembed, ostride, odist,
                    CUFFT_R2C, batch_, &work_r2c
                ));
            }

            // C2R plan
            if (mode_ == EPlanMode::Both || mode_ == EPlanMode::C2ROnly) {
                YK_CUFFT_CHECK(cufftCreate(&plan_c2r_));
                YK_CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));

                int inembed[1] = { n_complex_ };
                int onembed[1] = { N_ };
                int istride = 1, ostride = 1;
                int idist = n_complex_, odist = N_;

                YK_CUFFT_CHECK(cufftMakePlanMany(
                    plan_c2r_, rank, n,
                    inembed, istride, idist,
                    onembed, ostride, odist,
                    CUFFT_C2R, batch_, &work_c2r
                ));
            }

            // workspace（建议开，性能稳定）
            work_bytes_ = (work_r2c > work_c2r) ? work_r2c : work_c2r;
            if (work_bytes_ > 0) {
                YK_CUDA_CHECK(cudaMalloc(&d_work_, work_bytes_));
                if (plan_r2c_) YK_CUFFT_CHECK(cufftSetWorkArea(plan_r2c_, d_work_));
                if (plan_c2r_) YK_CUFFT_CHECK(cufftSetWorkArea(plan_c2r_, d_work_));
            }

            return true;
        }

        void setStream(cudaStream_t s) {
            stream_ = s;
            if (plan_r2c_) YK_CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
            if (plan_c2r_) YK_CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
        }

        void fft(float* d_real, cufftComplex* d_complex) const {
            YK_ASSERT(plan_r2c_ && "CudaFFT::fft called but R2C plan not created");
            YK_CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)d_real, d_complex));
        }

        void ifft(cufftComplex* d_complex, float* d_real) const {
            YK_ASSERT(plan_c2r_ && "CudaFFT::ifft called but C2R plan not created");
            YK_CUFFT_CHECK(cufftExecC2R(plan_c2r_, d_complex, (cufftReal*)d_real));
        }

        void release() {
            if (plan_r2c_) { YK_CUFFT_CHECK(cufftDestroy(plan_r2c_)); plan_r2c_ = 0; }
            if (plan_c2r_) { YK_CUFFT_CHECK(cufftDestroy(plan_c2r_)); plan_c2r_ = 0; }
            if (d_work_) { YK_CUDA_CHECK(cudaFree(d_work_)); d_work_ = nullptr; }

            N_ = 0;
            batch_ = 0;
            n_complex_ = 0;
            stream_ = 0;
            mode_ = EPlanMode::Both;
            work_bytes_ = 0;
        }

        int N() const { return N_; }
        int batch() const { return batch_; }
        int n_complex() const { return n_complex_; }
        cudaStream_t stream() const { return stream_; }
        EPlanMode mode() const { return mode_; }

    private:
        void move_from(CudaFFT& o) noexcept {
            plan_r2c_ = o.plan_r2c_; o.plan_r2c_ = 0;
            plan_c2r_ = o.plan_c2r_; o.plan_c2r_ = 0;
            d_work_ = o.d_work_; o.d_work_ = nullptr;

            work_bytes_ = o.work_bytes_; o.work_bytes_ = 0;

            N_ = o.N_; o.N_ = 0;
            batch_ = o.batch_; o.batch_ = 0;
            n_complex_ = o.n_complex_; o.n_complex_ = 0;
            stream_ = o.stream_; o.stream_ = 0;
            mode_ = o.mode_; o.mode_ = EPlanMode::Both;
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
    };

} // namespace YK
