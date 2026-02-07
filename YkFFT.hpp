#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>
#include <driver_types.h>
#include "YkGlobals.h"

namespace YK {


    class CudaFFT {
    public:
        bool init(int N, int batch, cudaStream_t stream = 0) {
            release();
            N_ = N; batch_ = batch; stream_ = stream;
            int n_complex = N / 2 + 1;

            YK_CUFFT_CHECK(cufftPlanMany(&plan_r2c_, 1, &N_,
                nullptr, 1, N_,
                nullptr, 1, n_complex,
                CUFFT_R2C, batch_));
            YK_CUFFT_CHECK(cufftPlanMany(&plan_c2r_, 1, &N_,
                nullptr, 1, n_complex,
                nullptr, 1, N_,
                CUFFT_C2R, batch_));

            YK_CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
            YK_CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
            return true;
        }

        void setStream(cudaStream_t s) {
            stream_ = s;
            if (plan_r2c_) YK_CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
            if (plan_c2r_) YK_CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
        }

        void fft(float* d_real, cufftComplex* d_complex) {
            YK_CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)d_real, d_complex));
        }

        void ifft(cufftComplex* d_complex, float* d_real) {
            YK_CUFFT_CHECK(cufftExecC2R(plan_c2r_, d_complex, (cufftReal*)d_real));
        }

        void release() {
            if (plan_r2c_) { cufftDestroy(plan_r2c_); plan_r2c_ = 0; }
            if (plan_c2r_) { cufftDestroy(plan_c2r_); plan_c2r_ = 0; }
            N_ = 0; batch_ = 0; stream_ = 0;
        }

        ~CudaFFT() { release(); }

        int N() const { return N_; }
        int batch() const { return batch_; }

    private:
        cufftHandle plan_r2c_ = 0;
        cufftHandle plan_c2r_ = 0;
        int N_ = 0;
        int batch_ = 0;
        cudaStream_t stream_ = 0;
    };

} // namespace YK
