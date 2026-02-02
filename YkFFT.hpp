#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace YK {

    // --- 第一步：通用 FFT 包装 (基础工具) ---
    class CudaFFT {
    public:
        static void fft(float* d_real, cufftComplex* d_complex, int N, int batch) {
            cufftHandle plan;
            cufftPlanMany(&plan, 1, &N, NULL, 1, N, NULL, 1, N / 2 + 1, CUFFT_R2C, batch);
            cufftExecR2C(plan, (cufftReal*)d_real, d_complex);
            cufftDestroy(plan);
        }

        static void ifft(cufftComplex* d_complex, float* d_real, int N, int batch) {
            cufftHandle plan;
            cufftPlanMany(&plan, 1, &N, NULL, 1, N / 2 + 1, NULL, 1, N, CUFFT_C2R, batch);
            cufftExecC2R(plan, d_complex, (cufftReal*)d_real);
            cufftDestroy(plan);
        }
    };
}


