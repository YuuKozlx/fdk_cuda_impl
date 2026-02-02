#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "YkFFT.hpp"

namespace YK {

        __global__ void _kernel_pointwise_mul(cufftComplex* data, const float* weights, int n_complex, int batch) {
            int u = blockIdx.x * blockDim.x + threadIdx.x;
            int b = blockIdx.y;
            if (u < n_complex && b < batch) {
                int idx = b * n_complex + u;
                data[idx].x *= weights[u];
                data[idx].y *= weights[u];
            }
        }

        void convolveFrequency(float* d_padded_data, const float* d_filter_fft, int paddedN, int batch) {

            int n_complex = paddedN / 2 + 1;
            cufftComplex* d_complex_buf;
            cudaMalloc(&d_complex_buf, batch * n_complex * sizeof(cufftComplex));

            CudaFFT::fft(d_padded_data, d_complex_buf, paddedN, batch);

            dim3 block(256, 1);
            dim3 grid((n_complex + 255) / 256, batch);
            _kernel_pointwise_mul << <grid, block >> > (d_complex_buf, d_filter_fft, n_complex, batch);

            CudaFFT::ifft(d_complex_buf, d_padded_data, paddedN, batch);

            cudaFree(d_complex_buf);
        }


};