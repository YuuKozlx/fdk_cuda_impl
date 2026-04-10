
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "YkCreateFilterKernelHelpers.cuh"
#include "YkCreateFilterKernelLaunch.cuh"

namespace YK {
    namespace Filter {
        namespace detail {

            // (0) Identity weights: w[k] = gain * (bake_invN ? 1/N : 1)
            static __global__ void kernel_fill_identity_weights(
                float* __restrict__ w,
                int n_complex,
                int N,
                float gain,
                bool bake_invN)
            {
                int k = blockIdx.x * blockDim.x + threadIdx.x;
                if (k >= n_complex) return;
                float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
                w[k] = gain * invN;
            }

            // (1) Analytic frequency-domain build (direct)
            // w[k] = gain * ramp(f) * window(x) * (optional 1/N), cutoff applied
            static __global__ void kernel_build_weights_analytic_freq(
                float* __restrict__ w,
                int n_complex,
                int N,
                SFilterKernelDesc desc,
                bool bake_invN)
            {
                int k = blockIdx.x * blockDim.x + threadIdx.x;
                if (k >= n_complex) return;

                if (desc.kind == EFilterKernel::None) {
                    float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
                    w[k] = desc.gain * invN;
                    return;
                }

                float f = (N > 0) ? ((float)k / (float)N) : 0.0f; // [0, 0.5]

                if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; return; }

                float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
                if (f > cc) { w[k] = 0.0f; return; }

                float ramp = f;
                float x = (cc > 0.0f) ? (f / cc) : 0.0f;
                float shape = window_shape(x, desc.kind);

                float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
                w[k] = desc.gain * ramp * shape * invN;
            }

            // (2) Spatial discrete Ram-Lak kernel (DU=1 convention)
            static __global__ void kernel_gen_spatial_rl_kernel_du1(
                float* __restrict__ h,
                int N,
                bool bake_invN)
            {
                int u = blockIdx.x * blockDim.x + threadIdx.x;
                if (u >= N) return;

                int n = (u <= N / 2) ? u : (u - N);
                int an = (n < 0) ? -n : n;

                float val = 0.0f;
                if (n == 0) {
                    val = 1.0f / 4.0f;
                }
                else if (an & 1) {
                    const float pi = 3.14159265358979323846f;
                    float fn = (float)n;
                    val = -1.0f / (pi * pi * fn * fn);
                }

                if (bake_invN && N > 0) val *= (1.0f / (float)N);
                h[u] = val;
            }

            // (3) Extract ramp weights from FFT(RL)
            static __global__ void kernel_extract_weights_from_fft(
                const cufftComplex* __restrict__ src,
                float* __restrict__ dst,
                int n_complex,
                int mode,           // 0 = RealPart, 1 = Magnitude
                bool force_dc_zero)
            {
                int k = blockIdx.x * blockDim.x + threadIdx.x;
                if (k >= n_complex) return;

                float re = src[k].x;
                float im = src[k].y;
                float v = (mode == 1) ? sqrtf(re * re + im * im) : re;

                if (force_dc_zero && k == 0) v = 0.0f;
                dst[k] = v;
            }

            // (4) Apply window / cutoff / gain / DC on ramp weights (in-place)
            static __global__ void kernel_apply_window_to_weights_inplace(
                float* __restrict__ w,
                int n_complex,
                int N,
                SFilterKernelDesc desc)
            {
                int k = blockIdx.x * blockDim.x + threadIdx.x;
                if (k >= n_complex) return;

                if (desc.kind == EFilterKernel::None) {
                    w[k] = w[k] * desc.gain;
                    return;
                }

                float f = (N > 0) ? ((float)k / (float)N) : 0.0f;

                if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; return; }

                float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
                if (f > cc) { w[k] = 0.0f; return; }

                float x = (cc > 0.0f) ? (f / cc) : 0.0f;
                float shape = window_shape(x, desc.kind);

                w[k] = w[k] * (desc.gain * shape);
            }

        }; // namespace detail

        bool flt_launch_kernel_fill_identity_weights(
            float* d_w,
            int n_complex,
            int N,
            float gain,
            bool bake_invN,
            cudaStream_t stream)
        {
            dim3 block(256, 1);
            dim3 grid((n_complex + block.x - 1) / block.x, 1);
            detail::kernel_fill_identity_weights << <grid, block, 0, stream >> > (
                d_w, n_complex, N, gain, bake_invN);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        bool flt_launch_kernel_build_weights_analytic_freq(
            float* d_w,
            int n_complex,
            int N,
            SFilterKernelDesc desc,
            bool bake_invN,
            cudaStream_t stream)
        {
            dim3 block(256, 1);
            dim3 grid((n_complex + block.x - 1) / block.x, 1);
            detail::kernel_build_weights_analytic_freq << <grid, block, 0, stream >> > (
                d_w, n_complex, N, desc, bake_invN);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        bool flt_launch_kernel_gen_spatial_rl_kernel_du1(
            float* d_h,
            int N,
            bool bake_invN,
            cudaStream_t stream)
        {
            dim3 block(256, 1);
            dim3 grid((N + block.x - 1) / block.x, 1);
            detail::kernel_gen_spatial_rl_kernel_du1 << <grid, block, 0, stream >> > (
                d_h, N, bake_invN);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        bool flt_launch_kernel_extract_weights_from_fft(
            const cufftComplex* d_src,
            float* d_dst,
            int n_complex,
            ERampExtractMode mode,
            bool force_dc_zero,
            cudaStream_t stream)
        {
            dim3 block(256, 1);
            dim3 grid((n_complex + block.x - 1) / block.x, 1);
            detail::kernel_extract_weights_from_fft << <grid, block, 0, stream >> > (
                d_src, d_dst, n_complex, (int)mode, force_dc_zero);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }


        bool flt_launch_kernel_apply_window_to_weights_inplace(
            float* d_w,
            int n_complex,
            int N,
            SFilterKernelDesc desc,
            cudaStream_t stream)
        {
            dim3 block(256, 1);
            dim3 grid((n_complex + block.x - 1) / block.x, 1);
            detail::kernel_apply_window_to_weights_inplace << <grid, block, 0, stream >> > (
                d_w, n_complex, N, desc);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

    } // namespace Filter
} // namespace YK
