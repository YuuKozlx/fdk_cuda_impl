#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include <driver_types.h>
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkFilterTypes.hpp"
#include "YkCreateFilterKernelHelpers.cuh"

namespace YK {
    namespace Filter {

        bool flt_launch_kernel_fill_identity_weights(
            float* d_w,
            int n_complex,
            int N,
            float gain,
            bool bake_invN,
            cudaStream_t stream);


        bool flt_launch_kernel_build_weights_analytic_freq(
            float* d_w,
            int n_complex,
            int N,
            SFilterKernelDesc desc,
            bool bake_invN,
            cudaStream_t stream);


        bool flt_launch_kernel_gen_spatial_rl_kernel_du1(
            float* d_h,
            int N,
            bool bake_invN,
            cudaStream_t stream);

        bool flt_launch_kernel_build_spatial_ramp(
            float* d_spatial,
            int N,
            const float* d_ramp,
            int ramp_size,
            bool bake_invN,
            cudaStream_t stream);


        bool flt_launch_kernel_extract_weights_from_fft(
            const cufftComplex* d_src,
            float* d_dst,
            int n_complex,
            ERampExtractMode mode,
            cudaStream_t stream);


        bool flt_launch_kernel_apply_window_to_weights_inplace(
            float* d_w,
            int n_complex,
            int N,
            SFilterKernelDesc desc,
            cudaStream_t stream);


        void flt_launch_kernel_scale_inplace(
            float* data, int n, float scale,
            cudaStream_t stream);


    } // namespace Filter
} // namespace YK
