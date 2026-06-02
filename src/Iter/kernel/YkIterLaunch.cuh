#pragma once
#include <cuda_runtime.h>


namespace YK {
    namespace Iter {
        void invert_launch(float* d, float eps, size_t n, cudaStream_t stream);
        void invert_scale_launch(float* d, float scale, float eps, size_t n, cudaStream_t stream);
        void fill_ones_launch(float* d, size_t n, cudaStream_t stream);
        void fill_val_launch(float* d, float val, size_t n, cudaStream_t stream);
        void mul_launch(float* out, const float* w, size_t n, cudaStream_t stream);
        void residual_launch(const float* meas, const float* fwd,
            float* out, size_t n, cudaStream_t stream);
        void addmul_launch(float* vol, const float* bp, const float* pw,
            size_t n, cudaStream_t stream);
        void update_launch(float* vol, const float* bp, const float* w,
            float lambda, float eps, size_t n, cudaStream_t stream);
        void clamp_launch(float* d, size_t n, float lo, float hi, cudaStream_t stream);
        // YkIterLaunch.cuh 新增声明
        void clamp_min_launch(float* d, size_t n, float lo, cudaStream_t stream);
        void clamp_max_launch(float* d, size_t n, float hi, cudaStream_t stream);
        void divide_launch(float* out, const float* denom, float eps,
            size_t n, cudaStream_t stream);
        void threshold_invert_launch(
            float* x, float threshold, size_t n, cudaStream_t stream);
        void reduce_mean_z_launch(
            const float* vol3d, float* vol2d,
            int Nx, int Ny, int Nz, cudaStream_t stream);
        void addmul_2d_launch(
            float* vol, const float* bp, const float* pw2d, float lambda,
            int Nx, int Ny, int Nz, cudaStream_t stream);

    } // namespace Iter
} // namespace YK