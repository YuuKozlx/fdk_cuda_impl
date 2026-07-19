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

        void threshold_inf_launch(float* d, size_t n, float thresh, cudaStream_t stream);
        void multiply_launch(float* a, const float* b, size_t n, cudaStream_t stream);
        void rcp_launch(float* d, size_t n, cudaStream_t stream);


        // ── axpy：x += alpha * y ──────────────────────────────────────────
        void axpy_launch(float* x, const float* y, float alpha,
            size_t n, cudaStream_t stream);


        // ── scale：x *= alpha ─────────────────────────────────────────────
        void scale_launch(float* x, float alpha, size_t n, cudaStream_t stream);

        void dot_launch(const float* a, const float* b,
            size_t n, float* h_result, cudaStream_t stream);

        // ============================================================================
        // 需并入 Iter kernels 的两个新 kernel(声明加到 Iter 头文件):
        //
        void mean_z_to_2d_launch(const float* d_vol3d, float* d_out2d,
            int nx, int ny, int nz, cudaStream_t stream);
        void update_v2d_launch(float* d_vol, const float* d_bp,
            const float* d_v2d, float lambda, float eps,
            int nx, int ny, int nz, cudaStream_t stream);
    } // namespace Iter
} // namespace YK