// YkIterKernels.cuh
#include "YkIterLaunch.cuh"
#include <cuda_runtime.h>
#include "YkArith.cuh"
namespace YK
{
    namespace Iter
    {
        //namespace detail
        //{

        //    // r = meas - fwd，逐元素残差
        //    __global__ void residual_kernel(
        //        const float* d_meas,
        //        const float* d_fwd,
        //        float* d_res,
        //        size_t       n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d_res[i] = d_meas[i] - d_fwd[i];
        //    }

        //    // x += λ * bp / (w + eps)，逐元素更新
        //    __global__ void update_kernel(
        //        float* d_vol,
        //        const float* d_bp,
        //        const float* d_weight,
        //        float        lambda,
        //        float        eps,
        //        size_t       n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d_vol[i] += lambda * d_bp[i] / (d_weight[i] + eps);
        //    }


        //    __global__ void fill_ones_kernel(float* d, size_t n) {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i < n) d[i] = 1.f;
        //    }



        //    __global__ void clamp_kernel(
        //        float* d_vol,
        //        float  vmin,
        //        float  vmax,
        //        size_t n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d_vol[i] = fmaxf(vmin, fminf(vmax, d_vol[i]));
        //    }



        //    // invert: d[i] = 1 / (d[i] + eps)
        //    __global__ void invert_kernel(float* d, float eps, size_t n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d[i] = 1.f / (d[i] + eps);
        //    }

        //    // invert_scale: d[i] = scale / (d[i] + eps)
        //    __global__ void invert_scale_kernel(float* d, float scale, float eps, size_t n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d[i] = scale / (d[i] + eps);
        //    }

        //    // mul: d[i] *= w[i]
        //    __global__ void mul_kernel(float* d, const float* w, size_t n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d[i] *= w[i];
        //    }

        //    // addmul: d[i] += a[i] * w[i]
        //    __global__ void addmul_kernel(float* d, const float* a, const float* w, size_t n)
        //    {
        //        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        //        if (i >= n) return;
        //        d[i] += a[i] * w[i];
        //    }

        //};





        //void clamp_launch(
        //    float* d_vol,
        //    size_t       n,
        //    float        vmin,
        //    float        vmax,
        //    cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (int)((n + block - 1) / block);
        //   detail::clamp_kernel << <grid, block, 0, stream >> > (d_vol, vmin, vmax, n);
        //}


        //void fill_ones_launch(float* d, size_t n, cudaStream_t stream) {
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::fill_ones_kernel << <grid, block, 0, stream >> > (d, n);
        //}
        //// launch 函数
        //void residual_launch(const float* d_meas, const float* d_fwd,
        //    float* d_res, size_t n, cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::residual_kernel << <grid, block, 0, stream >> > (d_meas, d_fwd, d_res, n);
        //}

        //void update_launch(float* d_vol, const float* d_bp, const float* d_weight,
        //    float lambda, float eps, size_t n, cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::update_kernel << <grid, block, 0, stream >> > (d_vol, d_bp, d_weight,
        //        lambda, eps, n);
        //}


        //void invert_launch(float* d, float eps, size_t n, cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::invert_kernel << <grid, block, 0, stream >> > (d, eps, n);
        //}


        //void invert_scale_launch(float* d, float scale, float eps, size_t n, cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::invert_scale_kernel << <grid, block, 0, stream >> > (d, scale, eps, n);
        //}


        //void mul_launch(float* d, const float* w, size_t n, cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::mul_kernel << <grid, block, 0, stream >> > (d, w, n);
        //}

        //void addmul_launch(float* d, const float* a, const float* w, size_t n, cudaStream_t stream)
        //{
        //    const int block = 256;
        //    const int grid = (n + block - 1) / block;
        //    detail::addmul_kernel << <grid, block, 0, stream >> > (d, a, w, n);
        //}

        void invert_launch(float* d, float eps, size_t n, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [eps] __device__(float& x, size_t) {
                x = (x > eps) ? 1.f / x : 0.f;
            });
        }

        void invert_scale_launch(float* d, float scale, float eps,
            size_t n, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [scale, eps] __device__(float& x, size_t) {
                x = (x > eps) ? scale / x : 0.f;
            });
        }

        void fill_ones_launch(float* d, size_t n, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [] __device__(float& x, size_t) {
                x = 1.f;
            });
        }

        void fill_val_launch(float* d, float val, size_t n, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [val] __device__(float& x, size_t) {
                x = val;
            });
        }

        void mul_launch(float* out, const float* w, size_t n, cudaStream_t stream)
        {
            YK::elemwise(out, n, stream, [w] __device__(float& x, size_t i) {
                x *= w[i];
            });
        }

        void residual_launch(const float* meas, const float* fwd,
            float* out, size_t n, cudaStream_t stream)
        {
            YK::elemwise(out, n, stream, [meas, fwd] __device__(float& x, size_t i) {
                x = meas[i] - fwd[i];
            });
        }

        void addmul_launch(float* vol, const float* bp, const float* pw,
            size_t n, cudaStream_t stream)
        {
            YK::elemwise(vol, n, stream, [bp, pw] __device__(float& x, size_t i) {
                x += bp[i] * pw[i];
            });
        }

        void update_launch(float* vol, const float* bp, const float* w,
            float lambda, float eps, size_t n, cudaStream_t stream)
        {
            YK::elemwise(vol, n, stream, [bp, w, lambda, eps] __device__(float& x, size_t i) {
                x += lambda * bp[i] / (w[i] + eps);
            });
        }

        void clamp_launch(float* d, size_t n, float lo, float hi, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [lo, hi] __device__(float& x, size_t) {
                x = fminf(fmaxf(x, lo), hi);
            });
        }

        // YkIterKernels.cu 新增实现
        void clamp_min_launch(float* d, size_t n, float lo, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [lo] __device__(float& x, size_t) {
                if (x < lo) x = lo;
            });
        }

        void clamp_max_launch(float* d, size_t n, float hi, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [hi] __device__(float& x, size_t) {
                if (x > hi) x = hi;
            });
        }


        // r[i] /= (denom[i] + eps)  —— SART 的 R 行归一化用
        void divide_launch(float* out, const float* denom, float thresh,
            size_t n, cudaStream_t stream)
        {
            YK::elemwise(out, n, stream, [denom, thresh] __device__(float& x, size_t i) {
                x = (denom[i] > thresh) ? (x / denom[i]) : 0.f;
            });
        }

        // 1. threshold_invert：小于阈值置0，其余取倒数
        void threshold_invert_launch(float* x, float threshold, size_t n, cudaStream_t stream)
        {
            YK::elemwise(x, n, stream, [threshold] __device__(float& v, size_t i) {
                v = (v <= threshold) ? 0.f : 1.f / v;
            });
        }

        // 2. reduce_mean_z：[Nz,Ny,Nx] → [Ny,Nx] 沿Z均值
        // elemwise作用在输出vol2d上，大小Nx*Ny
        void reduce_mean_z_launch(
            const float* vol3d, float* vol2d,
            int Nx, int Ny, int Nz, cudaStream_t stream)
        {
            const size_t vol_xy = (size_t)Nx * Ny;
            YK::elemwise(vol2d, vol_xy, stream,
                [vol3d, Nx, Ny, Nz] __device__(float& v, size_t i) {
                int x = (int)(i % Nx);
                int y = (int)(i / Nx);
                float sum = 0.f;
                for (int z = 0; z < Nz; ++z)
                    sum += vol3d[(size_t)z * Ny * Nx + y * Nx + x];
                v = sum / (float)Nz;
            });
        }

        // 3. addmul_2d：x[z,y,x] += lambda * bp[z,y,x] * pw2d[y,x]
        //    pw2d 为 1/V，lambda 作为标量逐轮传入（支持 lambda_red）
        void addmul_2d_launch(
            float* vol, const float* bp, const float* pw2d, float lambda,
            int Nx, int Ny, int Nz, cudaStream_t stream)
        {
            const size_t vol_n = (size_t)Nx * Ny * Nz;
            YK::elemwise(vol, vol_n, stream,
                [bp, pw2d, lambda, Nx, Ny] __device__(float& x, size_t i) {
                int xy = (int)(i % ((size_t)Nx * Ny));
                x += lambda * bp[i] * pw2d[xy];
            });
        }

    };

};



