// YkIterKernels.cuh
#include "YkIterLaunch.cuh"
#include "common/YkDeviceWorkspace.hpp"
#include <cuda_runtime.h>
#include "YkArith.cuh"
#include <global/YkMacro.hpp>
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


        void threshold_inf_launch(float* d, size_t n, float thresh, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [thresh] __device__(float& x, size_t) {
                if (x <= thresh) x = 1e30f;
            });
        }


        void multiply_launch(float* a, const float* b, size_t n, cudaStream_t stream)
        {
            YK::elemwise(a, n, stream, [b] __device__(float& x, size_t i) {
                x *= b[i];
            });
        }

        void rcp_launch(float* d, size_t n, cudaStream_t stream)
        {
            YK::elemwise(d, n, stream, [] __device__(float& x, size_t) {
                x = __frcp_rn(x);
            });
        }


        // ── axpy：x += alpha * y ──────────────────────────────────────────
        void axpy_launch(float* x, const float* y, float alpha,
            size_t n, cudaStream_t stream)
        {
            YK::elemwise(x, n, stream, [y, alpha] __device__(float& xi, size_t i) {
                xi += alpha * y[i];
            });
        }

        // ── scale：x *= alpha ─────────────────────────────────────────────
        void scale_launch(float* x, float alpha, size_t n, cudaStream_t stream)
        {
            YK::elemwise(x, n, stream, [alpha] __device__(float& xi, size_t) {
                xi *= alpha;
            });
        }

        void subtract_launch(float* out, const float* a, const float* b,
            size_t n, cudaStream_t stream)
        {
            YK::elemwise(out, n, stream, [a, b] __device__(float& value, size_t i) {
                value = a[i] - b[i];
            });
        }

        void linear_combination_launch(float* out, const float* a,
            const float* b, float scale_b, size_t n, cudaStream_t stream)
        {
            YK::elemwise(out, n, stream,
                [a, b, scale_b] __device__(float& value, size_t i) {
                    value = a[i] + scale_b * b[i];
                });
        }



        __device__ __forceinline__ float warp_sum(float value)
        {
            const unsigned mask = __activemask();
            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                value += __shfl_down_sync(mask, value, offset);
            return value;
        }

        // dot_reduce_kernel 放在文件作用域
        __global__ void dot_reduce_kernel(const float* a, const float* b,
            float* block_results, size_t n)
        {
            extern __shared__ float warp_results[];
            const int tid = static_cast<int>(threadIdx.x);
            const int lane = tid & 31;
            const int warp_id = tid >> 5;
            const int warps_per_block = (static_cast<int>(blockDim.x) + 31) >> 5;
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

            float val = 0.f;
            while (i < n) {
                val += a[i] * b[i];
                i += gridDim.x * blockDim.x;
            }

            val = warp_sum(val);
            if (lane == 0)
                warp_results[warp_id] = val;
            __syncthreads();

            if (warp_id == 0) {
                val = (lane < warps_per_block) ? warp_results[lane] : 0.f;
                val = warp_sum(val);
            }

            if (tid == 0)
                block_results[blockIdx.x] = val;
        }

        // ── dot product：result = sum(a * b) ─────────────────────────────
        // 用两阶段 reduce，无需 cuBLAS
        void dot_launch(const float* a, const float* b,
            size_t n, float* h_result, cudaStream_t stream)
        {
            if (n == 0) {
                *h_result = 0.f;
                return;
            }
            const int block = 256;
            const int max_blocks = 1024;
            const int grid = (int)std::min(
                (size_t)max_blocks, (n + block - 1) / block);

            // reduction 的中间结果也走统一工作区，避免异常路径泄漏。
            DeviceWorkspaceF32 d_block;
            int device_id = 0;
            YK_CUDA_CHECK(cudaGetDevice(&device_id));
            d_block.allocate(grid, device_id);

            // block 内 reduce kernel，使用文件作用域的专用 kernel。
            const int warps_per_block = (block + 31) / 32;
            dot_reduce_kernel << <grid, block, warps_per_block * sizeof(float), stream >> > (
                a, b, d_block.data(), n);
            YK_CUDA_KERNEL_CHECK();

            std::vector<float> h_block(grid);
            YK_CUDA_CHECK(cudaMemcpyAsync(h_block.data(), d_block.data(),
                grid * sizeof(float), cudaMemcpyDeviceToHost, stream));
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            float sum = 0.f;
            for (int i = 0; i < grid; ++i)
                sum += h_block[i];
            *h_result = sum;
        }




        __global__ void mean_z_to_2d_kernel(
            const float* __restrict__ vol3d, float* __restrict__ out2d,
            int nx, int ny, int nz)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= nx || y >= ny) return;
            const size_t stride = (size_t)nx * ny;
            const size_t xy = (size_t)y * nx + x;
            float sum = 0.f;
            for (int z = 0; z < nz; ++z) sum += vol3d[(size_t)z * stride + xy];
            out2d[xy] = sum / (float)nz;
        }

        __global__ void update_v2d_kernel(
            float* __restrict__ vol, const float* __restrict__ bp,
            const float* __restrict__ v2d,
            float lambda, float eps, int nx, int ny, int nz)
        {
            size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            const size_t n = (size_t)nx * ny * nz;
            if (i >= n) return;
            const size_t xy = i % ((size_t)nx * ny);
            vol[i] += lambda * bp[i] / (v2d[xy] + eps);
        }

        void mean_z_to_2d_launch(const float* d_vol3d, float* d_out2d,   // ← const float*
            int nx, int ny, int nz, cudaStream_t stream)
        {
            dim3 blk(16, 16);
            dim3 grd((nx + 15) / 16, (ny + 15) / 16);
            mean_z_to_2d_kernel << <grd, blk, 0, stream >> > (d_vol3d, d_out2d, nx, ny, nz);
        }

        void update_v2d_launch(float* d_vol, const float* d_bp,
            const float* d_v2d, float lambda, float eps,                 // ← 带 eps
            int nx, int ny, int nz, cudaStream_t stream)
        {
            const size_t n = (size_t)nx * ny * nz;
            const int blk = 256;
            const size_t grd = (n + blk - 1) / blk;
            update_v2d_kernel << <(unsigned)grd, blk, 0, stream >> > (
                d_vol, d_bp, d_v2d, lambda, eps, nx, ny, nz);
        }
    };

};



