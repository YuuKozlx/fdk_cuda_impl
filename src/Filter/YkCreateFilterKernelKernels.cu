
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
            // 公共：从 blockDim/gridDim 推导 warp 身份
            // 调用处直接 inline，无函数调用开销
            // warp 索引 计算范式
            //    ┌─────────────────────────────────────────┐
            //    │  第一层：身份确定（WARP_STRIDE_INIT）   │
            //    │  确定当前线程在全局中的 warp 编号       │
            //    │  以及 warp 内的 lane 位置               │
            //    └─────────────────────────────────────────┘
            //    ↓
            //    ┌─────────────────────────────────────────┐
            //    │  第二层：stride loop                    │
            //    │  base = warp_global * 32                │
            //    │  每轮步进 n_warps * 32                  │
            //    │  保证全局数据无遗漏、无重复             │
            //    └─────────────────────────────────────────┘
            //    ↓
            //    ┌─────────────────────────────────────────┐
            //    │  第三层：lane 内计算                    │
            //    │  k = base + lane                        │
            //    │  warp 内 32 个 lane 访问连续地址        │
            //    │  保证 coalesced access                  │
            //    └─────────────────────────────────────────┘
// =============================================================================
// WARP STRIDE LOOP 范式说明
// =============================================================================
//
// 一、基本概念
//
//   GPU 线程的最小执行单位是 warp，每个 warp 固定包含 32 个线程。
//   warp 内的线程编号称为 lane，范围 [0, 31]。
//
//   threadIdx.x:  0  1  2 ... 31 | 32 33 34 ... 63 | 64 65 ...
//   lane:         0  1  2 ... 31 |  0  1  2 ... 31 |  0  1  ...
//                 <─── warp 0 ──> <─── warp 1 ────> <── warp 2 ──>
//
// 二、WARP_STRIDE_INIT 宏展开说明
//
//   lane          = threadIdx.x & 31 相当于对32取余
//                   当前线程在 warp 内的位置 [0, 31]
//                   决定本线程在每轮循环中访问哪个元素
//
//   warp_in_blk   = threadIdx.x >> 5
//                   当前线程所在 warp 在 block 内的编号
//
//   warps_per_blk = blockDim.x >> 5
//                   每个 block 包含的 warp 数量
//                   blockDim.x = 256 时固定为 8
//
//   warp_global   = blockIdx.x * warps_per_blk + warp_in_blk
//                   当前 warp 的全局唯一编号
//                   自动将块间偏移纳入计算，无需手动处理指针偏移
//                   grid=1 时等价于 warp_in_blk
//                   grid=2 时 block1 的 warp 编号从 8 开始
//
//   n_warps       = gridDim.x * warps_per_blk
//                   全局 warp 总数，作为每轮循环的步进量
//                   grid=1, block=256 时 = 8
//                   grid=2, block=256 时 = 16
//
// 三、stride loop 结构
//
//   for (int base = warp_global * 32; base < n; base += n_warps * 32)
//   {
//       int k = base + lane;
//       if (k < n) { /* 计算 */ }
//   }
//
//   每轮循环：
//     base        → 当前 warp 负责的起始下标
//     base + lane → 本线程负责的元素下标
//     步进量      → n_warps * 32，跳过所有 warp 本轮已覆盖的区间
//
//   示意（grid=1, block=256, n_warps=8, n=512）：
//
//     第1轮：
//       warp0 → k=[0,   31]
//       warp1 → k=[32,  63]
//       ...
//       warp7 → k=[224, 255]
//
//     第2轮（base += 8*32 = 256）：
//       warp0 → k=[256, 287]
//       warp1 → k=[288, 319]
//       ...
//       warp7 → k=[480, 511]   → 全部覆盖，循环结束
//
// 四、三个核心性质
//
//   1. Coalesced Access
//      warp 内 32 个 lane 每轮访问连续的 32 个地址
//      硬件将 32 次访问合并为单次内存事务，带宽利用率最优
//
//   2. 无数据竞争
//      warp_global 全局唯一，每个 warp 负责不重叠的区间
//
//   3. 自适应 grid 大小
//      n_warps 由 gridDim.x 动态推导
//      launch 侧修改 grid 大小时 kernel 代码零修改
//      grid=1：每个 warp 多跑几轮，适合小数据
//      grid>1：warp 并行分摊，适合大数据
//
// 五、适用场景
//
//   适合：1D 连续数组的逐元素操作（填充、缩放、变换）
//   不适合：需要跨 warp 通信、二维索引、或非连续访存的场景
//
// =============================================================================
#define WARP_STRIDE_INIT()                                              \
    const int lane        = threadIdx.x & 31;                          \
    const int warp_in_blk = threadIdx.x >> 5;                          \
    const int warps_per_blk = (blockDim.x + 31) >> 5;                   \
    const int warp_global = blockIdx.x * warps_per_blk + warp_in_blk; \
    const int n_warps     = gridDim.x  * warps_per_blk;

            // (0) 填充常数权重 — warp stride 版
            static __global__ void kernel_fill_identity_weights(
                float* __restrict__ w,
                int n_complex, int N, float gain, bool bake_invN)
            {
                WARP_STRIDE_INIT()

                    // 值全部相同，host 端预算好，所有线程只做写操作
                    float val = gain * ((bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f);

                for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                {
                    int k = base + lane;
                    if (k < n_complex) w[k] = val;
                }
            }

            // (0b) 原地增益缩放
            // 用途：None 情况下的 apply_window 退化路径
            static __global__ void kernel_scale_inplace(
                float* __restrict__ w,
                int n_complex, float gain)
            {
                WARP_STRIDE_INIT()

                    for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                    {
                        int k = base + lane;
                        if (k < n_complex) w[k] *= gain;
                    }
            }

            // (1) 频域直接构建滤波权重 — warp stride 版
            static __global__ void kernel_build_weights_analytic_freq(
                float* __restrict__ w,
                int n_complex, int N,
                SFilterKernelDesc desc, bool bake_invN)
            {
                const int lane = threadIdx.x & 31;
                const int warp_in_blk = threadIdx.x >> 5;
                const int warps_per_blk = (blockDim.x + 31) >> 5;
                const int warp_global = blockIdx.x * warps_per_blk + warp_in_blk;
                const int n_warps = gridDim.x * warps_per_blk;

                float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
                float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;

                for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                {
                    int k = base + lane;
                    if (k >= n_complex) break;

                    float f = (N > 0) ? ((float)k / (float)N) : 0.0f;

                    if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; continue; }
                    if (f > cc) { w[k] = 0.0f; continue; }

                    w[k] = desc.gain * f * window_shape_desc(f / cc, desc) * invN;
                }
            }


            // ----------------------------------------------------------------
            // kernel_gen_spatial_rl_kernel_du1
            //
            // 生成离散 Ram-Lak 空域核，长度 N（paddedN），纯数字离散，不含 du。
            //
            // 数学背景：
            //   连续 Ram-Lak 核采样值（t = n*du）：
            //     h(0)    = 1 / (4*du^2)
            //     h(n奇)  = -1 / (pi^2 * n^2 * du^2)
            //
            //   正确离散化需乘以黎曼步长 du（连续积分 → 离散求和）：
            //     h[n] = h(n*du) * du
            //     h[0]    = 1 / (4*du)
            //     h[n奇]  = -1 / (pi^2 * n^2 * du)
            //
            //   本函数省略了 du，输出的是纯数字核：
            //     h̃[0]    = 0.25          = h[0]  * du
            //     h̃[n奇]  = -1/(pi^2*n^2) = h[n奇]* du
            //
            //   即 h̃[n] = h[n] * du，比正确离散核少一个 du。
            //
            // 缺失补偿：
            //   FFT 卷积本身也缺少黎曼步长 du（离散求和 vs 连续积分），
            //   净缺因子为 du（两个 du 抵消一个）。
            //   由调用方在卷积后施加 postScale = 1/du 补偿。
            //
            // bake_invN：
            //   将 1/N 烘焙进核，抵消 cuFFT C2R IFFT 的 N 倍放大，
            //   避免在滤波后单独做一次全局缩放。
            // ----------------------------------------------------------------
            // (2) 空域离散 Ram-Lak 核 — warp stride 版
            static __global__ void kernel_gen_spatial_rl_kernel_du1(
                float* __restrict__ h, int N, bool bake_invN)
            {
                // ----------------------------------------------------------------
                // 离散 Ram-Lak 空域核，du=1 特化版本
                //
                // 推导：
                //   连续核采样：h(n*du) = 1/(4*du^2)        n=0
                //                        -1/(pi^2*n^2*du^2)  n 奇
                //
                //   离散化（乘黎曼步长 du）：
                //     h[n] = h(n*du) * du = 1/(4*du)         n=0
                //                          -1/(pi^2*n^2*du)   n 奇
                //
                //   令 du=1，得到纯数字离散核：
                //     h[n] = 0.25                             n=0
                //     h[n] = -1/(pi^2*n^2)                   n 奇
                //     h[n] = 0                                n 偶, n≠0
                //
                // 使用方须知：
                //   实际探测器间距为 du_real != 1 时，卷积结果少一个 du_real 因子，
                //   调用方需在滤波后施加 postScale = 1/du_real 补偿。
                // ----------------------------------------------------------------
                constexpr float du = 1.f;   // 本核以 du=1 为单位，非省略
                constexpr float pi = 3.14159265358979323846f;

                float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;

                WARP_STRIDE_INIT()

                    for (int base = warp_global * 32; base < N; base += n_warps * 32)
                    {
                        int u = base + lane;
                        if (u >= N) break;

                        int n = (u <= N / 2) ? u : (u - N);
                        int an = (n < 0) ? -n : n;

                        float val = 0.0f;
                        if (n == 0) {
                            val = 1.0f / (4.0f * du);                      // = 0.25
                        }
                        else if (an & 1) {
                            float fn = (float)n;
                            val = -1.0f / (pi * pi * fn * fn * du);        // = -1/(pi^2*n^2)
                        }

                        h[u] = val * invN;
                    }
            }

            static __global__ void kernel_build_spatial_ramp(
                float* __restrict__ h, int N,
                const float* __restrict__ ramp, int ramp_size,
                bool bake_invN)
            {
                WARP_STRIDE_INIT()
                const int center = ramp_size / 2;
                const float invN = (bake_invN && N > 0) ? 1.0f / (float)N : 1.0f;
                for (int base = warp_global * 32; base < N; base += n_warps * 32) {
                    const int u = base + lane;
                    if (u >= N) break;
                    float value = 0.0f;
                    for (int j = 0; j < ramp_size; ++j) {
                        int index = j - center;
                        if (index < 0) index += N;
                        if (u == index) value = ramp[j];
                    }
                    h[u] = value * invN;
                }
            }

            // (3) 从 FFT(RL) 提取实数权重 — warp stride 版
            static __global__ void kernel_extract_weights_from_fft(
                const cufftComplex* __restrict__ src,
                float* __restrict__ dst,
                int n_complex, ERampExtractMode mode, bool force_dc_zero)
            {
                WARP_STRIDE_INIT()

                    // 用 float2 读保证 64-bit LD，与 cufftComplex 内存布局完全一致
                    const float2* src2 = reinterpret_cast<const float2*>(src);

                for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                {
                    int k = base + lane;
                    if (k >= n_complex) break;

                    float2 c = src2[k];                             // 64-bit LD
                    float  v = (mode == ERampExtractMode::Magnitude) ? hypotf(c.x, c.y) : c.x;

                    if (force_dc_zero && k == 0) v = 0.0f;
                    dst[k] = v;
                }
            }

            // (4) 原地施加窗函数 — warp stride 版
            static __global__ void kernel_apply_window_to_weights_inplace(
                float* __restrict__ w,
                int n_complex, int N, SFilterKernelDesc desc)
            {
                const int lane = threadIdx.x & 31;
                const int warp_in_blk = threadIdx.x >> 5;
                const int warps_per_blk = (blockDim.x + 31) >> 5;
                const int warp_global = blockIdx.x * warps_per_blk + warp_in_blk;
                const int n_warps = gridDim.x * warps_per_blk;

                float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;

                for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                {
                    int k = base + lane;
                    if (k >= n_complex) break;

                    float f = (N > 0) ? ((float)k / (float)N) : 0.0f;

                    if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; continue; }
                    if (f > cc) { w[k] = 0.0f; continue; }

                    w[k] *= desc.gain * window_shape(f / cc, desc.kind);
                }
            }

#undef WARP_STRIDE_INIT

        } // namespace detail


        bool flt_launch_kernel_fill_identity_weights(
            float* d_w, int n_complex, int N, float gain, bool bake_invN,
            cudaStream_t stream)
        {
            SKernelLaunchPolicy policy;
            policy.block_threads = 256;
            dim3 block(policy.block_threads, 1, 1);
            dim3 grid(2, 1, 1);

            detail::kernel_fill_identity_weights << <grid, block, 0, stream >> > (
                d_w, n_complex, N, gain, bake_invN);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        bool flt_launch_kernel_build_weights_analytic_freq(
            float* d_w, int n_complex, int N,
            SFilterKernelDesc desc, bool bake_invN, cudaStream_t stream)
        {
            SKernelLaunchPolicy policy;
            policy.block_threads = 256;
            dim3 block(policy.block_threads, 1, 1);
            dim3 grid(2, 1, 1);


            detail::kernel_build_weights_analytic_freq << <grid, block, 0, stream >> > (
                d_w, n_complex, N, desc, bake_invN);

            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }


        bool flt_launch_kernel_gen_spatial_rl_kernel_du1(
            float* d_h, int N, bool bake_invN, cudaStream_t stream)
        {
            SKernelLaunchPolicy policy;
            policy.block_threads = 256;
            dim3 block(policy.block_threads, 1, 1);
            dim3 grid(2, 1, 1);
            detail::kernel_gen_spatial_rl_kernel_du1 << <grid, block, 0, stream >> > (
                d_h, N, bake_invN);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        bool flt_launch_kernel_build_spatial_ramp(
            float* d_spatial, int N, const float* d_ramp,
            int ramp_size, bool bake_invN, cudaStream_t stream)
        {
            if (!d_spatial || !d_ramp || N <= 0 || ramp_size <= 0)
                return false;
            SKernelLaunchPolicy policy;
            dim3 block(policy.normalizedBlockThreads(), 1, 1);
            dim3 grid(2, 1, 1);
            YK::validateWarpLaunch(block, grid);
            detail::kernel_build_spatial_ramp<<<grid, block, 0, stream>>>(
                d_spatial, N, d_ramp, ramp_size, bake_invN);
            YK_CUDA_KERNEL_CHECK();
            return cudaGetLastError() == cudaSuccess;
        }

        bool flt_launch_kernel_extract_weights_from_fft(
            const cufftComplex* d_src, float* d_dst,
            int n_complex, ERampExtractMode mode, bool force_dc_zero,
            cudaStream_t stream)
        {
            SKernelLaunchPolicy policy;
            policy.block_threads = 256;
            dim3 block(policy.block_threads, 1, 1);
            dim3 grid(2, 1, 1);
            detail::kernel_extract_weights_from_fft << <grid, block, 0, stream >> > (
                d_src, d_dst, n_complex, mode, force_dc_zero);
            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        bool flt_launch_kernel_apply_window_to_weights_inplace(
            float* d_w, int n_complex, int N,
            SFilterKernelDesc desc, cudaStream_t stream)
        {
            SKernelLaunchPolicy policy;
            policy.block_threads = 256;
            dim3 block(policy.block_threads, 1, 1);
            dim3 grid(2, 1, 1);


            detail::kernel_apply_window_to_weights_inplace << <grid, block, 0, stream >> > (
                d_w, n_complex, N, desc);

            YK_CUDA_KERNEL_CHECK();
            return (cudaGetLastError() == cudaSuccess);
        }

        void flt_launch_kernel_scale_inplace(
            float* data, int n, float scale,
            cudaStream_t stream)
        {
            int sm_count = 0;
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

            const int warps_per_blk = 256 / 32;
            const int total_warps = (n + 31) / 32;
            const int blocks_need = (total_warps + warps_per_blk - 1) / warps_per_blk;
            const int blocks = std::min(blocks_need, sm_count * 2);
            dim3 block(256, 1, 1);
            dim3 grid(blocks, 1, 1);
            detail::kernel_scale_inplace << <grid, block, 0, stream >> > (data, n, scale);
        }



    }
}

//namespace YK {
//    namespace Filter {
//
//
//
//        namespace detail { //非warp实现
//
//            // (0) Identity weights: w[k] = gain * (bake_invN ? 1/N : 1)
//            static __global__ void kernel_fill_identity_weights(
//                float* __restrict__ w,
//                int n_complex,
//                int N,
//                float gain,
//                bool bake_invN)
//            {
//                int k = blockIdx.x * blockDim.x + threadIdx.x;
//                if (k >= n_complex) return;
//                float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
//                w[k] = gain * invN;
//            }
//
//            // (1) Analytic frequency-domain build (direct)
//            // w[k] = gain * ramp(f) * window(x) * (optional 1/N), cutoff applied
//            static __global__ void kernel_build_weights_analytic_freq(
//                float* __restrict__ w,
//                int n_complex,
//                int N,
//                SFilterKernelDesc desc,
//                bool bake_invN)
//            {
//                int k = blockIdx.x * blockDim.x + threadIdx.x;
//                if (k >= n_complex) return;
//
//                if (desc.kind == EFilterKernel::None) {
//                    float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
//                    w[k] = desc.gain * invN;
//                    return;
//                }
//
//                float f = (N > 0) ? ((float)k / (float)N) : 0.0f; // [0, 0.5]
//
//                if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; return; }
//
//                float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
//                if (f > cc) { w[k] = 0.0f; return; }
//
//                float ramp = f;
//                float x = (cc > 0.0f) ? (f / cc) : 0.0f;
//                float shape = window_shape(x, desc.kind);
//
//                float invN = (bake_invN && N > 0) ? (1.0f / (float)N) : 1.0f;
//                w[k] = desc.gain * ramp * shape * invN;
//            }
//
//            // (2) Spatial discrete Ram-Lak kernel (DU=1 convention)
//            static __global__ void kernel_gen_spatial_rl_kernel_du1(
//                float* __restrict__ h,
//                int N,
//                bool bake_invN)
//            {
//                int u = blockIdx.x * blockDim.x + threadIdx.x;
//                if (u >= N) return;
//
//                int n = (u <= N / 2) ? u : (u - N);
//                int an = (n < 0) ? -n : n;
//
//                float val = 0.0f;
//                if (n == 0) {
//                    val = 1.0f / 4.0f;
//                }
//                else if (an & 1) {
//                    const float pi = 3.14159265358979323846f;
//                    float fn = (float)n;
//                    val = -1.0f / (pi * pi * fn * fn);
//                }
//
//                if (bake_invN && N > 0) val *= (1.0f / (float)N);
//                h[u] = val;
//            }
//
//            // (3) Extract ramp weights from FFT(RL)
//            static __global__ void kernel_extract_weights_from_fft(
//                const cufftComplex* __restrict__ src,
//                float* __restrict__ dst,
//                int n_complex,
//                int mode,           // 0 = RealPart, 1 = Magnitude
//                bool force_dc_zero)
//            {
//                int k = blockIdx.x * blockDim.x + threadIdx.x;
//                if (k >= n_complex) return;
//
//                float re = src[k].x;
//                float im = src[k].y;
//                float v = (mode == 1) ? sqrtf(re * re + im * im) : re;
//
//                if (force_dc_zero && k == 0) v = 0.0f;
//                dst[k] = v;
//            }
//
//            // (4) Apply window / cutoff / gain / DC on ramp weights (in-place)
//            static __global__ void kernel_apply_window_to_weights_inplace(
//                float* __restrict__ w,
//                int n_complex,
//                int N,
//                SFilterKernelDesc desc)
//            {
//                int k = blockIdx.x * blockDim.x + threadIdx.x;
//                if (k >= n_complex) return;
//
//                if (desc.kind == EFilterKernel::None) {
//                    w[k] = w[k] * desc.gain;
//                    return;
//                }
//
//                float f = (N > 0) ? ((float)k / (float)N) : 0.0f;
//
//                if (desc.force_dc_zero && k == 0) { w[k] = 0.0f; return; }
//
//                float cc = (desc.cutoff > 0.0f) ? desc.cutoff : 0.5f;
//                if (f > cc) { w[k] = 0.0f; return; }
//
//                float x = (cc > 0.0f) ? (f / cc) : 0.0f;
//                float shape = window_shape(x, desc.kind);
//
//                w[k] = w[k] * (desc.gain * shape);
//            }
//
//        }; // namespace detail
//
//
//        bool flt_launch_kernel_fill_identity_weights(
//            float* d_w,
//            int n_complex,
//            int N,
//            float gain,
//            bool bake_invN,
//            cudaStream_t stream)
//        {
//            dim3 block(256, 1);
//            dim3 grid((n_complex + block.x - 1) / block.x, 1);
//            detail::kernel_fill_identity_weights << <grid, block, 0, stream >> > (
//                d_w, n_complex, N, gain, bake_invN);
//            YK_CUDA_KERNEL_CHECK();
//            return (cudaGetLastError() == cudaSuccess);
//        }
//
//        bool flt_launch_kernel_build_weights_analytic_freq(
//            float* d_w,
//            int n_complex,
//            int N,
//            SFilterKernelDesc desc,
//            bool bake_invN,
//            cudaStream_t stream)
//        {
//            dim3 block(256, 1);
//            dim3 grid((n_complex + block.x - 1) / block.x, 1);
//            detail::kernel_build_weights_analytic_freq << <grid, block, 0, stream >> > (
//                d_w, n_complex, N, desc, bake_invN);
//            YK_CUDA_KERNEL_CHECK();
//            return (cudaGetLastError() == cudaSuccess);
//        }
//
//        bool flt_launch_kernel_gen_spatial_rl_kernel_du1(
//            float* d_h,
//            int N,
//            bool bake_invN,
//            cudaStream_t stream)
//        {
//            dim3 block(256, 1);
//            dim3 grid((N + block.x - 1) / block.x, 1);
//            detail::kernel_gen_spatial_rl_kernel_du1 << <grid, block, 0, stream >> > (
//                d_h, N, bake_invN);
//            YK_CUDA_KERNEL_CHECK();
//            return (cudaGetLastError() == cudaSuccess);
//        }
//
//        bool flt_launch_kernel_extract_weights_from_fft(
//            const cufftComplex* d_src,
//            float* d_dst,
//            int n_complex,
//            ERampExtractMode mode,
//            bool force_dc_zero,
//            cudaStream_t stream)
//        {
//            dim3 block(256, 1);
//            dim3 grid((n_complex + block.x - 1) / block.x, 1);
//            detail::kernel_extract_weights_from_fft << <grid, block, 0, stream >> > (
//                d_src, d_dst, n_complex, (int)mode, force_dc_zero);
//            YK_CUDA_KERNEL_CHECK();
//            return (cudaGetLastError() == cudaSuccess);
//        }
//
//
//        bool flt_launch_kernel_apply_window_to_weights_inplace(
//            float* d_w,
//            int n_complex,
//            int N,
//            SFilterKernelDesc desc,
//            cudaStream_t stream)
//        {
//            dim3 block(256, 1);
//            dim3 grid((n_complex + block.x - 1) / block.x, 1);
//            detail::kernel_apply_window_to_weights_inplace << <grid, block, 0, stream >> > (
//                d_w, n_complex, N, desc);
//            YK_CUDA_KERNEL_CHECK();
//            return (cudaGetLastError() == cudaSuccess);
//        }
//
//    }; // namespace Filter
//};// namespace YK

