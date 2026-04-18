#include <cuda_runtime_api.h>
#include "../global/YkGlobals.h"
#include "YkConv.hpp"
#include <algorithm>

namespace YK {
    namespace Filter {
        namespace detail {
            // =============================================================================
            // pointwise_mul kernels
            // ���ܣ��� batch �鸴�����ݵ�ÿ������Զ�Ӧ��ʵ��Ȩ��
            //       data[b][u] *= weights[u]   (u = 0 .. n_complex-1, b = 0 .. batch-1)
            //
            // �ṩ�ĸ��汾����������������
            //   v1 - ������       : 1�̴߳���1��������ֱ��ӳ��
            //   v2 - warp stride  : warp����ô棬stride loop��������ߴ�
            //   v3 - float2 ������: ����64-bit LD/ST�������ڴ�����
            //   v4 - float4 ������: ����128-bit LD/ST��ÿ�̴߳���2������
            // =============================================================================


                // -----------------------------------------------------------------------------
                // v1: ������
                // �߳�ӳ�䣺blockIdx.x * blockDim.x + threadIdx.x �� ֱ�Ӷ�Ӧ u�������±꣩
                //           blockIdx.y                              �� ��Ӧ batch ά�� b
                // ȱ�㣺n_complex �ϴ�ʱ��Ҫ��� block��grid �����ݳߴ�仯
                // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v1(
                cufftComplex* data,        // [batch, n_complex] �������飨in/out��
                const float* weights,     // [n_complex]        ʵ��Ȩ�أ�ֻ����
                int            n_complex,   // ÿ�����ݵĸ�������
                int            batch)       // ��������
            {
                int u = blockIdx.x * blockDim.x + threadIdx.x;  // �����±�
                int b = blockIdx.y;                              // batch �±�

                if (u < n_complex && b < batch)
                {
                    int idx = b * n_complex + u;    // չƽ��������±�
                    float w = weights[u];
                    data[idx].x *= w;               // ʵ��
                    data[idx].y *= w;               // �鲿
                }
            }


#define WARP_STRIDE_INIT()                                               \
    const int lane          = threadIdx.x & 31;                          \
    const int warp_in_blk   = threadIdx.x >> 5;                          \
    const int warps_per_blk = blockDim.x  >> 5;                          \
    const int warp_global   = blockIdx.x * warps_per_blk + warp_in_blk; \
    const int n_warps       = gridDim.x  * warps_per_blk;
#undef WARP_STRIDE_INIT



#define WARP_STRIDE_INIT_BATCH()                                          \
    const int b             = blockIdx.y;                                 \
    const int lane          = threadIdx.x & 31;                           \
    const int warp_in_blk   = threadIdx.x >> 5;                           \
    const int warps_per_blk = blockDim.x  >> 5;                           \
    const int warp_global   = blockIdx.x * warps_per_blk + warp_in_blk;  \
    const int n_warps       = gridDim.x  * warps_per_blk;
            // -----------------------------------------------------------------------------
            // v2: warp stride ��
            // ���ĸĶ����� warp��32�̣߳�Ϊ��λ�� stride loop
            //   - warp �� 32 ���̴߳��������� 32 ������ �� ��֤ coalesced access
            //   - stride = n_warps * 32��ÿ�ֲ���һ�� block �ĸ��ǿ���
            //   - grid.x �̶�Ϊ 1������ n_complex �仯
            //
            // �߳̽�ɫ��
            //   lane    = threadIdx.x % 32   �� warp ��ƫ�ƣ�0~31��
            //   warp_id = threadIdx.x / 32   �� �� block �ڵڼ��� warp
            // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v2(
                cufftComplex* data,
                const float* weights,
                int            n_complex,
                int            batch)
            {
                WARP_STRIDE_INIT_BATCH()
                    if (b >= batch) return;

                for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                {
                    int u = base + lane;
                    if (u < n_complex)
                    {
                        int   idx = b * n_complex + u;
                        float w = weights[u];
                        data[idx].x *= w;
                        data[idx].y *= w;
                    }
                }
            }

            // -----------------------------------------------------------------------------
            // v3: float2 ��������
            // cufftComplex �����Ͼ��� float2��x=ʵ��, y=�鲿����
            // ��ָ���ؽ���Ϊ float2* �󣬱����������ɵ��� 64-bit ָ�
            //   LDG.E.64��һ�ζ� 8 �ֽڣ� / STG.E.64��һ��д 8 �ֽڣ�
            // ��� v2 ������ 32-bit ���ʣ��ڴ�������� 50%
            //
            // ���� stride loop �߼��� v2 ��ȫ��ͬ
            // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v3(
                float2* data,        // �� cufftComplex* �ȼۣ���ʽ�� float2 ����������
                const float* weights,
                int            n_complex,
                int            batch)
            {
                WARP_STRIDE_INIT_BATCH()

                    if (b >= batch) return;


                for (int base = warp_global * 32; base < n_complex; base += n_warps * 32)
                {
                    int u = base + lane;
                    if (u < n_complex)
                    {
                        int    idx = b * n_complex + u;

                        float2 c = data[idx];           // 64-bit LD��һ�ζ�ȡ��������
                        float  w = weights[u];

                        c.x *= w;                       // ʵ��
                        c.y *= w;                       // �鲿

                        data[idx] = c;                  // 64-bit ST��һ��д����������
                    }
                }
            }

            // -----------------------------------------------------------------------------
            // v4: float4 ��������
            // float4 = 16�ֽ� = 2�� cufftComplex
            // �� data �ؽ���Ϊ float4* �󣬵��� 128-bit LD/ST ��������������
            //   float4.x, .y �� ��һ��������ʵ��, �鲿��
            //   float4.z, .w �� �ڶ���������ʵ��, �鲿��
            //
            // ��������±궼�� n2 = n_complex/2 Ϊ��׼��n_complex ����Ϊż��
            //
            // Ȩ��������
            //   u ��Ӧ float4 �±꣬����ԭʼ���� [u*2, u*2+1]
            //   �� weights[u*2]   ���� .x .y
            //   �� weights[u*2+1] ���� .z .w
            // -----------------------------------------------------------------------------
            __global__ void _kernel_pointwise_mul_v4(
                float4* data,        // �ؽ��ͺ��ָ�룬ÿԪ�ظ���2������
                const float* weights,     // ����ԭʼ weights[n_complex]������ȡ����
                int            n2,          // = n_complex / 2��float4 Ԫ������
                int            batch)
            {
                WARP_STRIDE_INIT_BATCH()

                    if (b >= batch) return;


                // stride loop �� v2/v3 ��ͬ��ֻ��Ԫ�ص�λ��"1������"��Ϊ"2������"
                for (int base = warp_global * 32; base < n2; base += n_warps * 32)
                {
                    int u = base + lane;                // float4 �±�
                    if (u < n2)
                    {
                        int    idx = b * n2 + u;

                        float4 c = data[idx];          // 128-bit LD��һ�ζ�ȡ 2 ������

                        // u ��Ӧԭʼ�����±� u*2 �� u*2+1����ȡһ��Ȩ��
                        float  w0 = weights[u * 2];     // ��һ��������Ȩ��
                        float  w1 = weights[u * 2 + 1]; // �ڶ���������Ȩ��

                        c.x *= w0;  c.y *= w0;          // ��һ��������ʵ�����鲿
                        c.z *= w1;  c.w *= w1;          // �ڶ���������ʵ�����鲿

                        data[idx] = c;                  // 128-bit ST��һ��д�� 2 ������
                    }
                }
            }

#undef WARP_STRIDE_INIT_BATCH

        } // namespace detail

        void launch_pointwise_mul(
            cufftComplex* data,
            const float* weights,
            int            n_complex,
            int            batch,
            cudaStream_t   stream)
        {
            SKernelLaunchPolicy policy;
            dim3 block(policy.block_threads, 1);

            // fft������Ϊ 2 �ı��� �����f4�棬�������f2��
            const int n_elem = (n_complex % 2 == 0) ? n_complex / 2 : n_complex;
            const int warps_per_blk = policy.block_threads >> 5;
            const int warps_need = (n_elem + 31) / 32;
            const int grid_x = std::min((warps_need + warps_per_blk - 1) / warps_per_blk, 8);
            dim3 grid(grid_x, batch);

            if (n_complex % 2 == 0)
                detail::_kernel_pointwise_mul_v4 << <grid, block, 0, stream >> > (
                    reinterpret_cast<float4*>(data), weights, n_complex / 2, batch);
            else
                detail::_kernel_pointwise_mul_v3 << <grid, block, 0, stream >> > (
                    reinterpret_cast<float2*>(data), weights, n_complex, batch);

            YK_CUDA_CHECK(cudaGetLastError());
        }

    }
}
