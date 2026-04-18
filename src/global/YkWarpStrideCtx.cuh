// =============================================================================
// YkWarpStrideCtx.cuh
//
// Warp Stride Loop ������ģ��
//
// ��;��ͳһ���� CUDA kernel �е� warp ���ݼ��㣬�����ظ���д
//       ��������ά��ѡ���Ӧ�ػ����������㿪��
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace YK {

    // -----------------------------------------------------------------------------
    // ����ö��
    //
    //   X       : 1D �������飬������ x ����
    //             ���ã��˲�Ȩ����䡢scale��window �� 1D kernel
    //
    //   XY      : 2D��batch �� blockIdx.y�������� x ���� stride loop
    //             ���ã�pointwise_mul ���� batch ά�ȵ� kernel
    //
    //   RowWarp : �м���ÿ�� warp ��ռһ�У���� stride loop ����
    //             ���ã�preweight �� per-row ���� kernel
    // -----------------------------------------------------------------------------
    enum class EWarpStrideAxis { X, XY, RowWarp };


    // ��ģ�壨δ�ػ�ʱ�����ã�
    template<EWarpStrideAxis Axis>
    struct WarpStrideCtx;


    // -----------------------------------------------------------------------------
    // �ػ���X �� 1D kernel
    //
    // ��Launch ��ʽ �� С���ݣ�n <= 2048��һ���Գ�ʼ������
    //   // �� block �㹻��stride loop �Զ����ָ���
    //   dim3 block(256, 1, 1);
    //   dim3 grid(1, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n);
    //
    //   ���ó�����
    //     �˲�Ȩ�س�ʼ����n = 512~2048��
    //     ����·�������������ȼ��㱾��������
    //     ÿ�߳��� n/256 �֣��� block ���и���
    //
    // ��Launch ��ʽ �� �����ݣ�n > 2048����·������
    //   // �� SM �����̶� grid�������������
    //   static int sm_count = 0;
    //   if (sm_count == 0)
    //       cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    //
    //   const int warps_per_blk = 256 / 32;                               // = 8
    //   const int warps_need    = (n + 31) / 32;
    //   const int blocks_need   = (warps_need + warps_per_blk - 1) / warps_per_blk;
    //   const int grid_x        = min(blocks_need, sm_count * 2);         // ����
    //   dim3 block(256, 1, 1);
    //   dim3 grid(grid_x, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n);
    //
    //   ���ó�����
    //     ���ģ��Ԫ�ز�����n = ��ʮ�����ϣ�
    //     ��·������Ҫ����������� SM
    //     ÿ�߳��� ceil(n / (grid_x*256)) ��
    //
    // ��Kernel ��ʽ��
    //   __global__ void kernel(..., int n)
    //   {
    //       WarpStrideCtx<EWarpStrideAxis::X> ctx;
    //
    //       for (int base = ctx.warp_global * 32; base < n; base += ctx.n_warps * 32)
    //       {
    //           int k = base + ctx.lane;
    //           if (k >= n) break;        // β���߽籣������������
    //           // ... ���� ...
    //       }
    //   }
    //
    // ���߽�˵����
    //   - ѭ������ base < n     : ����Խ��ʱ������ѭ����
    //   - if (k >= n) break     : β�� warp �ڲ��� lane Խ��ʱ����
    //   - ����Ҫ���� return ����: stride loop ��Ȼ���� warp �����������������
    // -----------------------------------------------------------------------------
    template<>
    struct WarpStrideCtx<EWarpStrideAxis::X>
    {
        int lane;
        int warp_global;
        int n_warps;

        __device__ __forceinline__ WarpStrideCtx()
        {
            lane = static_cast<int>(threadIdx.x) & 31;
            int warp_in = static_cast<int>(threadIdx.x) >> 5;
            int nw_blk = static_cast<int>(blockDim.x) >> 5;
            warp_global = static_cast<int>(blockIdx.x) * nw_blk + warp_in;
            n_warps = static_cast<int>(gridDim.x) * nw_blk;
        }
    };


    // -----------------------------------------------------------------------------
    // �ػ���XY �� 2D batch kernel
    //
    // ��Launch ��ʽ �� С���ݣ�n_complex <= 2048��batch ��С����
    //   // x ���� block��y �����Ӧ batch
    //   dim3 block(256, 1, 1);
    //   dim3 grid(1, batch, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n, batch);
    //
    //   ���ó�����
    //     n_complex = 512~2048��batch = ��ʮ������
    //     ÿ�߳��� x ������ n/256 �֣�y ���� 1:1 ��Ӧ batch
    //     batch ��Сʱ gridDim.y ���ᳬ��Ӳ�����ƣ�65535��
    //
    // ��Launch ��ʽ �� �����ݣ�n_complex > 2048 �� batch ���󣩡�
    //   static int sm_count = 0;
    //   if (sm_count == 0)
    //       cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    //
    //   const int warps_per_blk = 256 / 32;
    //   const int warps_need    = (n + 31) / 32;
    //   const int blocks_need   = (warps_need + warps_per_blk - 1) / warps_per_blk;
    //   const int grid_x        = min(blocks_need, sm_count * 2);         // x ����
    //   const int grid_y        = min(batch, 65535);                      // y ����
    //   dim3 block(256, 1, 1);
    //   dim3 grid(grid_x, grid_y, 1);
    //   kernel<<<grid, block, 0, stream>>>(..., n, batch);
    //
    //   ע�⣺batch ���� 65535 ʱ��Ҫ�� kernel �ڶ� b Ҳ�� stride loop
    //         ��ǰ�ػ������� b �� stride������ batch �赥������
    //
    // ��Kernel ��ʽ��
    //   __global__ void kernel(..., int n, int batch)
    //   {
    //       WarpStrideCtx<EWarpStrideAxis::XY> ctx;
    //       if (ctx.b >= batch) return;   // y ���򶥲�Խ�磺�����߳�������
    //
    //       for (int base = ctx.warp_global * 32; base < n; base += ctx.n_warps * 32)
    //       {
    //           int u = base + ctx.lane;
    //           if (u >= n) break;         // x ����β��Խ��
    //           int idx = ctx.b * n + u;
    //           // ... ���� ...
    //       }
    //   }
    //
    // ���߽�˵����
    //   - if (ctx.b >= batch) return : y ���� gridDim.y ���� batch ʱ��
    //                                  �����߳̿�������ֱ���˳�
    //   - if (u >= n) break          : x ����β�� lane Խ�磬����ѭ��
    //   - y ������ return��x ������ break
    //     ԭ��y Խ����ζ�������߳����κ�����
    //           x Խ��ֻ�ǵ�ǰ��β���������ִο����������ݣ�stride loop��
    // -----------------------------------------------------------------------------
    template<>
    struct WarpStrideCtx<EWarpStrideAxis::XY>
    {
        int b;
        int lane;
        int warp_global;
        int n_warps;

        __device__ __forceinline__ WarpStrideCtx()
        {
            b = static_cast<int>(blockIdx.y);
            lane = static_cast<int>(threadIdx.x) & 31;
            int warp_in = static_cast<int>(threadIdx.x) >> 5;
            int nw_blk = static_cast<int>(blockDim.x) >> 5;
            warp_global = static_cast<int>(blockIdx.x) * nw_blk + warp_in;
            n_warps = static_cast<int>(gridDim.x) * nw_blk;
        }
    };


    // -----------------------------------------------------------------------------
    // �ػ���RowWarp �� �м� per-row kernel
    //
    // ��Launch ��ʽ �� С���ݣ�K*Nv ��С������ <= sm_count*warps_per_blk����
    //   // ��ȷ���䣬ÿ��ǡ��һ�� warp���� stride
    //   const int total_rows    = K * Nv;
    //   const int warps_per_blk = 256 / 32;                               // = 8
    //   const int blocks        = (total_rows + warps_per_blk - 1) / warps_per_blk;
    //   dim3 block(256, 1, 1);
    //   dim3 grid(blocks, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(...);
    //
    //   ���ó�����
    //     K*Nv ��С��block ���������
    //     ÿ�� warp ֻ��һ�У���� stride loop ִֻ��һ��
    //     Nu �ϴ�ʱ���� lane loop �����㹻������
    //
    // ��Launch ��ʽ �� �����ݣ�K*Nv �ܴ���Ҫ���� block ������
    //   static int sm_count = 0;
    //   if (sm_count == 0)
    //       cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    //
    //   const int total_rows    = K * Nv;
    //   const int warps_per_blk = 256 / 32;
    //   const int blocks_need   = (total_rows + warps_per_blk - 1) / warps_per_blk;
    //   const int blocks        = min(blocks_need, sm_count * 2);         // ����
    //   dim3 block(256, 1, 1);
    //   dim3 grid(blocks, 1, 1);
    //   kernel<<<grid, block, 0, stream>>>(...);
    //
    //   ���ó�����
    //     K = ����֡��Nv = �����У�K*Nv = ���򵽼�ʮ��
    //     �����޺� block ���̶���stride loop ��̯ʣ����
    //     geo[i] ������ȡ�� K ��С����������� L1 cache
    //
    // ��Kernel ��ʽ��
    //   __global__ void kernel(..., int Nu, int Nv, int K)
    //   {
    //       WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;
    //
    //       // ��㣺���� stride loop��һ�� warp ��������
    //       for (int row = ctx.warp_global; row < K * Nv; row += ctx.n_warps)
    //       {
    //           int i = row / Nv;
    //           int v = row - i * Nv;
    //
    //           if (i >= K) continue;      // �����Լ�飬�����쳣�У�����������
    //
    //           // ÿ�����¼����м��������� geo[i]��...
    //
    //           // �ڲ㣺���� lane loop�������̶� 32
    //           for (int u = ctx.lane; u < Nu; u += 32)
    //           {
    //               // ... ���� ...
    //           }                          // ����������ʽԽ���飬ѭ��������֤
    //       }
    //   }
    //
    // ���߽�˵����
    //   - ���ѭ������ row < K*Nv : �б��Խ��ʱ������ѭ������Ȼ�˳�
    //   - if (i >= K) continue    : �����Ա������쳣���������� return
    //                               ԭ�򣺺��������账����return �ᵼ��©����
    //   - �ڲ�ѭ������ u < Nu     : ������Ȼ�߽磬������� break
    //   - ���ڶ����� return ����  : �� X/XY ��ͬ������ return �ᵼ��
    //                               ������©����
    //
    // ��С���� vs �����ݵĺ�������
    //   С���ݣ�blocks ��ȷ���ǣ�ÿ warp ֻ��һ�У�stride loop �˻�Ϊ����
    //   �����ݣ�blocks �����ޣ�ÿ warp �ܶ��У�stride loop ��̯ʣ��
    //   ������� kernel ������ȫ��ͬ��ֻ�� launch ��� blocks ���㲻ͬ
    //
    // ���� X/XY �Ĺؼ�����
    //   X/XY    : Ԫ�ؼ� stride��warp �����������Ԫ�أ�Խ���� break
    //   RowWarp : �м� stride��warp �������У�����Э����Խ���� continue
    // -----------------------------------------------------------------------------
    template<>
    struct WarpStrideCtx<EWarpStrideAxis::RowWarp>
    {
        int lane;
        int warp_global;
        int n_warps;

        __device__ __forceinline__ WarpStrideCtx()
        {
            lane = static_cast<int>(threadIdx.x) & 31;
            int warp_in = static_cast<int>(threadIdx.x) >> 5;
            int nw_blk = static_cast<int>(blockDim.x) >> 5;
            warp_global = static_cast<int>(blockIdx.x) * nw_blk + warp_in;
            n_warps = static_cast<int>(gridDim.x) * nw_blk;
        }
    };

} // namespace YK