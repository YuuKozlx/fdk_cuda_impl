// YkAlignPadCropVec.hpp
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "YkGlobals.h" // SKernelLaunchPolicy, YK_CUDA_CHECK, YK_CUDA_KERNEL_CHECK, etc.
#include "YkFDKFilter2.hpp"

namespace YK {
    namespace cg = cooperative_groups;

    // ============================================================
    // Policy (same style as preweight)
    // ============================================================
    inline SKernelLaunchPolicy normalizeAlignPadCropPolicy(SKernelLaunchPolicy p) {
        if (p.block_threads < 32) p.block_threads = 32;
        p.block_threads = (p.block_threads + 31) & ~31; // multiple of 32
        p.block_threads = std::min(p.block_threads, 1024);
        return p;
    }

    // ============================================================
    // helpers
    // ============================================================
    __host__ __device__ __forceinline__ int computePaddedN_nextPow2_2Nu(int Nu) {
        int need = 2 * Nu;
        int n = 1;
        while (n < need) n <<= 1;
        return n;
    }

    __host__ __device__ __forceinline__ int computeStartU_centerAxis(int Nu, int paddedN, float offsetU_pix) {
        // axis_idx = (Nu-1)/2 + offsetU_pix
        const float axis_idx = (Nu - 1) * 0.5f + offsetU_pix;
        return (int)lrintf(paddedN * 0.5f - axis_idx);
    }

    // ============================================================
    // row-warp pad kernel
    //  - 1 warp handles one row: row = i*Nv + v
    //  - lane handles u with stride 32
    // ============================================================
    __global__ void align_pad_chunk_rowwarp_kernel(
        const float* __restrict__ src,       // [K*Nv*Nu]
        float* __restrict__ dst,             // [K*Nv*paddedN]
        const int* __restrict__ startu,      // [K]
        int Nu, int Nv, int paddedN,
        int K,
        int bounds_check)                    // 0/1
    {
        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

        const int warps_per_block = int(tb.size() / 32);
        const int warp_id_in_block = int(tb.thread_rank() / 32);
        const int warp_global = int(blockIdx.x) * warps_per_block + warp_id_in_block;

        const int total_warps = K * Nv; // rows
        if (warp_global >= total_warps) return;

        const int i = warp_global / Nv;
        const int v = warp_global - i * Nv;
        const int sU = startu[i];

        const size_t base_src = ((size_t)i * (size_t)Nv + (size_t)v) * (size_t)Nu;
        const size_t base_dst = ((size_t)i * (size_t)Nv + (size_t)v) * (size_t)paddedN;

        for (int u = (int)warp.thread_rank(); u < paddedN; u += 32) {
            int su = u - sU;
            float val = 0.0f;

            if (!bounds_check) {
                // caller guarantees su in [0,Nu)
                val = src[base_src + (size_t)su];
            }
            else {
                if ((unsigned)su < (unsigned)Nu) val = src[base_src + (size_t)su];
            }

            dst[base_dst + (size_t)u] = val;
        }
    }

    // ============================================================
    // row-warp crop kernel
    //  - src: [K*Nv*paddedN]
    //  - dst: [K*Nv*Nu]
    // ============================================================
    __global__ void align_crop_chunk_rowwarp_kernel(
        const float* __restrict__ src,       // [K*Nv*paddedN]
        float* __restrict__ dst,             // [K*Nv*Nu]
        const int* __restrict__ startu,      // [K]
        int Nu, int Nv, int paddedN,
        int K,
        int bounds_check)                    // 0/1
    {
        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

        const int warps_per_block = int(tb.size() / 32);
        const int warp_id_in_block = int(tb.thread_rank() / 32);
        const int warp_global = int(blockIdx.x) * warps_per_block + warp_id_in_block;

        const int total_warps = K * Nv;
        if (warp_global >= total_warps) return;

        const int i = warp_global / Nv;
        const int v = warp_global - i * Nv;
        const int sU = startu[i];

        const size_t base_src = ((size_t)i * (size_t)Nv + (size_t)v) * (size_t)paddedN;
        const size_t base_dst = ((size_t)i * (size_t)Nv + (size_t)v) * (size_t)Nu;

        for (int u = (int)warp.thread_rank(); u < Nu; u += 32) {
            int su = sU + u;
            float val = 0.0f;

            if (!bounds_check) {
                // caller guarantees su in [0,paddedN)
                val = src[base_src + (size_t)su];
            }
            else {
                if ((unsigned)su < (unsigned)paddedN) val = src[base_src + (size_t)su];
            }

            dst[base_dst + (size_t)u] = val;
        }
    }

    // ============================================================
    // AlignPadCropManagerVec (chunk-only)
    //  - K=1 covers single-view usage
    //  - stores host startU/offsetU lists for debugging
    //  - maintains device startU buffer for row-warp kernels
    // ============================================================
    class AlignPadCropManagerVec {
    public:
        AlignPadCropManagerVec() = default;
        ~AlignPadCropManagerVec() { release(); }

        AlignPadCropManagerVec(const AlignPadCropManagerVec&) = delete;
        AlignPadCropManagerVec& operator=(const AlignPadCropManagerVec&) = delete;

        AlignPadCropManagerVec(AlignPadCropManagerVec&& o) noexcept { move_from_(o); }
        AlignPadCropManagerVec& operator=(AlignPadCropManagerVec&& o) noexcept {
            if (this != &o) { release(); move_from_(o); }
            return *this;
        }

        // policy (same pattern as preweight)
        void setPolicy(SKernelLaunchPolicy p) { policy_ = normalizeAlignPadCropPolicy(p); }
        SKernelLaunchPolicy policy() const { return policy_; }

        // ------------------------------------------------------------
        // init (thin)
        // chunkCapacity: optional pre-alloc for startU device buffer
        // ------------------------------------------------------------
        bool init(int Nu, int Nv, int chunkCapacity = 1, cudaStream_t stream = (cudaStream_t)0) {
            if (Nu <= 0 || Nv <= 0) return false;
            return init_(Nu, Nv, chunkCapacity, stream);
        }

        bool init(const SDimensions3D& dims, int chunkCapacity = 1, cudaStream_t stream = (cudaStream_t)0) {
            const int Nu = (int)dims.iPU;
            const int Nv = (int)dims.iPV;
            if (Nu <= 0 || Nv <= 0) return false;
            return init_(Nu, Nv, chunkCapacity, stream);
        }

        // query
        int Nu() const { return Nu_; }
        int Nv() const { return Nv_; }
        int paddedN() const { return paddedN_; }
        int chunkK() const { return chunk_K_; }

        const std::vector<int>& chunkStartU() const { return chunk_startu_; }
        const std::vector<float>& chunkOffsetU() const { return chunk_offsetu_; }

        // ------------------------------------------------------------
        // Chunk pad (one call, K views)
        //  src_chunk: [K*Nv*Nu]
        //  dst_padded_chunk: [K*Nv*paddedN]
        //  offsetU_pix_list: host pointer size K
        //
        // K=1 即单张
        // ------------------------------------------------------------
        bool padChunk(
            const float* d_src_chunk,
            float* d_dst_padded_chunk,
            const float* offsetU_pix_list, // host, size K
            int K)
        {
            if (!inited_ || !d_src_chunk || !d_dst_padded_chunk || !offsetU_pix_list) return false;
            if (K <= 0) return false;

            beginChunk_(K);
            ensureChunkCapacity_(K);

            // compute host lists
            for (int i = 0; i < K; ++i) {
                const float off = offsetU_pix_list[i];
                chunk_offsetu_[(size_t)i] = off;
                chunk_startu_[(size_t)i] = computeStartU_centerAxis(Nu_, paddedN_, off);
            }

            // upload startU list
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_chunk_startu_,
                chunk_startu_.data(),
                (size_t)K * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_));

            // row-warp launch
            const int blockThreads = policy_.block_threads;
            const int warps_per_block = blockThreads / 32;
            const int total_warps = K * Nv_;
            const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

            align_pad_chunk_rowwarp_kernel << <blocks, blockThreads, 0, stream_ >> > (
                d_src_chunk, d_dst_padded_chunk, d_chunk_startu_,
                Nu_, Nv_, paddedN_, K,
                policy_.bounds_check ? 1 : 0);
            YK_CUDA_KERNEL_CHECK();

            chunk_K_ = K;
            return true;
        }

        bool padChunk(
            const float* d_src_chunk,
            float* d_dst_padded_chunk,
            const std::vector<float>& offsetU_pix_list
        )
        {
            return padChunk(d_src_chunk, d_dst_padded_chunk,
                offsetU_pix_list.data(), (int)offsetU_pix_list.size());
        }

        // ------------------------------------------------------------
        // Chunk crop (one call, K views)
        //  src_padded_chunk: [K*Nv*paddedN]
        //  dst_chunk: [K*Nv*Nu]
        //
        // uses d_chunk_startu_ from last padChunk()
        // ------------------------------------------------------------
        bool cropChunk(
            const float* d_src_padded_chunk,
            float* d_dst_chunk,
            cudaStream_t stream = 0)
        {
            if (!inited_ || !d_src_padded_chunk || !d_dst_chunk) return false;

            const int K = chunk_K_;
            if (K <= 0) return false;                 // 没 padChunk 过就 crop：直接失败
            if (!d_chunk_startu_) return false;       // 防御

            // capacity 理论上 padChunk 已 ensure 过，这里可以不再 ensure
            // 但保留也无妨（不会改变 d_chunk_startu_ 的内容）
            ensureChunkCapacity_(K);

            const int blockThreads = policy_.block_threads;
            const int warps_per_block = blockThreads / 32;
            const int total_warps = K * Nv_;
            const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

            align_crop_chunk_rowwarp_kernel << <blocks, blockThreads, 0, stream >> > (
                d_src_padded_chunk, d_dst_chunk, d_chunk_startu_,
                Nu_, Nv_, paddedN_, K,
                policy_.bounds_check ? 1 : 0);
            YK_CUDA_KERNEL_CHECK();

            return true;
        }

        // cleanup
        void release()
        {
            if (d_chunk_startu_) {
                cudaFree(d_chunk_startu_);
                d_chunk_startu_ = nullptr;
            }
            chunk_capacity_ = 0;

            chunk_K_ = 0;
            chunk_startu_.clear();
            chunk_offsetu_.clear();

            inited_ = false;
            Nu_ = Nv_ = paddedN_ = 0;
        }

    private:
        bool init_(int Nu, int Nv, int chunkCapacity, cudaStream_t stream)
        {
            release();

            Nu_ = Nu;
            Nv_ = Nv;
            stream_ = stream;
            paddedN_ = computePaddedN_nextPow2_2Nu(Nu_);
            inited_ = (paddedN_ >= Nu_);

            policy_ = normalizeAlignPadCropPolicy(policy_);

            if (chunkCapacity > 0) ensureChunkCapacity_(chunkCapacity);
            return inited_;
        }

        void beginChunk_(int K)
        {
            chunk_startu_.assign((size_t)K, 0);
            chunk_offsetu_.assign((size_t)K, 0.0f);
        }

        void ensureChunkCapacity_(int K)
        {
            if (K <= 0) return;
            if (K <= chunk_capacity_ && d_chunk_startu_) return;

            int newCap = std::max(1, chunk_capacity_);
            while (newCap < K) newCap <<= 1;

            int* newBuf = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&newBuf, (size_t)newCap * sizeof(int)));

            if (d_chunk_startu_) cudaFree(d_chunk_startu_);
            d_chunk_startu_ = newBuf;
            chunk_capacity_ = newCap;
        }

        void move_from_(AlignPadCropManagerVec& o) noexcept
        {
            Nu_ = o.Nu_;
            Nv_ = o.Nv_;
            paddedN_ = o.paddedN_;
            inited_ = o.inited_;

            policy_ = o.policy_;

            chunk_K_ = o.chunk_K_;
            chunk_capacity_ = o.chunk_capacity_;
            chunk_startu_ = std::move(o.chunk_startu_);
            chunk_offsetu_ = std::move(o.chunk_offsetu_);
            d_chunk_startu_ = o.d_chunk_startu_;

            o.d_chunk_startu_ = nullptr;
            o.chunk_capacity_ = 0;
            o.chunk_K_ = 0;
            o.inited_ = false;
            o.Nu_ = o.Nv_ = o.paddedN_ = 0;
        }

    private:
        // cached dims
        int Nu_ = 0;
        int Nv_ = 0;
        int paddedN_ = 0;
        bool inited_ = false;
        cudaStream_t stream_;

        // policy like preweight
        SKernelLaunchPolicy policy_{};

        // chunk state (host cached)
        int chunk_K_ = 0;
        std::vector<int>   chunk_startu_;
        std::vector<float> chunk_offsetu_;

        // device buffer for startU
        int chunk_capacity_ = 0;
        int* d_chunk_startu_ = nullptr; // [capacity]
    };

} // namespace YK
