#pragma once
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFDKVecPreWeight.hpp"
#include "YkFDKVecAlignPadCrop.hpp"
#include "YkFDKFilter.hpp"

// ============================================================
// Online FDK logging switch
// ============================================================
#ifndef YK_FDK_ONLINE_LOG
#define YK_FDK_ONLINE_LOG 0
#endif

#if YK_FDK_ONLINE_LOG
#define YK_FDK_LOGI(fmt, ...) std::printf("[FDK][ONLINE][I] " fmt "\n", ##__VA_ARGS__)
#define YK_FDK_LOGW(fmt, ...) std::printf("[FDK][ONLINE][W] " fmt "\n", ##__VA_ARGS__)
#define YK_FDK_LOGE(fmt, ...) std::printf("[FDK][ONLINE][E] " fmt "\n", ##__VA_ARGS__)
#else
#define YK_FDK_LOGI(fmt, ...)
#define YK_FDK_LOGW(fmt, ...)
#define YK_FDK_LOGE(fmt, ...)
#endif

namespace YK {

    // 你已有的 BP kernel（保持原样）
    // 这里声明一下，定义在你的 .cu/.hpp 里即可
    __global__ void fdk_vec_backproject_chunk_kernel(
        const float* __restrict__ views_chunk,            // [K*Nv*Nu]
        const SConeProjectionVec* __restrict__ d_geo,     // [Ang]
        const float* __restrict__ d_dtheta,               // [Ang]
        float* __restrict__ vol,                          // [Nz*Ny*Nx]
        int Nx, int Ny, int Nz, float vox,
        int Nu, int Nv,
        int K, int base_a);

    class FdkVecOnlineStreamer {
    public:
        FdkVecOnlineStreamer() = default;
        ~FdkVecOnlineStreamer() { release(); }

        FdkVecOnlineStreamer(const FdkVecOnlineStreamer&) = delete;
        FdkVecOnlineStreamer& operator=(const FdkVecOnlineStreamer&) = delete;

        bool init(const std::vector<SConeProjectionVec>& h_geo,
            int Nu, int Nv, int Ang,
            int Nx, int Ny, int Nz, float vox,
            int Kchunk,
            cudaStream_t stream)
        {
            release();

            h_geo_ = &h_geo;
            Nu_ = Nu; Nv_ = Nv; Ang_ = Ang;
            Nx_ = Nx; Ny_ = Ny; Nz_ = Nz; vox_ = vox;
            Kchunk_ = (Kchunk > 0 ? Kchunk : 1);
            stream_ = stream;

            if (!h_geo_ || (int)h_geo_->size() != Ang_) {
                YK_FDK_LOGE("init failed: geo.size=%zu Ang=%d", h_geo_ ? h_geo_->size() : 0ull, Ang_);
                return false;
            }
            if (Nu_ <= 0 || Nv_ <= 0 || Ang_ <= 0 || Nx_ <= 0 || Ny_ <= 0 || Nz_ <= 0) {
                YK_FDK_LOGE("init failed: invalid dims");
                return false;
            }

            // ---------------- derived ----------------
            {
                GeoDerivedManagerVec::GeoDerivedOptions opt;
                opt.offset_mode = GeoDerivedManagerVec::EOffsetMode::PerView;
                opt.isocenter = make_float3(0, 0, 0);
                opt.dtheta_eps = 1e-8f;
                if (!derived_.init(Nu_, Nv_, opt)) {
                    YK_FDK_LOGE("init failed: derived.init");
                    return false;
                }
                if (!derived_.build_geo_params(*h_geo_)) {
                    YK_FDK_LOGE("init failed: derived.build");
                    return false;
                }
            }
            const auto& gv = derived_.views();
            du0_ = derived_.du0_mm();

            // ---------------- upload geo ----------------
            YK_CUDA_CHECK(cudaMalloc(&d_geo_, (size_t)Ang_ * sizeof(SConeProjectionVec)));
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_geo_, h_geo_->data(),
                (size_t)Ang_ * sizeof(SConeProjectionVec),
                cudaMemcpyHostToDevice, stream_));

            // ---------------- upload dtheta ----------------
            {
                std::vector<float> h_dtheta(Ang_, 1e-8f);
                if ((int)gv.size() == Ang_) {
                    for (int a = 0; a < Ang_; ++a) h_dtheta[a] = gv[a].dtheta;
                }
                YK_CUDA_CHECK(cudaMalloc(&d_dtheta_, (size_t)Ang_ * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_dtheta_, h_dtheta.data(),
                    (size_t)Ang_ * sizeof(float),
                    cudaMemcpyHostToDevice, stream_));
            }

            // ---------------- managers ----------------
            pw_.init(Nu_, Nv_, 256, stream_);
            align_.init(Nu_, Nv_, stream_);
            paddedN_ = align_.paddedN();

            if (!fm_.init(Nu_, paddedN_, du0_, /*batch=*/Nv_, stream_)) {
                YK_FDK_LOGE("init failed: FilterManager.init (Nu=%d paddedN=%d du0=%f batch=%d)",
                    Nu_, paddedN_, du0_, Nv_);
                return false;
            }

            // ---------------- buffers ----------------
            view_elems_ = (size_t)Nu_ * (size_t)Nv_;

            YK_CUDA_CHECK(cudaMalloc(&d_view_in_, view_elems_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_view_pw_, view_elems_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_view_flt_, view_elems_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_padded_, (size_t)Nv_ * (size_t)paddedN_ * sizeof(float)));

            YK_CUDA_CHECK(cudaMalloc(&d_chunk_, (size_t)Kchunk_ * view_elems_ * sizeof(float)));

            is_inited_ = true;
            YK_FDK_LOGI("init ok: Nu=%d Nv=%d Ang=%d | Vol=%dx%dx%d vox=%.3f | Kchunk=%d paddedN=%d du0=%.6f stream=%p",
                Nu_, Nv_, Ang_, Nx_, Ny_, Nz_, vox_, Kchunk_, paddedN_, du0_, (void*)stream_);
            return true;
        }

        // 设置重建体的 device 指针，并清零
        void resetVolume(float* d_vol)
        {
            d_vol_ = d_vol;
            if (!d_vol_) {
                YK_FDK_LOGE("resetVolume: d_vol is null");
                return;
            }
            const size_t n = (size_t)Nx_ * (size_t)Ny_ * (size_t)Nz_;
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_, 0, n * sizeof(float), stream_));
            cur_a_ = 0;
            chunk_fill_ = 0;
            chunk_a_base_ = 0;
            bp_launch_count_ = 0;
            YK_FDK_LOGI("resetVolume: volume cleared, start online accumulation");
        }

        // 在线推送一帧 view（host ptr），内部攒 chunk，到达 Kchunk 时触发 BP
        bool push_view(int a, const float* h_view /*[Nv*Nu]*/)
        {
            if (!is_inited_) {
                YK_FDK_LOGE("push_view: not initialized");
                return false;
            }
            if (!d_vol_) {
                YK_FDK_LOGE("push_view: d_vol not set. Call resetVolume(d_vol) first.");
                return false;
            }
            if (!h_view) {
                YK_FDK_LOGE("push_view: h_view is null");
                return false;
            }
            if (a < 0 || a >= Ang_) {
                YK_FDK_LOGE("push_view: invalid a=%d (Ang=%d)", a, Ang_);
                return false;
            }

            if (chunk_fill_ == 0) chunk_a_base_ = a;

            YK_FDK_LOGI("push_view: a=%d arrived (chunk_fill=%d/%d)", a, chunk_fill_, Kchunk_);

            // 1) H2D
            YK_CUDA_CHECK(cudaMemcpyAsync(d_view_in_, h_view,
                view_elems_ * sizeof(float),
                cudaMemcpyHostToDevice, stream_));

#if YK_FDK_ONLINE_LOG
            // 你可以把这句注释掉（太吵）
            // YK_FDK_LOGI("push_view: a=%d H2D done (%zu floats)", a, view_elems_);
#endif

        // 2) preweight (K=1)
            pw_.setStream(stream_);
            pw_.applyChunk(d_view_in_, d_view_pw_, d_geo_, /*K=*/1, /*base_a=*/a);

            // 3) pad (offsetU per view)
            float offsetU = 0.0f;
            const auto& gv = derived_.views();
            if ((int)gv.size() == Ang_ && gv[a].offset_valid) offsetU = gv[a].offsetU_pix;

            align_.setStream(stream_);
            align_.pad(d_view_pw_, d_padded_, offsetU);

            // 4) filter (lazy weights)
            fm_.setStream(stream_);
            if (fm_.weightsDirty()) {
                YK_FDK_LOGI("push_view: a=%d trigger filter build (lazy)", a);
            }
            fm_.apply(d_padded_);

            // 5) crop back
            align_.crop(d_padded_, d_view_flt_);

            // 6) pack into chunk
            {
                float* d_slot = d_chunk_ + (size_t)chunk_fill_ * view_elems_;
                YK_CUDA_CHECK(cudaMemcpyAsync(d_slot, d_view_flt_,
                    view_elems_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_));
            }

            YK_FDK_LOGI("push_view: a=%d packed into chunk slot %d", a, chunk_fill_);
            chunk_fill_++;

            // 7) BP when chunk full
            if (chunk_fill_ == Kchunk_) {
                launch_bp_chunk_(chunk_a_base_, chunk_fill_);
                chunk_fill_ = 0;
            }

            cur_a_ = a + 1;
            return true;
        }

        // flush：把未满 chunk 的残余也做 BP
        void flush()
        {
            if (!is_inited_) return;
            if (!d_vol_) return;

            if (chunk_fill_ > 0) {
                YK_FDK_LOGW("flush: remaining chunk_fill=%d, launch final BP (base_a=%d)",
                    chunk_fill_, chunk_a_base_);
                launch_bp_chunk_(chunk_a_base_, chunk_fill_);
                chunk_fill_ = 0;
            }
            else {
                YK_FDK_LOGI("flush: no remaining views");
            }
        }

        void release()
        {
            if (d_geo_) { cudaFree(d_geo_); d_geo_ = nullptr; }
            if (d_dtheta_) { cudaFree(d_dtheta_); d_dtheta_ = nullptr; }

            if (d_view_in_) { cudaFree(d_view_in_);  d_view_in_ = nullptr; }
            if (d_view_pw_) { cudaFree(d_view_pw_);  d_view_pw_ = nullptr; }
            if (d_view_flt_) { cudaFree(d_view_flt_); d_view_flt_ = nullptr; }
            if (d_padded_) { cudaFree(d_padded_);   d_padded_ = nullptr; }
            if (d_chunk_) { cudaFree(d_chunk_);    d_chunk_ = nullptr; }

            h_geo_ = nullptr;
            d_vol_ = nullptr;

            Nu_ = Nv_ = Ang_ = 0;
            Nx_ = Ny_ = Nz_ = 0;
            vox_ = 1.0f;
            Kchunk_ = 1;
            paddedN_ = 0;
            du0_ = 1.0f;

            view_elems_ = 0;

            cur_a_ = 0;
            chunk_a_base_ = 0;
            chunk_fill_ = 0;
            bp_launch_count_ = 0;

            stream_ = 0;
            is_inited_ = false;
        }

    private:
        void launch_bp_chunk_(int base_a, int K)
        {
            dim3 block(8, 8, 4);
            dim3 grid((Nx_ + block.x - 1) / block.x,
                (Ny_ + block.y - 1) / block.y,
                (Nz_ + block.z - 1) / block.z);

            YK_FDK_LOGI("BP launch: base_a=%d K=%d | grid=(%u,%u,%u) block=(%u,%u,%u)",
                base_a, K,
                grid.x, grid.y, grid.z,
                block.x, block.y, block.z);

            fdk_vec_backproject_chunk_kernel << <grid, block, 0, stream_ >> > (
                d_chunk_, d_geo_, d_dtheta_, d_vol_,
                Nx_, Ny_, Nz_, vox_,
                Nu_, Nv_,
                K, base_a);
            YK_CUDA_KERNEL_CHECK();

            ++bp_launch_count_;
            YK_FDK_LOGI("BP done: launches=%d", bp_launch_count_);
        }

    private:
        // host geo ref
        const std::vector<SConeProjectionVec>* h_geo_ = nullptr;

        // dims
        int Nu_ = 0, Nv_ = 0, Ang_ = 0;
        int Nx_ = 0, Ny_ = 0, Nz_ = 0;
        float vox_ = 1.0f;

        int Kchunk_ = 1;
        int paddedN_ = 0;
        float du0_ = 1.0f;

        size_t view_elems_ = 0;

        // derived & managers
        GeoDerivedManagerVec derived_;
        PreweightManagerVec pw_;
        AlignPadCropManagerVec align_;
        FilterManager fm_;

        // device resources
        SConeProjectionVec* d_geo_ = nullptr;
        float* d_dtheta_ = nullptr;

        float* d_view_in_ = nullptr;
        float* d_view_pw_ = nullptr;
        float* d_view_flt_ = nullptr;
        float* d_padded_ = nullptr;
        float* d_chunk_ = nullptr;

        float* d_vol_ = nullptr;

        // online state
        int cur_a_ = 0;
        int chunk_a_base_ = 0;
        int chunk_fill_ = 0;
        int bp_launch_count_ = 0;

        cudaStream_t stream_ = 0;
        bool is_inited_ = false;
    };

} // namespace YK
