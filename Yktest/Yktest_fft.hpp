#pragma once
#include "Filter/YkFFT.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "YKtestconv.hpp"
#include "global/YkMacro.hpp"

namespace YKTest {
    using namespace YK;

    static inline void dump_spectrum_one_batch(
        const cufftComplex* spec_b, // 指向某一个 batch 的频谱起始
        int n_complex,
        int max_k = 32)
    {
        std::printf("k : Re             Im             |X|\n");
        for (int k = 0; k < std::min(n_complex, max_k); ++k) {
            float re = spec_b[k].x;
            float im = spec_b[k].y;
            float mag = std::sqrt(re * re + im * im);
            std::printf("%3d : % .6e  % .6e  % .6e\n", k, re, im, mag);
        }
    }

    static inline void fill_delta_batch(std::vector<float>& h, int N, int batch, int k0 = 0) {
        std::fill(h.begin(), h.end(), 0.0f);
        for (int b = 0; b < batch; ++b) {
            h[b * N + (k0 % N)] = 1.0f;
        }
    }

    static inline void fill_sine_batch(
        std::vector<float>& h,
        int N,
        int batch,
        int m,              // 频率 bin（整数，0 < m < N/2）
        float amp = 1.0f)
    {
        std::fill(h.begin(), h.end(), 0.0f);
        const float two_pi = 2.0f * float(M_PI);

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < N; ++i) {
                float phase = two_pi * m * i / float(N);
                h[b * N + i] = amp * std::sin(phase);
            }
        }
    }

    static inline bool nearly_equal(float a, float b, float rel = 1e-4f, float abs = 1e-4f) {
        float diff = std::fabs(a - b);
        if (diff <= abs) return true;
        return diff <= rel * std::max(std::fabs(a), std::fabs(b));
    }



    YK_INLINE   bool testFFT() {
        const int N = 1024;
        const int batch = 4;

        const int m = 7;              // sine frequency bin
        const int dump_batch = 0;
        const int dump_bins = 40;

        // 误差阈值：IFFT 后是 N*x
        const float tol_rel = 2e-4f;
        const float tol_abs = 2e-4f;

        std::printf("[testFFT] N=%d batch=%d sine m=%d\n", N, batch, m);

        cudaStream_t stream = 0;

        CudaFFT fft;
        if (!fft.init(N, batch, CudaFFT::EPlanMode::Both, stream)) {
            std::printf("[testFFT] fft.init failed\n");
            return false;
        }

        const int n_complex = fft.n_complex();
        const size_t real_bytes = size_t(N) * batch * sizeof(float);
        const size_t complex_bytes = size_t(n_complex) * batch * sizeof(cufftComplex);

        // host
        std::vector<float> h_in(size_t(N) * batch);
        std::vector<float> h_rec(size_t(N) * batch);
        fill_sine_batch(h_in, N, batch, m, 1.0f);

        // device
        float* d_real_in = nullptr;
        float* d_real_rec = nullptr;
        cufftComplex* d_cplx = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_real_in, real_bytes));
        YK_CUDA_CHECK(cudaMalloc(&d_real_rec, real_bytes));
        YK_CUDA_CHECK(cudaMalloc(&d_cplx, complex_bytes));

        YK_CUDA_CHECK(cudaMemcpyAsync(d_real_in, h_in.data(), real_bytes, cudaMemcpyHostToDevice, stream));
        YK_CUDA_CHECK(cudaMemsetAsync(d_real_rec, 0, real_bytes, stream));
        YK_CUDA_CHECK(cudaMemsetAsync(d_cplx, 0, complex_bytes, stream));

        // ----------------
        // FFT: R2C
        // ----------------
        fft.fft(d_real_in, d_cplx);

        // 拷回频谱并打印（可选）
        std::vector<cufftComplex> h_spec(size_t(n_complex) * batch);
        YK_CUDA_CHECK(cudaMemcpyAsync(h_spec.data(), d_cplx, complex_bytes, cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            const cufftComplex* spec_b = h_spec.data() + size_t(dump_batch) * n_complex;
            std::printf("[testFFT] Spectrum dump batch=%d (R2C k=0..N/2)\n", dump_batch);
            dump_spectrum_one_batch(spec_b, n_complex, dump_bins);
        }

        // ----------------
        // IFFT: C2R
        // ----------------
        fft.ifft(d_cplx, d_real_rec);

        // 拷回重建
        YK_CUDA_CHECK(cudaMemcpyAsync(h_rec.data(), d_real_rec, real_bytes, cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

        // ----------------
        // Check: rec ≈ N * in
        // ----------------
        bool ok = true;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        int max_flat = -1;

        for (int b = 0; b < batch; ++b) {
            const float* in = h_in.data() + size_t(b) * N;
            const float* rec = h_rec.data() + size_t(b) * N;

            for (int i = 0; i < N; ++i) {
                float expect = float(N) * in[i];
                float got = rec[i];

                float abs_err = std::fabs(got - expect);
                float rel_err = abs_err / (std::fabs(expect) + 1e-12f);

                if (abs_err > max_abs_err) { max_abs_err = abs_err; max_rel_err = rel_err; max_flat = b * N + i; }

                if (!nearly_equal(got, expect, tol_rel, tol_abs)) {
                    ok = false;
                    std::printf("[testFFT][IFFT] mismatch b=%d i=%d got=% .6e expect=% .6e abs=%g rel=%g\n",
                        b, i, got, expect, abs_err, rel_err);
                    goto DONE_CHECK;
                }
            }
        }

    DONE_CHECK:
        std::printf("[testFFT][IFFT] max_abs_err=%g max_rel_err=%g (flat=%d)\n",
            max_abs_err, max_rel_err, max_flat);
        std::printf("[testFFT] %s\n", ok ? "PASS (FFT+IFFT)" : "FAIL (FFT+IFFT)");

        YK_CUDA_CHECK(cudaFree(d_real_in));
        YK_CUDA_CHECK(cudaFree(d_real_rec));
        YK_CUDA_CHECK(cudaFree(d_cplx));

        return ok;
    }

} // namespace YKTest
