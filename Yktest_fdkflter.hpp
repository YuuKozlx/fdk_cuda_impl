#pragma once
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

#include <cuda_runtime.h>

#include "YkGlobals.h"
#include "YkFDKCreateFilterKernel.hpp"   // FilterKernelFFT, FilterKernelDesc, EFilterKernel,...

namespace YKTest {

    using namespace YK;

    static inline void dump_weights_compare(
        const float* hA, const float* hB,
        int n_complex, int N,
        int dump_bins)
    {
        std::printf("k   f=k/N     analytic              discrete              ratio(B/A)           diff(B-A)\n");
        std::printf("-------------------------------------------------------------------------------------------------\n");
        for (int k = 0; k < std::min(n_complex, dump_bins); ++k) {
            float f = (N > 0) ? (float)k / (float)N : 0.0f;
            float a = hA[k];
            float b = hB[k];
            float ratio = (std::fabs(a) > 1e-20f) ? (b / a) : 0.0f;
            float diff = b - a;
            std::printf("%4d %8.6f  % .8e  % .8e  % .8e  % .8e\n", k, f, a, b, ratio, diff);
        }
    }

    static inline void summarize_diff(
        const float* hA, const float* hB,
        int n_complex,
        int& k_max_abs, float& max_abs,
        int& k_max_rel, float& max_rel)
    {
        k_max_abs = -1; max_abs = 0.0f;
        k_max_rel = -1; max_rel = 0.0f;

        for (int k = 0; k < n_complex; ++k) {
            float a = hA[k];
            float b = hB[k];
            float abs_err = std::fabs(b - a);
            float rel_err = abs_err / (std::fabs(a) + 1e-20f);

            if (abs_err > max_abs) { max_abs = abs_err; k_max_abs = k; }
            if (rel_err > max_rel) { max_rel = rel_err; k_max_rel = k; }
        }
    }

    inline bool testFilterWeightsSpectra_RamLak()
    {
        // -----------------------------
        // Config (和你 FilterManager 同步逻辑)
        // -----------------------------
        const int Nu = 512;
        int paddedN = 1;
        while (paddedN < 2 * Nu) paddedN <<= 1;   // e.g. Nu=512 -> paddedN=1024
        const int n_complex = paddedN / 2 + 1;

        const int dump_bins = 512;

        std::printf("[testFilterWeightsSpectra] Nu=%d paddedN=%d n_complex=%d\n", Nu, paddedN, n_complex);

        cudaStream_t stream = 0;

        // -----------------------------
        // Create builder
        // -----------------------------
        FilterKernelFFT kernel;
        kernel.prepare(paddedN, stream);

        // device weights
        float* d_w_analytic = nullptr;
        float* d_w_discrete = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_w_analytic, (size_t)n_complex * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_w_discrete, (size_t)n_complex * sizeof(float)));

        // -----------------------------
        // 1) Analytic RamLak weights (your kernel_build_filter_weights_fft)
        // -----------------------------
        FilterKernelDesc desc;
        desc.kind = EFilterKernel::RamLak;
        desc.cutoff = 0.5f;             // 注意：你实现里 Nyquist=0.5
        desc.gain = 1.0f;
        desc.normalized_ramp = true;
        // 如果你的 desc 里有 force_dc_zero，按你需要开/关
        // desc.force_dc_zero = true/false;

        kernel.build_analytic(d_w_analytic, desc, /*bake_invN=*/true);

        // -----------------------------
        // 2) Discrete RamLak (DU=1) weights: spatial kernel -> FFT -> RealPart
        // -----------------------------
        kernel.build_discrete_ramlak_du1(
            d_w_discrete,
            /*bake_invN=*/true,
            FilterKernelFFT::EKernelToWeightsMode::RealPart);

        // -----------------------------
        // Copy back & dump
        // -----------------------------
        std::vector<float> hA(n_complex), hB(n_complex);

        YK_CUDA_CHECK(cudaMemcpyAsync(hA.data(), d_w_analytic, (size_t)n_complex * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaMemcpyAsync(hB.data(), d_w_discrete, (size_t)n_complex * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

        std::printf("\n[testFilterWeightsSpectra] Dump first %d bins\n", dump_bins);
        dump_weights_compare(hA.data(), hB.data(), n_complex, paddedN, dump_bins);

        int k_abs = -1, k_rel = -1;
        float max_abs = 0.0f, max_rel = 0.0f;
        summarize_diff(hA.data(), hB.data(), n_complex, k_abs, max_abs, k_rel, max_rel);

        float f_abs = (paddedN > 0 && k_abs >= 0) ? (float)k_abs / (float)paddedN : 0.0f;
        float f_rel = (paddedN > 0 && k_rel >= 0) ? (float)k_rel / (float)paddedN : 0.0f;

        std::printf("\n[testFilterWeightsSpectra] Summary:\n");
        std::printf("  max_abs_diff = %.6e at k=%d (f=%.6f)  A=%.6e  B=%.6e\n",
            max_abs, k_abs, f_abs,
            (k_abs >= 0 ? hA[k_abs] : 0.0f),
            (k_abs >= 0 ? hB[k_abs] : 0.0f));

        std::printf("  max_rel_diff = %.6e at k=%d (f=%.6f)  A=%.6e  B=%.6e\n\n",
            max_rel, k_rel, f_rel,
            (k_rel >= 0 ? hA[k_rel] : 0.0f),
            (k_rel >= 0 ? hB[k_rel] : 0.0f));

        // -----------------------------
        // Cleanup
        // -----------------------------
        YK_CUDA_CHECK(cudaFree(d_w_analytic));
        YK_CUDA_CHECK(cudaFree(d_w_discrete));
        kernel.release();

        // 这个 test 主要是打印对比，不强行判 PASS/FAIL
        // 你也可以根据 max_abs/max_rel 自己设阈值
        std::printf("[testFilterWeightsSpectra] DONE\n");
        return true;
    }

} // namespace YKTest
